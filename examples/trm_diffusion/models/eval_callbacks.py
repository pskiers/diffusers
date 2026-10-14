"""
models/eval_callbacks.py — Pluggable eval callbacks.

Callbacks have the signature:
    callback(model, dataloader, accelerator, **kwargs) -> dict[str, float]

They are invoked from eval_step after generic loss metrics are computed.
Models accept a list of callbacks and merge all returned dicts.

Sampling parameters (num_inference_steps, cfg_scale, batch_size) are owned
by model.sampling_pipeline rather than individual callbacks.  Callbacks own
only evaluation-specific knobs: num_samples, num_log_images, cell_size, etc.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import wandb as _wandb
from eval.mnist_eval import (
    evaluate_grids,
    extract_and_resize_sudoku,
    load_or_train_classifier,
    make_panel_image,
    plot_thinker_ts_curve,
    sample_grids,
)
from eval.maze_eval import evaluate_mazes, make_maze_panel_image
from eval.steiner_eval import evaluate_steiner, make_steiner_panel_image
from eval.polygon_eval import evaluate_polygon, make_polygon_panel_image, _orders_equivalent
from eval.ball_drop_eval import evaluate_ball_drop, make_ball_drop_panel_image
from eval.squares_eval import evaluate_squares, make_squares_panel_image
from eval.topodiff_eval import evaluate_topodiff, make_topodiff_panel_image
from datasets.data_sample import DataSample


def _select_best_of_n(valid: np.ndarray, objective: np.ndarray, maximize: bool):
    """Per-instance best-of-N selection, matching the geometric-solver
    paper's evaluation protocol (arXiv 2510.21697): among the N independent
    candidates generated for one instance, discard invalid ones and keep the
    single best by objective (max area / min length). An instance with zero
    valid candidates is excluded from the ratio (any_valid=False).

    Args:
        valid, objective: (B, N) arrays.

    Returns:
        any_valid: (B,) bool — instance had >=1 valid candidate.
        best_val:  (B,) float — objective of the best valid candidate (NaN if none).
        best_idx:  (B,) int — index in [0, N) of the best valid candidate (-1 if none).
    """
    B, N = valid.shape
    obj = objective.astype(np.float64).copy()
    obj[~valid] = -np.inf if maximize else np.inf
    best_idx = obj.argmax(axis=1) if maximize else obj.argmin(axis=1)
    any_valid = valid.any(axis=1)
    best_val = obj[np.arange(B), best_idx]
    best_val = np.where(any_valid, best_val, np.nan)
    best_idx = np.where(any_valid, best_idx, -1)
    return any_valid, best_val, best_idx


def _weighted_mean(pairs: list) -> float:
    """pairs: list of (value, weight). Returns NaN for an empty list."""
    if not pairs:
        return float("nan")
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return float("nan")
    return float(sum(v * w for v, w in pairs) / total_w)


class EvalCallbackBase:
    """Abstract base for eval callbacks."""

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        raise NotImplementedError


# ── CLEVR / latent-DiT ────────────────────────────────────────────────────────


class ImageGenEvalCallback(EvalCallbackBase):
    """
    DDIM sampling + side-by-side WandB logging for latent painters (CLEVR etc.).

    Uses model.sampling_pipeline for all sampling parameters.  Reads from the
    model at call time:
      model.sampling_pipeline
      model.noise_shape     — (C, H, W) tuple, no batch dim
      model.decode_for_eval(z) -> Tensor in [0, 1]
      model.condition_keys   — DataSample fields to pull from batch
      model.null_condition_sample(sample) -> DataSample

    Args:
        num_log_images: number of images to log to WandB.
    """

    def __init__(self, num_log_images: int = 8):
        self.num_log_images = num_log_images

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}

        step = kwargs.get("step", None)
        device = accelerator.device
        n_log = self.num_log_images
        pipeline = model.sampling_pipeline

        sample_images, gt_images = [], []
        for batch in dataloader:
            if len(sample_images) >= n_log:
                break
            B = min(batch["images"].shape[0], n_log - len(sample_images))

            sample_kwargs = {}
            for k in model.condition_keys:
                val = batch.get(k) if hasattr(batch, "get") else batch.get(k, None)
                if val is not None:
                    sample_kwargs[k] = val[:B].to(device)
            emb_mask = batch.get("embedding_mask") if hasattr(batch, "get") else None
            if emb_mask is not None:
                sample_kwargs["embedding_mask"] = emb_mask[:B].to(device)

            base_sample = DataSample(**sample_kwargs)
            latents = pipeline.sample_one_batch(model, base_sample, device)
            imgs = model.decode_for_eval(latents)
            sample_images.extend(imgs.cpu().unbind(0))

            gt_raw = batch["images"][:B]
            gt_images.extend(model.images_to_log(gt_raw).cpu().unbind(0))

        if sample_images and step is not None:
            try:
                combined = []
                for gt, s in zip(gt_images, sample_images):
                    combined.extend([gt, s])
                grid = make_grid(torch.stack(combined), nrow=4)
                grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                _wandb.log({"val/samples": _wandb.Image(grid_np)}, step=step)
            except Exception:
                pass

        return {}


class T2ISampleLogCallback(EvalCallbackBase):
    """
    Text-to-image: samples images for the first `num_log_images` captions of
    the eval set (token_conditions / token_mask) with model.sampling_pipeline,
    from fixed noise (`seed`) so successive evals are comparable, and logs each
    next to its reference image with its prompt: to WandB as one captioned
    image per prompt, and to `out_dir` (if set) as one PNG with the prompts
    written under the pairs. The prompt is decoded from the token ids, so it is
    exactly what the model saw (truncated to the prompt length).

    Args:
        num_log_images: captions to sample.
        seed: noise seed.
        out_dir: optional directory for samples_step<step>.png grids.
        tokenizer: tokenizer the token ids came from.
    """

    def __init__(
        self,
        num_log_images: int = 8,
        seed: int = 0,
        out_dir: Optional[str] = None,
        tokenizer: str = "google/flan-t5-large",
    ):
        self.num_log_images = num_log_images
        self.seed = seed
        self.out_dir = out_dir
        self.tokenizer_name = tokenizer
        self._tokenizer = None

    def _decode(self, token_ids: torch.Tensor) -> list[str]:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return [p.strip() for p in self._tokenizer.batch_decode(token_ids, skip_special_tokens=True)]

    @staticmethod
    def _caption_lines(text: str, width_px: int, max_lines: int = 8) -> list[str]:
        import textwrap

        lines = textwrap.wrap(text or "(empty prompt)", width=width_px // 9)
        if len(lines) > max_lines:
            lines = lines[: max_lines - 1] + [lines[max_lines - 1][:-3] + "..."]
        return lines

    @staticmethod
    def _with_caption(image: np.ndarray, lines: list[str], n_lines: int) -> np.ndarray:
        """`image` (H, W, 3) uint8 with `lines` on a white strip of n_lines below it."""
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default(size=16)
        line_h = 20
        canvas = Image.new("RGB", (image.shape[1], image.shape[0] + line_h * n_lines + 8), "white")
        canvas.paste(Image.fromarray(image), (0, 0))
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(lines):
            draw.text((6, image.shape[0] + 4 + i * line_h), line, fill="black", font=font)
        return np.asarray(canvas)

    @torch.no_grad()
    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        step = kwargs.get("step", None)
        device = accelerator.device
        pipeline = model.sampling_pipeline

        token_ids, token_mask, refs = [], [], []
        for batch in dataloader:
            take = self.num_log_images - sum(len(t) for t in token_ids)
            if take <= 0:
                break
            token_ids.append(batch.token_conditions[:take])
            token_mask.append(batch.token_mask[:take])
            refs.append(batch.images[:take])
        conditions = DataSample(token_conditions=torch.cat(token_ids), token_mask=torch.cat(token_mask)).to(device)

        generator = torch.Generator(device=device).manual_seed(self.seed)
        samples = []
        for start in range(0, len(conditions.token_conditions), pipeline.batch_size):
            chunk = conditions.slice(start, start + pipeline.batch_size)
            samples.append(pipeline.sample_one_batch(model, chunk, device, generator=generator))
        samples = model.decode_for_eval(torch.cat(samples)).cpu()
        refs = model.images_to_log(torch.cat(refs)).cpu()
        prompts = self._decode(conditions.token_conditions.cpu())

        # One (reference | sample) image per prompt.
        pairs = [
            (torch.cat([ref, s], dim=2).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            for ref, s in zip(refs, samples)
        ]
        if self.out_dir is not None:
            from PIL import Image

            lines = [self._caption_lines(t, p.shape[1]) for p, t in zip(pairs, prompts)]
            n_lines = max(len(ls) for ls in lines)  # one strip height for the whole grid
            captioned = torch.from_numpy(np.stack([self._with_caption(p, ls, n_lines) for p, ls in zip(pairs, lines)]))
            grid = make_grid(captioned.permute(0, 3, 1, 2), nrow=2, padding=8, pad_value=255)
            os.makedirs(self.out_dir, exist_ok=True)
            Image.fromarray(grid.permute(1, 2, 0).numpy()).save(os.path.join(self.out_dir, f"samples_step{step}.png"))
        if step is not None:
            try:
                _wandb.log(
                    {"val/samples": [_wandb.Image(p, caption=f"reference | sample — {t}") for p, t in zip(pairs, prompts)]},
                    step=step,
                )
            except Exception:
                pass
        return {}


class CoReBenchEvalCallback(EvalCallbackBase):
    """
    T2I-CoReBench during training (see eval/corebench_eval.py,
    eval/corebench_judge.py). Every `every_n_steps` global steps:

      1. every rank generates its share of the 1,080 prompts x 4 images;
      2. judging is split over every node: on each node, local rank 0 starts
         `judges_per_node` VLM judges (`judge_python`: an interpreter of the
         separate judge environment), each on its own
         `judge_tensor_parallel_size` GPUs and its own shard of the prompts
         (round-robin, all of a prompt's images together). Ranks wait on
         marker files, not in a collective, which would time out over a
         judging run;
      3. rank 0 merges the shards with the protocol's own scoring (see
         eval/corebench_eval.py: merge_judge_shards — the same scores one judge
         over everything gives) and returns corebench/{overall, composition,
         reasoning, mean_of_dimensions, <12 dimensions>} (0-100); the images
         are then deleted unless keep_images.

    A judge failure is logged (out_dir/step*/judge/shard*/judge.log) and returns
    no metrics; it never stops training. Set eval.step_timeout to cover a whole
    run (generation + judging).

    Also runs from eval.py (all ranks: `accelerate launch eval.py ...`): with
    a checkpoint whose step is at least every_n_steps (or every_n_steps=0),
    and always without a checkpoint (as step 0).

    Args:
        every_n_steps: run when the eval step reaches the next multiple.
        out_dir: root for step<N>/{images,judge,judged.json}; must be on a
            filesystem every rank (every node) can see.
        protocol: "looped_dit" (Looped-DiT's judging, default) or "official".
        judge_python: Python of the judge environment; None skips the whole
            evaluation (with a warning), since its images could not be scored.
        judge_model: looped_dit: HF id (default Qwen/Qwen3-VL-32B-Thinking);
            official: their judge key (default Qwen3_VL_30B_A3B_Thinking).
        judge_tensor_parallel_size: GPUs per judge (a divisor of 8). 2 fits
            either default judge on 80 GB GPUs; 24 GB GPUs need 4 or 8.
        judges_per_node: judges per node, each on its own GPUs; None = as many
            as the node's GPUs allow.
        gpu_memory_utilization: vLLM's GPU memory share; the paused training
            processes keep a few GB per GPU.
        judge_args: extra flags for eval/corebench_judge.py.
        judge_env: extra environment variables for the judges only (e.g.
            HF_HOME for where their ~66 GB of weights live).
        samples_per_prompt, steps, cfg_scale, seed: generation (Looped-DiT's:
            4, 100 Euler steps, CFG 6.0).
        batch_size: images per sampling batch, per GPU.
        tokenizer, prompt_length: the text encoder's tokenizer.
        keep_images: keep the 4,320 PNGs after judging.
        model_name: image directory name for the official protocol.
        prompts_per_dimension: evaluate only the first k prompts of each
            dimension (e.g. a quick check, or cheaper frequent evals); scores
            on a subset are noisier and not comparable to full-benchmark ones.
            None = all 90.
    """

    def __init__(
        self,
        every_n_steps: int,
        out_dir: str,
        protocol: str = "looped_dit",
        judge_python: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_tensor_parallel_size: int = 2,
        judges_per_node: Optional[int] = None,
        gpu_memory_utilization: float = 0.85,
        judge_args: Optional[list] = None,
        judge_env: Optional[dict] = None,
        samples_per_prompt: int = 4,
        steps: int = 100,
        cfg_scale: float = 6.0,
        seed: int = 0,
        batch_size: int = 16,
        tokenizer: str = "google/flan-t5-large",
        prompt_length: int = 256,
        keep_images: bool = False,
        model_name: str = "model",
        prompts_per_dimension: Optional[int] = None,
        poll_seconds: float = 20.0,
    ):
        from eval.corebench_eval import PROTOCOLS

        if protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
        self.every_n_steps = every_n_steps
        self.out_dir = out_dir
        self.protocol = protocol
        self.judge_python = judge_python
        self.judge_model = judge_model
        self.judge_tensor_parallel_size = judge_tensor_parallel_size
        self.judges_per_node = judges_per_node
        self.gpu_memory_utilization = gpu_memory_utilization
        self.judge_args = [str(a) for a in (judge_args or [])]  # Hydra parses e.g. [--batch-size,1024] into ints
        self.judge_env = {str(k): str(v) for k, v in (judge_env or {}).items()}
        self.gen_kwargs = dict(samples_per_prompt=samples_per_prompt, steps=steps, cfg_scale=cfg_scale, seed=seed,
                               batch_size=batch_size, model_name=model_name)
        self.tokenizer_name = tokenizer
        self.prompt_length = prompt_length
        self.keep_images = keep_images
        self.prompts_per_dimension = prompts_per_dimension
        self.poll_seconds = poll_seconds
        self._next_step = every_n_steps
        self._warned = False

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        step = kwargs.get("step", None)
        if step is None:  # eval.py without a checkpoint: run now
            step = 0
        elif step < self._next_step:
            return {}
        else:
            self._next_step = step + self.every_n_steps
        if self.judge_python is None:
            if not self._warned and accelerator.is_main_process:
                print("[CoReBench] judge_python not set (e.g. COREBENCH_JUDGE_PYTHON) — skipping CoReBench eval", flush=True)
            self._warned = True
            return {}
        return self.run(model, accelerator, step)

    def _judge_layout(self, num_processes: int) -> tuple[int, int, list[list[str]]]:
        """(nodes, judges per node, each local judge's GPU ids). Processes
        are spread evenly over nodes, as torchrun/accelerate launch place them."""
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", num_processes))
        nodes = max(1, num_processes // local_world)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpus = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
        tp = self.judge_tensor_parallel_size
        per_node = self.judges_per_node or len(gpus) // tp
        if per_node < 1 or per_node * tp > len(gpus):
            raise ValueError(f"{per_node} judge(s) x {tp} GPUs do not fit the node's {len(gpus)} GPUs "
                             f"— set judge_tensor_parallel_size / judges_per_node")
        return nodes, per_node, [gpus[j * tp : (j + 1) * tp] for j in range(per_node)]

    def _judge_command(self, image_dir, results_dir, gpus, data_dir=None, tasks=None) -> tuple[list, dict]:
        script = Path(__file__).resolve().parents[1] / "eval" / "corebench_judge.py"
        cmd = [self.judge_python, str(script), "--protocol", self.protocol, "--image-dir", str(image_dir),
               "--out-dir", str(results_dir), "--samples-per-prompt", str(self.gen_kwargs["samples_per_prompt"]),
               "--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        if self.protocol == "looped_dit":
            cmd += ["--tensor-parallel-size", str(len(gpus)), "--data", str(data_dir)]
            cmd += ["--judge-model", self.judge_model] if self.judge_model else []
        else:  # official: tensor parallelism = all visible GPUs, as in its code
            cmd += ["--model-name", self.gen_kwargs["model_name"], "--tasks", *tasks]
            cmd += ["--mllm", self.judge_model] if self.judge_model else []
        env = {**os.environ, **self.judge_env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
        return cmd + self.judge_args, env

    def run(self, model, accelerator, step: int) -> dict:
        """One full evaluation, labelled `step`, regardless of every_n_steps."""
        import json
        import shutil
        import subprocess
        import time

        from transformers import AutoTokenizer

        from eval.corebench_eval import (
            corebench_prompts,
            generate_corebench_images,
            link_shard_images,
            merge_judge_shards,
            summarize_scores,
            write_prompt_files,
        )

        rank, world = accelerator.process_index, accelerator.num_processes
        root = Path(self.out_dir) / f"step{step:08d}"
        image_dir, done = root / "images", root / "judged.json"
        prompts = corebench_prompts(self.prompts_per_dimension)
        nodes, per_node, judge_gpus = self._judge_layout(world)
        num_shards = nodes * per_node
        local_world = world // nodes
        node, is_node_leader = rank // local_world, rank % local_world == 0
        spp = self.gen_kwargs["samples_per_prompt"]

        def shard_dir(s):
            return root / "judge" / f"shard{s}"

        def wait_for(paths):
            while not all(p.exists() for p in paths):
                time.sleep(self.poll_seconds)

        def write_json(path, obj):  # atomic: other ranks read it as soon as it exists
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(obj, indent=2))
            os.replace(tmp, path)

        if not done.exists():
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, model_max_length=self.prompt_length)
            t0 = time.time()
            n = generate_corebench_images(model, image_dir, tokenizer, accelerator.device, self.protocol, rank, world,
                                          prompts=prompts, **self.gen_kwargs)
            seconds = time.time() - t0
            print(f"[CoReBench rank {rank}] generated {n} images in {seconds / 60:.1f} min", flush=True)
            write_json(root / f".generated.rank{rank}", {"images": n, "seconds": seconds})
            torch.cuda.empty_cache()  # leave the judges as much GPU memory as possible

            if is_node_leader:  # this node's judges, on this node's GPUs, each on its shard
                wait_for([root / f".generated.rank{r}" for r in range(world)])
                jobs = []
                for j, gpus in enumerate(judge_gpus):
                    s = node * per_node + j
                    shard, d = prompts[s::num_shards], shard_dir(s)
                    d.mkdir(parents=True, exist_ok=True)
                    if self.protocol == "looped_dit":
                        write_prompt_files(shard, d / "prompts")
                        cmd, env = self._judge_command(image_dir, d / "results", gpus, data_dir=d / "prompts")
                    else:
                        link_shard_images(shard, image_dir, d / "images", spp, self.gen_kwargs["model_name"])
                        tasks = sorted({p.group for p in shard})
                        cmd, env = self._judge_command(d / "images", d / "results", gpus, tasks=tasks)
                    log = open(d / "judge.log", "w")
                    judgments = spp * sum(len(p.metadata["Checklist"]) for p in shard)
                    jobs.append((s, judgments, log, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
                for s, judgments, log, proc in jobs:
                    code = proc.wait()
                    log.close()
                    write_json(shard_dir(s) / "done.json", {"returncode": code, "judgments": judgments})

            if accelerator.is_main_process:
                wait_for([shard_dir(s) / "done.json" for s in range(num_shards)])
                shards = [json.loads((shard_dir(s) / "done.json").read_text()) for s in range(num_shards)]
                failed = [s for s, sh in enumerate(shards) if sh["returncode"] != 0]
                outcome = {}
                if failed:
                    outcome["error"] = f"judge shard(s) {failed} failed, see {shard_dir(failed[0]) / 'judge.log'}"
                else:
                    try:
                        outcome["scores"] = merge_judge_shards(self.protocol, [shard_dir(s) / "results" for s in range(num_shards)])
                    except Exception as e:  # e.g. too many unparseable judgments in total
                        outcome["error"] = f"merging judge shards failed: {e}"
                for s, sh in enumerate(shards):
                    timing_file = shard_dir(s) / "results" / "scores.json"
                    sh["timing"] = json.loads(timing_file.read_text()).get("timing", {}) if timing_file.exists() else {}
                outcome["judge_shards"] = shards
                outcome["generation"] = [json.loads((root / f".generated.rank{r}").read_text()) for r in range(world)]
                write_json(done, outcome)
            else:
                wait_for([done])

        outcome = json.loads(done.read_text())
        if not accelerator.is_main_process:
            return {}
        if "error" in outcome:
            print(f"[CoReBench] step {step}: {outcome['error']}", flush=True)
            return {}
        if not self.keep_images:
            shutil.rmtree(image_dir, ignore_errors=True)
            for shard_images in (root / "judge").glob("shard*/images"):  # official: the shards' symlink views
                shutil.rmtree(shard_images, ignore_errors=True)
        metrics = summarize_scores(outcome["scores"])
        print(f"[CoReBench] step {step}: " + "  ".join(f"{k}={v:.1f}" for k, v in metrics.items()), flush=True)
        return {f"corebench/{k}": v for k, v in metrics.items()}


# ── MNIST Sudoku ──────────────────────────────────────────────────────────────


class SudokuDDIMEvalCallback(EvalCallbackBase):
    """
    DDIM sampling eval for MNIST Sudoku models with full thinker trajectory
    tracking.  Sampling parameters come from model.sampling_pipeline.

    Runs sample_grids with the thinker condition, evaluates with a digit
    classifier, and logs WandB panels + the thinker-timestep accuracy curve.

    Returns: cell_acc, puzzle_acc and (when include_thinker_metrics=True)
             thinker_cell_acc_best/mean, thinker_puzzle_acc_best/mean,
             thinker_deviation_from_best, painter_dev_from_best/mean_thinker.

    Args:
        classifier_path: path passed to load_or_train_classifier. If None the
                         callback is a no-op.
        cell_size: pixel size of each Sudoku cell.
        include_thinker_metrics: set False to skip thinker accuracy metrics.
        num_samples: total number of samples to evaluate.
        num_log_images: number of images to log to WandB.
        painter_size: pixel size of the full painter image. If None, derived
                      as cell_size * 9.
    """

    def __init__(
        self,
        classifier_path: str | None = None,
        cell_size: int = 16,
        include_thinker_metrics: bool = True,
        num_samples: int = 1000,
        num_log_images: int = 8,
        painter_size: int | None = None,
    ):
        self.cell_size = cell_size
        self.include_thinker_metrics = include_thinker_metrics
        self.num_samples = num_samples
        self.num_log_images = num_log_images
        self.painter_size = painter_size if painter_size is not None else cell_size * 9
        self.eval_clf = None
        if classifier_path is not None:
            self.eval_clf = load_or_train_classifier(classifier_path, None, cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if self.eval_clf is None or not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "SudokuDDIMEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        painter_size = self.painter_size
        n_total = self.num_samples
        n_log = self.num_log_images
        token_offset = getattr(model, "token_offset", 0)

        all_cell_acc, all_puzzle_acc, all_constraint_acc, all_given_consistent_acc = [], [], [], []
        all_thinker_cell_best, all_thinker_cell_mean = [], []
        all_thinker_puzzle_best, all_thinker_puzzle_mean = [], []
        all_thinker_deviation = []
        all_painter_dev_best, all_painter_dev_mean = [], []
        ts_cell_accs: dict[int, list] = {}
        ts_puzzle_accs: dict[int, list] = {}
        panels: list = []
        n_done = 0

        n_batches = (n_total + pipeline.batch_size - 1) // pipeline.batch_size
        for batch in tqdm(dataloader, "Sampling eval", total=n_batches):
            if n_done >= n_total:
                break
            solutions = batch["solution"]
            given_masks = batch.get("solution_mask")
            B_cur = solutions.shape[0]

            sr = sample_grids(
                model,
                model._batch_to_sample(batch, device),
                num_train_timesteps=model.scheduler.config.num_train_timesteps,
                beta_schedule=model.scheduler.config.beta_schedule,
                prediction_type=model.scheduler.config.prediction_type,
                num_steps=pipeline.num_inference_steps,
                device=device,
                solutions=solutions,
                painter_size=painter_size,
                given_masks=given_masks,
                cfg_scale=pipeline.cfg_scale,
            )
            acc = evaluate_grids(sr["generated"], solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
            all_cell_acc.append(acc["cell_acc"])
            all_puzzle_acc.append(acc["puzzle_acc"])
            all_constraint_acc.append(acc.get("constraint_puzzle_acc", 0.0))
            if acc.get("given_consistent_puzzle_acc") is not None:
                all_given_consistent_acc.append(acc["given_consistent_puzzle_acc"])

            if self.include_thinker_metrics:
                for key, lst in [
                    ("thinker_cell_acc_best", all_thinker_cell_best),
                    ("thinker_cell_acc_mean", all_thinker_cell_mean),
                    ("thinker_puzzle_acc_best", all_thinker_puzzle_best),
                    ("thinker_puzzle_acc_mean", all_thinker_puzzle_mean),
                    ("thinker_deviation_from_best", all_thinker_deviation),
                ]:
                    if key in sr:
                        lst.append(sr[key])

                for t_step, a in sr.get("ts_cell_acc", []):
                    ts_cell_accs.setdefault(t_step, []).append(a)
                for t_step, a in sr.get("ts_puzzle_acc", []):
                    ts_puzzle_accs.setdefault(t_step, []).append(a)

                painter_preds = acc["preds"]
                _gm = given_masks[:B_cur].cpu() if given_masks is not None else None
                for tp_raw, dev_lst in [
                    (sr.get("best_thinker_preds"), all_painter_dev_best),
                    (sr.get("mean_thinker_preds"), all_painter_dev_mean),
                ]:
                    if tp_raw is not None:
                        tp = tp_raw - token_offset
                        N = tp.shape[1]
                        if N > painter_preds.shape[1]:
                            continue
                        diff = painter_preds[:, :N] != tp
                        if _gm is not None:
                            blank = ~_gm[:, :N]
                            dev = diff[blank].float().mean().item() if blank.any() else diff.float().mean().item()
                        else:
                            dev = diff.float().mean().item()
                        dev_lst.append(dev)

            if _wandb is not None and len(panels) < n_log:
                n_new = min(n_log - len(panels), B_cur)
                _sc = batch.get("spatial_conditions")
                conds_vis = _sc if _sc is not None else batch.get("token_conditions")
                if conds_vis is not None and conds_vis.dim() == 4:
                    conds_vis = conds_vis.cpu()
                tp_all = sr.get("best_thinker_preds")
                tt_all = sr.get("best_thinker_ts")
                if tp_all is not None and tp_all.shape[1] != 81:
                    tp_all = None
                sols_np = solutions.cpu().numpy()
                for i in range(n_new):
                    tp = (tp_all[i] - token_offset).numpy() if tp_all is not None else None
                    tt = tt_all[i] if tp_all is not None and tt_all is not None else None
                    cond_img = conds_vis[i] if (conds_vis is not None and conds_vis.dim() == 4) else None
                    panel = make_panel_image(cond_img, sr["generated"][i], sols_np[i], thinker_preds=tp, thinker_t=tt)
                    panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_cur

        result: dict = {
            "cell_acc": float(np.mean(all_cell_acc)),
            "puzzle_acc": float(np.mean(all_puzzle_acc)),
            "constraint_puzzle_acc": float(np.mean(all_constraint_acc)),
        }
        if all_given_consistent_acc:
            result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent_acc))
        if self.include_thinker_metrics:
            if all_thinker_cell_best:
                result["thinker_cell_acc_best"] = float(np.mean(all_thinker_cell_best))
                result["thinker_cell_acc_mean"] = float(np.mean(all_thinker_cell_mean))
                result["thinker_puzzle_acc_best"] = float(np.mean(all_thinker_puzzle_best))
                result["thinker_puzzle_acc_mean"] = float(np.mean(all_thinker_puzzle_mean))
                result["thinker_deviation_from_best"] = float(np.mean(all_thinker_deviation))
            if all_painter_dev_best:
                result["painter_dev_from_best_thinker"] = float(np.mean(all_painter_dev_best))
                result["painter_dev_from_mean_thinker"] = float(np.mean(all_painter_dev_mean))
        if panels:
            result["samples"] = panels
        if ts_cell_accs and _wandb is not None:
            result["thinker_vs_timestep"] = _wandb.Image(plot_thinker_ts_curve(ts_cell_accs, ts_puzzle_accs))

        return result


class SudokuRealSolutionCallback(EvalCallbackBase):
    """
    Oracle eval for MNIST Sudoku: DDIM sampling with real solution tokens as the
    condition (teacher forcing) to establish an accuracy upper bound.
    Sampling parameters come from model.sampling_pipeline.

    Only runs when model.has_realsolution_eval is True.

    Returns: real_cell_acc, real_puzzle_acc

    Args:
        classifier_path: same semantics as SudokuDDIMEvalCallback.
        cell_size: pixel size of each Sudoku cell.
        num_samples: total number of samples to evaluate.
        painter_size: pixel size of the full painter image. If None, derived
                      as cell_size * 9.
    """

    def __init__(
        self,
        classifier_path: str | None = None,
        cell_size: int = 16,
        num_samples: int = 1000,
        painter_size: int | None = None,
    ):
        self.cell_size = cell_size
        self.num_samples = num_samples
        self.painter_size = painter_size if painter_size is not None else cell_size * 9
        self.eval_clf = None
        if classifier_path is not None:
            self.eval_clf = load_or_train_classifier(classifier_path, None, cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not getattr(model, "has_realsolution_eval", False):
            return {}
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "SudokuRealSolutionCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        painter_size = self.painter_size
        n_total = self.num_samples

        all_real_cell, all_real_puzzle = [], []
        all_real_constraint, all_real_given_consistent = [], []
        n_real = 0
        n_batches = (n_total + pipeline.batch_size - 1) // pipeline.batch_size

        for batch in tqdm(dataloader, "Realsolution eval", total=n_batches):
            if n_real >= n_total:
                break
            solutions = batch["solution"]
            given_masks = batch.get("solution_mask")

            sr_r = sample_grids(
                model,
                model._batch_to_sample(batch, device),
                num_train_timesteps=model.scheduler.config.num_train_timesteps,
                beta_schedule=model.scheduler.config.beta_schedule,
                prediction_type=model.scheduler.config.prediction_type,
                num_steps=pipeline.num_inference_steps,
                device=device,
                solutions=solutions,
                painter_size=painter_size,
                given_masks=given_masks,
                cfg_scale=pipeline.cfg_scale,
            )
            acc_r = evaluate_grids(sr_r["generated"], solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
            all_real_cell.append(acc_r["cell_acc"])
            all_real_puzzle.append(acc_r["puzzle_acc"])
            all_real_constraint.append(acc_r.get("constraint_puzzle_acc", 0.0))
            if acc_r.get("given_consistent_puzzle_acc") is not None:
                all_real_given_consistent.append(acc_r["given_consistent_puzzle_acc"])
            n_real += solutions.shape[0]

        result = {
            "real_cell_acc": float(np.mean(all_real_cell)),
            "real_puzzle_acc": float(np.mean(all_real_puzzle)),
            "real_constraint_puzzle_acc": float(np.mean(all_real_constraint)),
        }
        if all_real_given_consistent:
            result["real_given_consistent_puzzle_acc"] = float(np.mean(all_real_given_consistent))
        return result


class SudokuEvalCallback(EvalCallbackBase):
    """
    Simple DDIM sampling eval for MNIST Sudoku: sample images then evaluate.

    No thinker trajectory tracking.  Uses model.sampling_pipeline for all
    sampling parameters.  Logs WandB panels (puzzle | generated | solution).

    Returns: cell_acc, puzzle_acc

    Args:
        classifier_path: path passed to load_or_train_classifier. If None the
                         callback is a no-op.
        cell_size: pixel size of each Sudoku cell.
        num_samples: total number of samples to evaluate.
        num_log_images: number of panel images to log to WandB.
        painter_size: pixel size of the full painter image. If None, derived
                      as cell_size * 9.
    """

    def __init__(
        self,
        classifier_path: str | None = None,
        cell_size: int = 16,
        num_samples: int = 1000,
        num_log_images: int = 8,
        painter_size: int | None = None,
    ):
        self.cell_size = cell_size
        self.num_samples = num_samples
        self.num_log_images = num_log_images
        self.painter_size = painter_size if painter_size is not None else cell_size * 9
        self.eval_clf = None
        if classifier_path is not None:
            self.eval_clf = load_or_train_classifier(classifier_path, None, cell_size, "cuda")
            for p in self.eval_clf.parameters():
                p.requires_grad_(False)

    def _prepare_for_eval(self, generated: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses to preprocess generated images before classification.

        Only affects the accuracy computation — logged panel images always show
        the raw model output.
        """
        return generated

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if self.eval_clf is None or not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "SudokuEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_total = self.num_samples
        n_log = self.num_log_images
        token_offset = getattr(model, "token_offset", 0)

        all_cell_acc, all_puzzle_acc, all_constraint_acc, all_given_consistent_acc = [], [], [], []
        panels: list = []
        n_done = 0

        n_batches = (n_total + pipeline.batch_size - 1) // pipeline.batch_size
        for batch in tqdm(dataloader, "Sudoku eval", total=n_batches):
            if n_done >= n_total:
                break
            solutions = batch["solution"]
            given_masks = batch.get("solution_mask")
            B_cur = solutions.shape[0]

            conditions = model._batch_to_sample(batch, device)
            generated = pipeline.sample_one_batch(model, conditions, device)
            generated = model.decode_for_eval(generated)  # (B, 1, H, W) in [0, 1]
            eval_images = self._prepare_for_eval(generated)

            acc = evaluate_grids(eval_images, solutions, self.eval_clf, self.cell_size, given_masks=given_masks)
            all_cell_acc.append(acc["cell_acc"])
            all_puzzle_acc.append(acc["puzzle_acc"])
            all_constraint_acc.append(acc.get("constraint_puzzle_acc", 0.0))
            if acc.get("given_consistent_puzzle_acc") is not None:
                all_given_consistent_acc.append(acc["given_consistent_puzzle_acc"])

            if _wandb is not None and len(panels) < n_log:
                n_new = min(n_log - len(panels), B_cur)
                _sc = batch.get("spatial_conditions")
                conds_vis = _sc if _sc is not None else batch.get("token_conditions")
                if conds_vis is not None and conds_vis.dim() == 4:
                    conds_vis = conds_vis.cpu()
                sols_np = solutions.cpu().numpy()
                gen_np = generated.cpu()
                for i in range(n_new):
                    cond_img = conds_vis[i] if (conds_vis is not None and conds_vis.dim() == 4) else None
                    panel = make_panel_image(cond_img, gen_np[i], sols_np[i])
                    panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_cur

        result: dict = {
            "cell_acc": float(np.mean(all_cell_acc)),
            "puzzle_acc": float(np.mean(all_puzzle_acc)),
            "constraint_puzzle_acc": float(np.mean(all_constraint_acc)),
        }
        if all_given_consistent_acc:
            result["given_consistent_puzzle_acc"] = float(np.mean(all_given_consistent_acc))
        if panels:
            result["samples"] = panels

        return result


class ScaledSudokuEvalCallback(SudokuEvalCallback):
    """SudokuEvalCallback variant for MNISTSudokuScaledDataset targets.

    There, the solved grid is pasted onto a black canvas at a random
    scale/offset, so the generated image can't be evaluated by unfolding it
    into 9×9 cells directly — the true scale/position isn't known at
    generation time. Crops each generated image to its non-background content
    bounding box and resizes back to painter_size before running the standard
    cell classifier.

    Args (in addition to SudokuEvalCallback's):
        bbox_threshold: pixel intensity above which a pixel counts as content
                        when finding the crop bounding box.
    """

    def __init__(self, *args, bbox_threshold: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.bbox_threshold = bbox_threshold

    def _prepare_for_eval(self, generated: torch.Tensor) -> torch.Tensor:
        return extract_and_resize_sudoku(generated, self.painter_size, self.bbox_threshold)


# ── Maze ──────────────────────────────────────────────────────────────────────


class MazeEvalCallback(EvalCallbackBase):
    """
    DDIM sampling eval for Maze models: sample images, then evaluate against
    the maze's own wall structure. No learned classifier is needed — unlike
    Sudoku's MNIST digit cells, MazeDataset renders every cell as a flat
    color from a small fixed palette, so cells are classified by nearest-
    color matching (see eval/maze_eval.py).

    Returns: cell_acc, puzzle_acc, constraint_puzzle_acc (see
    eval.maze_eval.evaluate_mazes for exact definitions).

    Args:
        grid_size, cell_size: must match the dataset's rendering parameters.
        num_samples: total number of samples to evaluate.
        num_log_images: number of panel images to log to WandB.
    """

    def __init__(
        self,
        grid_size: int = 8,
        cell_size: int = 16,
        num_samples: int = 1000,
        num_log_images: int = 8,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_samples = num_samples
        self.num_log_images = num_log_images

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "MazeEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_total = self.num_samples
        n_log = self.num_log_images

        per_sample_cell_acc, per_sample_exact, per_sample_valid, per_sample_size = [], [], [], []
        panels: list = []
        n_done = 0

        n_batches = (n_total + pipeline.batch_size - 1) // pipeline.batch_size
        for batch in tqdm(dataloader, "Maze eval", total=n_batches):
            if n_done >= n_total:
                break
            B_cur = batch["images"].shape[0]

            conditions = model._batch_to_sample(batch, device)
            generated = pipeline.sample_one_batch(model, conditions, device)
            generated = model.decode_for_eval(generated)  # (B, 3, H, W) in [0, 1]

            acc = evaluate_mazes(
                generated,
                batch["spatial_conditions"].to(device),
                batch["solution"].to(device),
                batch["solution_mask"].to(device),
                batch["token_conditions"].to(device),
                self.grid_size,
                self.cell_size,
            )
            per_sample_cell_acc.append(acc["per_sample_cell_acc"])
            per_sample_exact.append(acc["per_sample_exact"])
            per_sample_valid.append(acc["per_sample_valid"])
            per_sample_size.append(acc["per_sample_active_size"])

            if _wandb is not None and len(panels) < n_log:
                n_new = min(n_log - len(panels), B_cur)
                sol_np = batch["solution"].numpy()
                mask_np = batch["solution_mask"].numpy()
                tok_np = batch["token_conditions"].numpy()
                cond_cpu = batch["spatial_conditions"].cpu()
                gen_cpu = generated.cpu()
                for i in range(n_new):
                    panel = make_maze_panel_image(
                        cond_cpu[i], gen_cpu[i], tok_np[i], sol_np[i], mask_np[i],
                        self.grid_size, self.cell_size,
                    )
                    panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_cur

        cell_acc = np.concatenate(per_sample_cell_acc)
        exact = np.concatenate(per_sample_exact)
        valid = np.concatenate(per_sample_valid)
        size = np.concatenate(per_sample_size)

        result: dict = {
            "cell_acc": float(cell_acc.mean()),
            "puzzle_acc": float(exact.mean()),
            "constraint_puzzle_acc": float(valid.mean()),
        }
        # Difficulty breakdown: same three metrics, restricted to each active
        # maze size seen in this eval pass — e.g. constraint_puzzle_acc_size3
        # (matches the published SketchVLM benchmark's scale) vs
        # constraint_puzzle_acc_size8 (hardest size trained on), so a curriculum
        # over min/max_active_size doesn't hide a harder-size regression inside
        # one pooled average.
        for s in sorted(set(size.tolist())):
            m = size == s
            result[f"cell_acc_size{s}"] = float(cell_acc[m].mean())
            result[f"puzzle_acc_size{s}"] = float(exact[m].mean())
            result[f"constraint_puzzle_acc_size{s}"] = float(valid[m].mean())
        if panels:
            result["samples"] = panels
        return result


# ── Steiner Tree ──────────────────────────────────────────────────────────────


class SteinerEvalCallback(EvalCallbackBase):
    """
    Best-of-N DDIM sampling eval for Steiner Tree models, matching the
    original paper's protocol (arXiv 2510.21697, Table 2): for each puzzle
    instance, draw `num_candidates` independent noise samples, extract a
    graph from each (vertex/edge detection — see eval/steiner_eval.py),
    discard invalid ones (not a tree, disconnected, or missing a terminal),
    and score the single best (shortest) survivor. No learned classifier is
    needed — like Maze, pixels are read directly off the known rendering
    scheme.

    Metrics, one group per evaluated point range (see extra_eval_sets),
    logged as "{range}/valid_rate", "{range}/ratio_mean", "{range}/ratio_std"
    — e.g. "10-20/valid_rate", "21-30/ratio_mean" — no other metrics are
    logged (kept deliberately minimal):
      {range}/valid_rate     — fraction of instances with >=1 valid
                               candidate among the N generated.
      {range}/ratio_mean/std — mean/std of (best valid candidate's length /
                               exact optimal length), over instances with
                               >=1 valid candidate. The exact optimal length
                               is looked up from the dataset's own
                               generation-time record via
                               dataloader.dataset.optimal_length_for(puzzle_id)
                               (see SteinerTreeDataset), not re-solved or
                               re-derived lossily from a rendered image.

    The primary (in-distribution) dataloader is evaluated as its own named
    range (primary_range_name, default "10-20") exactly like extra_eval_sets
    entries — no special-cased unsuffixed keys. extra_eval_sets adds
    out-of-distribution point-range test sets (the paper's own 21-30/31-40/
    41-50 generalization splits, evaluated with the *same trained model* —
    no retraining needed, since embedding_conditions/embedding_mask are
    eval-only fields never fed to the network; the model only ever sees the
    rendered condition image regardless of point count). Each entry:
        {name, hf_filename, max_points, hf_repo (optional),
         num_samples (optional, defaults to this callback's num_samples)}

    Args:
        image_size: must match the dataset's rendering resolution.
        num_samples: number of puzzle instances to evaluate on the primary
                     (in-distribution) dataloader.
        num_candidates: independent noise samples per instance for the
                        best-of-N selection (paper uses 10). Lower this (or
                        num_samples) if periodic training-time eval is too
                        slow — the model batch actually sent through the
                        painter per step is chunked to roughly
                        sampling_pipeline.batch_size // num_candidates
                        instances at a time, so raising num_candidates alone
                        doesn't blow up GPU memory, only wall-clock.
        num_log_images: WandB panel images (primary dataloader only), shown
                        using each logged instance's best-of-N candidate.
        primary_range_name: label for the primary dataloader's metric group.
        extra_eval_sets: optional list of OOD point-range test-set specs.
    """

    def __init__(
        self,
        image_size: int = 128,
        num_samples: int = 1000,
        num_candidates: int = 10,
        num_log_images: int = 8,
        primary_range_name: str = "10-20",
        extra_eval_sets: Optional[list] = None,
    ):
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.num_log_images = num_log_images
        self.primary_range_name = primary_range_name
        self._extra_specs = list(extra_eval_sets) if extra_eval_sets else []
        self._extra_dataloaders = None  # built lazily on first __call__

    def _build_extra_dataloaders(self, batch_size: int) -> list:
        from datasets.steiner_dataset import SteinerTreeDataset

        loaders = []
        for spec in self._extra_specs:
            ds_kwargs = {"image_size": self.image_size, "max_points": spec.get("max_points", 20)}
            if spec.get("hf_repo") is not None:
                ds_kwargs["hf_repo"] = spec["hf_repo"]
            if spec.get("hf_filename") is not None:
                ds_kwargs["hf_filename"] = spec["hf_filename"]
            ds = SteinerTreeDataset(**ds_kwargs)
            dl = DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=SteinerTreeDataset.collate_fn
            )
            loaders.append((spec["name"], spec.get("num_samples", self.num_samples), dl))
        return loaders

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "SteinerEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        if self._extra_dataloaders is None:
            self._extra_dataloaders = self._build_extra_dataloaders(dataloader.batch_size)

        result = self._eval_one(
            model, dataloader, accelerator, self.num_samples, self.primary_range_name, log_panels=True
        )
        for name, n_samples, extra_dl in self._extra_dataloaders:
            result.update(self._eval_one(model, extra_dl, accelerator, n_samples, name, log_panels=False))
        return result

    def _eval_one(self, model, dataloader, accelerator, n_total: int, range_name: str, log_panels: bool) -> dict:
        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_log = self.num_log_images if log_panels else 0
        dataset = getattr(dataloader, "dataset", None)
        length_lookup = getattr(dataset, "optimal_length_for", None)
        # Model batch = chunk * num_candidates instances; keeps GPU memory
        # roughly matching the pipeline's configured batch size regardless
        # of how many candidates-per-instance we're generating.
        chunk = max(1, pipeline.batch_size // self.num_candidates)

        all_any_valid: list = []
        all_ratios: list = []
        panels: list = []
        n_done = 0

        n_batches = (n_total + dataloader.batch_size - 1) // dataloader.batch_size
        for batch in tqdm(dataloader, f"Steiner eval [{range_name}]", total=n_batches):
            if n_done >= n_total:
                break
            B_full = batch["images"].shape[0]

            for start in range(0, B_full, chunk):
                end = min(start + chunk, B_full)
                sub = batch.slice(start, end)
                B = end - start

                conditions = model._batch_to_sample(sub, device)
                gen_bn = pipeline.sample_best_of_n(model, conditions, device, self.num_candidates)  # (B, N, C, H, W)
                N = gen_bn.shape[1]
                gen_flat = model.decode_for_eval(gen_bn.reshape(B * N, *gen_bn.shape[2:]))  # (B*N, 1, H, W)

                emb_cond_rep = sub["embedding_conditions"].to(device).repeat_interleave(N, dim=0)
                emb_mask_rep = sub["embedding_mask"].to(device).repeat_interleave(N, dim=0)
                acc = evaluate_steiner(gen_flat, emb_cond_rep, emb_mask_rep, self.image_size)

                valid_bn = np.asarray(acc["per_sample_valid"]).reshape(B, N)
                length_bn = np.asarray(acc["per_sample_length"]).reshape(B, N)
                any_valid, best_length, best_idx = _select_best_of_n(valid_bn, length_bn, maximize=False)
                all_any_valid.append(any_valid)

                if length_lookup is not None:
                    puzzle_ids = sub["puzzle_id"].cpu().tolist()
                    for i in range(B):
                        if not any_valid[i]:
                            continue
                        opt_len = length_lookup(puzzle_ids[i])
                        if opt_len is not None and opt_len > 0:
                            all_ratios.append(float(best_length[i]) / float(opt_len))

                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B)
                    cond_cpu = sub["spatial_conditions"].cpu()
                    true_cpu = sub["images"].cpu()
                    gen_bn_cpu = gen_flat.reshape(B, N, *gen_flat.shape[1:]).cpu()
                    for i in range(n_new):
                        idx = int(best_idx[i]) if best_idx[i] >= 0 else 0
                        panel = make_steiner_panel_image(cond_cpu[i], gen_bn_cpu[i, idx], true_cpu[i])
                        panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_full

        any_valid_all = np.concatenate(all_any_valid) if all_any_valid else np.zeros(0, dtype=bool)
        result: dict = {
            f"{range_name}/valid_rate": float(any_valid_all.mean()) if len(any_valid_all) else float("nan"),
        }
        if all_ratios:
            result[f"{range_name}/ratio_mean"] = float(np.mean(all_ratios))
            result[f"{range_name}/ratio_std"] = float(np.std(all_ratios))
        if panels:
            result["samples"] = panels
        return result


# ── Max-Area Polygon ───────────────────────────────────────────────────────────


class PolygonEvalCallback(EvalCallbackBase):
    """
    Best-of-N DDIM sampling eval for Max-Area Polygon models, matching the
    original paper's protocol (arXiv 2510.21697, Table 3): for each puzzle
    instance, draw `num_candidates` independent noise samples, detect which
    known-point-pairs form edges in each (distance-transform-based — see
    eval/polygon_eval.py), discard candidates that don't form a valid simple
    polygon through every point (Hamiltonian cycle + no self-intersections,
    checked from exact known coordinates, not pixels), and score the single
    best (largest-area) survivor.

    Metrics, one group per evaluated point range (see extra_eval_sets),
    logged as "{range}/valid_rate", "{range}/ratio_mean", "{range}/ratio_std",
    "{range}/opt_rate" — e.g. "7-12/valid_rate", "13-15/opt_rate" — no other
    metrics are logged (kept deliberately minimal):
      {range}/valid_rate     — fraction of instances with >=1 valid
                               candidate among the N generated.
      {range}/ratio_mean/std — mean/std of (best valid candidate's area /
                               exact optimal area), over instances with >=1
                               valid candidate, looked up via
                               dataloader.dataset.optimal_area_for(puzzle_id)
                               (see PolygonDataset) — not re-solved or
                               re-derived lossily from a rendered image.
      {range}/opt_rate       — fraction of instances (with >=1 valid
                               candidate) whose best candidate's recovered
                               vertex order is *exactly* the optimal polygon
                               (up to rotation/reflection — see
                               eval.polygon_eval._orders_equivalent and
                               PolygonDataset.optimal_order_for), not just
                               close in area — matches the paper's
                               "Opt. Rate" column.

    The primary (in-distribution) dataloader is evaluated as its own named
    range (primary_range_name, default "7-12") exactly like extra_eval_sets
    entries — no special-cased unsuffixed keys. extra_eval_sets adds
    out-of-distribution point-range test sets (the paper's own 13-15
    generalization split, evaluated with the *same trained model* — no
    retraining needed, since embedding_conditions/embedding_mask are
    eval-only fields never fed to the network). Each entry:
        {name, hf_filename, max_points, hf_repo (optional),
         num_samples (optional, defaults to this callback's num_samples)}

    Args:
        image_size: must match the dataset's rendering resolution.
        num_samples: number of puzzle instances to evaluate on the primary
                     (in-distribution) dataloader.
        num_candidates: independent noise samples per instance for the
                        best-of-N selection (paper uses 10). Lower this (or
                        num_samples) if periodic training-time eval is too
                        slow — see SteinerEvalCallback's docstring for the
                        chunking rationale (identical here).
        num_log_images: WandB panel images (primary dataloader only), shown
                        using each logged instance's best-of-N candidate.
        primary_range_name: label for the primary dataloader's metric group.
        extra_eval_sets: optional list of OOD point-range test-set specs.
    """

    def __init__(
        self,
        image_size: int = 128,
        num_samples: int = 1000,
        num_candidates: int = 10,
        num_log_images: int = 8,
        primary_range_name: str = "7-12",
        extra_eval_sets: Optional[list] = None,
    ):
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.num_log_images = num_log_images
        self.primary_range_name = primary_range_name
        self._extra_specs = list(extra_eval_sets) if extra_eval_sets else []
        self._extra_dataloaders = None  # built lazily on first __call__

    def _build_extra_dataloaders(self, batch_size: int) -> list:
        from datasets.polygon_dataset import PolygonDataset

        loaders = []
        for spec in self._extra_specs:
            ds_kwargs = {"image_size": self.image_size, "max_points": spec.get("max_points", 12)}
            if spec.get("hf_repo") is not None:
                ds_kwargs["hf_repo"] = spec["hf_repo"]
            if spec.get("hf_filename") is not None:
                ds_kwargs["hf_filename"] = spec["hf_filename"]
            ds = PolygonDataset(**ds_kwargs)
            dl = DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=PolygonDataset.collate_fn
            )
            loaders.append((spec["name"], spec.get("num_samples", self.num_samples), dl))
        return loaders

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "PolygonEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        if self._extra_dataloaders is None:
            self._extra_dataloaders = self._build_extra_dataloaders(dataloader.batch_size)

        result = self._eval_one(
            model, dataloader, accelerator, self.num_samples, self.primary_range_name, log_panels=True
        )
        for name, n_samples, extra_dl in self._extra_dataloaders:
            result.update(self._eval_one(model, extra_dl, accelerator, n_samples, name, log_panels=False))
        return result

    def _eval_one(self, model, dataloader, accelerator, n_total: int, range_name: str, log_panels: bool) -> dict:
        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_log = self.num_log_images if log_panels else 0
        dataset = getattr(dataloader, "dataset", None)
        area_lookup = getattr(dataset, "optimal_area_for", None)
        order_lookup = getattr(dataset, "optimal_order_for", None)
        chunk = max(1, pipeline.batch_size // self.num_candidates)

        all_any_valid: list = []
        all_ratios: list = []
        all_exact_match: list = []
        panels: list = []
        n_done = 0

        n_batches = (n_total + dataloader.batch_size - 1) // dataloader.batch_size
        for batch in tqdm(dataloader, f"Polygon eval [{range_name}]", total=n_batches):
            if n_done >= n_total:
                break
            B_full = batch["images"].shape[0]

            for start in range(0, B_full, chunk):
                end = min(start + chunk, B_full)
                sub = batch.slice(start, end)
                B = end - start

                conditions = model._batch_to_sample(sub, device)
                gen_bn = pipeline.sample_best_of_n(model, conditions, device, self.num_candidates)  # (B, N, C, H, W)
                N = gen_bn.shape[1]
                gen_flat = model.decode_for_eval(gen_bn.reshape(B * N, *gen_bn.shape[2:]))  # (B*N, 1, H, W)

                emb_cond_rep = sub["embedding_conditions"].to(device).repeat_interleave(N, dim=0)
                emb_mask_rep = sub["embedding_mask"].to(device).repeat_interleave(N, dim=0)
                acc = evaluate_polygon(gen_flat, emb_cond_rep, emb_mask_rep, self.image_size)

                valid_bn = np.asarray(acc["per_sample_valid"]).reshape(B, N)
                area_bn = np.asarray(acc["per_sample_area"]).reshape(B, N)
                orders_bn = [acc["per_sample_order"][k * N : (k + 1) * N] for k in range(B)]
                any_valid, best_area, best_idx = _select_best_of_n(valid_bn, area_bn, maximize=True)
                all_any_valid.append(any_valid)

                puzzle_ids = sub["puzzle_id"].cpu().tolist()
                for i in range(B):
                    if not any_valid[i]:
                        continue
                    if area_lookup is not None:
                        opt_area = area_lookup(puzzle_ids[i])
                        if opt_area is not None and opt_area > 0:
                            all_ratios.append(float(best_area[i]) / float(opt_area))
                    if order_lookup is not None:
                        true_order = order_lookup(puzzle_ids[i])
                        best_order = orders_bn[i][int(best_idx[i])]
                        if true_order is not None and best_order is not None:
                            all_exact_match.append(_orders_equivalent(best_order, true_order))

                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B)
                    cond_cpu = sub["spatial_conditions"].cpu()
                    true_cpu = sub["images"].cpu()
                    gen_bn_cpu = gen_flat.reshape(B, N, *gen_flat.shape[1:]).cpu()
                    for i in range(n_new):
                        idx = int(best_idx[i]) if best_idx[i] >= 0 else 0
                        panel = make_polygon_panel_image(cond_cpu[i], gen_bn_cpu[i, idx], true_cpu[i])
                        panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_full

        any_valid_all = np.concatenate(all_any_valid) if all_any_valid else np.zeros(0, dtype=bool)
        result: dict = {
            f"{range_name}/valid_rate": float(any_valid_all.mean()) if len(any_valid_all) else float("nan"),
        }
        if all_ratios:
            result[f"{range_name}/ratio_mean"] = float(np.mean(all_ratios))
            result[f"{range_name}/ratio_std"] = float(np.std(all_ratios))
        if order_lookup is not None and all_exact_match:
            result[f"{range_name}/opt_rate"] = float(np.mean(all_exact_match))
        if panels:
            result["samples"] = panels
        return result


# ── Ball Drop ───────────────────────────────────────────────────────────────


class BallDropEvalCallback(EvalCallbackBase):
    """
    Best-of-N DDIM sampling eval for Ball Drop models: sample images, extract
    the drawn solution line(s) (pixel-color based — see eval/ball_drop_eval.py),
    re-simulate physics from the instance's recorded ball start position, and
    check whether the ball settles in the recorded target bucket.

    Unlike SteinerEvalCallback/PolygonEvalCallback, there is no ratio/
    optimality dimension — this is a reachability/success task (any line
    placement that lands the ball in the target bucket is equally valid, see
    datasets/ball_drop_generation.py's docstring), so the only per-instance
    question is pass/fail, not "how good."

    Main metric:
      valid_rate — fraction of instances with >=1 candidate (out of
                   num_candidates independent noise samples) whose drawn
                   line(s) route the ball into the target bucket.

    Diagnostic metrics (per individual generated candidate, not best-of-N):
      constraint_puzzle_acc, settled_rate (ball settled at all, regardless
      of bucket — distinguishes "wrong bucket" from "physics never
      resolved"), mean_lines_extracted.

    extra_eval_sets adds additional held-out test sets with metrics suffixed
    `_{name}` — in particular the real loganbolton/sketchvlm-physics-ball-drop
    benchmark (see datasets/ball_drop_real_data.py), a genuine external
    generalization check unlike Steiner/Polygon's synthetic OOD point
    ranges: those 198 instances were solved by SketchVLM's own (undisclosed)
    physics generator, not ours, and re-simulating them with our pymunk
    parameters reproduces their recorded outcome ~88.9% of the time — see
    that module's docstring for the validation. Each entry:
        {name, hf_filename, hf_repo (optional),
         num_samples (optional, defaults to this callback's num_samples)}

    Args:
        image_size: must match the dataset's rendering resolution.
        num_samples: number of puzzle instances to evaluate on the primary
                     (in-distribution) dataloader.
        num_candidates: independent noise samples per instance for the
                        best-of-N selection. Lower this (or num_samples) if
                        periodic training-time eval is too slow — chunking
                        rationale identical to SteinerEvalCallback's.
        num_log_images: WandB panel images (primary dataloader only), shown
                        using each logged instance's best-of-N candidate.
        extra_eval_sets: optional list of held-out test-set specs.
    """

    def __init__(
        self,
        image_size: int = 128,
        num_samples: int = 1000,
        num_candidates: int = 10,
        num_log_images: int = 8,
        extra_eval_sets: Optional[list] = None,
    ):
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.num_log_images = num_log_images
        self._extra_specs = list(extra_eval_sets) if extra_eval_sets else []
        self._extra_dataloaders = None  # built lazily on first __call__

    def _build_extra_dataloaders(self, batch_size: int) -> list:
        from datasets.ball_drop_dataset import BallDropDataset

        loaders = []
        for spec in self._extra_specs:
            ds_kwargs = {"image_size": self.image_size}
            if spec.get("hf_repo") is not None:
                ds_kwargs["hf_repo"] = spec["hf_repo"]
            if spec.get("hf_filename") is not None:
                ds_kwargs["hf_filename"] = spec["hf_filename"]
            ds = BallDropDataset(**ds_kwargs)
            dl = DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=BallDropDataset.collate_fn
            )
            loaders.append((spec["name"], spec.get("num_samples", self.num_samples), dl))
        return loaders

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "BallDropEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        if self._extra_dataloaders is None:
            self._extra_dataloaders = self._build_extra_dataloaders(dataloader.batch_size)

        result = self._eval_one(model, dataloader, accelerator, self.num_samples, suffix="", log_panels=True)
        for name, n_samples, extra_dl in self._extra_dataloaders:
            result.update(
                self._eval_one(model, extra_dl, accelerator, n_samples, suffix=f"_{name}", log_panels=False)
            )
        return result

    def _eval_one(self, model, dataloader, accelerator, n_total: int, suffix: str, log_panels: bool) -> dict:
        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_log = self.num_log_images if log_panels else 0
        dataset = getattr(dataloader, "dataset", None)
        spec_lookup = getattr(dataset, "physics_spec_for", None)
        chunk = max(1, pipeline.batch_size // self.num_candidates)

        weighted_constraint, weighted_settled, weighted_lines = [], [], []
        all_any_valid: list = []
        panels: list = []
        n_done = 0

        if spec_lookup is None:
            import logging
            logging.getLogger(__name__).warning(
                "BallDropEvalCallback: dataset has no physics_spec_for method, skipping eval."
            )
            return {}

        n_batches = (n_total + dataloader.batch_size - 1) // dataloader.batch_size
        for batch in tqdm(dataloader, "Ball drop eval" + suffix, total=n_batches):
            if n_done >= n_total:
                break
            B_full = batch["images"].shape[0]

            for start in range(0, B_full, chunk):
                end = min(start + chunk, B_full)
                sub = batch.slice(start, end)
                B = end - start

                puzzle_ids = sub["puzzle_id"].cpu().tolist()
                specs = [spec_lookup(pid) for pid in puzzle_ids]
                ball_start_x = np.array([s["ball_start_x"] for s in specs], dtype=np.float64)
                target_bucket = np.array([s["target_bucket"] for s in specs], dtype=np.int64)

                conditions = model._batch_to_sample(sub, device)
                gen_bn = pipeline.sample_best_of_n(model, conditions, device, self.num_candidates)  # (B, N, C, H, W)
                N = gen_bn.shape[1]
                gen_flat = model.decode_for_eval(gen_bn.reshape(B * N, *gen_bn.shape[2:]))  # (B*N, 3, H, W)

                ball_start_x_rep = np.repeat(ball_start_x, N)
                target_bucket_rep = np.repeat(target_bucket, N)
                acc = evaluate_ball_drop(gen_flat, ball_start_x_rep, target_bucket_rep, self.image_size)

                w = B * N
                weighted_constraint.append((acc["constraint_puzzle_acc"], w))
                weighted_settled.append((float(acc["per_sample_settled"].mean()), w))
                weighted_lines.append((float(acc["per_sample_num_lines_extracted"].mean()), w))

                valid_bn = np.asarray(acc["per_sample_valid"]).reshape(B, N)
                any_valid = valid_bn.any(axis=1)
                all_any_valid.append(any_valid)

                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B)
                    cond_cpu = sub["spatial_conditions"].cpu()
                    true_cpu = sub["images"].cpu()
                    gen_bn_cpu = gen_flat.reshape(B, N, *gen_flat.shape[1:]).cpu()
                    for i in range(n_new):
                        # No "best" objective here (success is binary) — log
                        # the first valid candidate if any, else candidate 0.
                        row = valid_bn[i]
                        idx = int(np.argmax(row)) if row.any() else 0
                        panel = make_ball_drop_panel_image(cond_cpu[i], gen_bn_cpu[i, idx], true_cpu[i])
                        panels.append(_wandb.Image(panel, caption=f"sample[{n_done + i}]"))

            n_done += B_full

        any_valid_all = np.concatenate(all_any_valid) if all_any_valid else np.zeros(0, dtype=bool)
        result: dict = {
            f"constraint_puzzle_acc{suffix}": _weighted_mean(weighted_constraint),
            f"settled_rate{suffix}": _weighted_mean(weighted_settled),
            f"mean_lines_extracted{suffix}": _weighted_mean(weighted_lines),
            f"valid_rate{suffix}": float(any_valid_all.mean()) if len(any_valid_all) else float("nan"),
        }
        if panels:
            result["samples"] = panels
        return result


# ── Inscribed Square ──────────────────────────────────────────────────────────


class SquaresEvalCallback(EvalCallbackBase):
    """
    DDIM sampling eval for the Inscribed Square dataset: sample squares
    conditioned on a curve, score with squareness_metric (paper Eq. 2) /
    alignment_score (paper Eq. 1) — see eval/squares_eval.py. Directly
    comparable to the paper's Table 1 Square↑/Align↑ columns, not to the
    original repo's own wandb training curves (which log a different,
    decay_scale-based training-progress diagnostic instead — see that
    module's docstring for why this callback deliberately doesn't use it).

    Unlike SteinerEvalCallback/PolygonEvalCallback/BallDropEvalCallback,
    there is no best-of-N valid-candidate selection: neither metric is a
    pass/fail constraint check (a generated square always gets scored, never
    discarded), and the paper's own reported evaluation protocol samples
    exactly once per instance. num_candidates > 1 here instead averages both
    metrics over that many independent generations per instance, for a less
    noisy per-instance estimate — a deliberate extension, not a reproduction
    of a best-of-N selection.

    Main metrics:
      squareness_mean — mean squareness_metric over all generated candidates.
      alignment_mean  — mean alignment_score over all generated candidates,
                        in raw pixel units (typically negative — see
                        alignment_score's docstring).

    Args:
        image_size: must match the dataset's rendering resolution.
        num_samples: number of puzzle instances to evaluate.
        num_candidates: independent noise samples per instance to average
                        metrics over (default 1, matching the paper's own
                        single-sample evaluation protocol).
        num_log_images: WandB panel images, each showing curve | generated |
                        ground-truth square for one instance (candidate 0).
    """

    def __init__(
        self,
        image_size: int = 128,
        num_samples: int = 256,
        num_candidates: int = 1,
        num_log_images: int = 8,
    ):
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_candidates = num_candidates
        self.num_log_images = num_log_images

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging

            logging.getLogger(__name__).warning(
                "SquaresEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_log = self.num_log_images
        chunk = max(1, pipeline.batch_size // self.num_candidates)

        weighted_squareness, weighted_alignment = [], []
        panels: list = []
        n_done = 0

        n_batches = (self.num_samples + dataloader.batch_size - 1) // dataloader.batch_size
        for batch in tqdm(dataloader, "Inscribed Square eval", total=n_batches):
            if n_done >= self.num_samples:
                break
            B_full = batch["images"].shape[0]

            for start in range(0, B_full, chunk):
                end = min(start + chunk, B_full)
                sub = batch.slice(start, end)
                B = end - start

                conditions = model._batch_to_sample(sub, device)
                gen_bn = pipeline.sample_best_of_n(model, conditions, device, self.num_candidates)  # (B, N, C, H, W)
                N = gen_bn.shape[1]
                gen_flat = model.decode_for_eval(gen_bn.reshape(B * N, *gen_bn.shape[2:]))  # (B*N, 1, H, W)

                cond_rep = sub["spatial_conditions"].to(device).repeat_interleave(N, dim=0)
                acc = evaluate_squares(gen_flat, cond_rep)

                w = B * N
                weighted_squareness.append((acc["squareness_mean"], w))
                weighted_alignment.append((acc["alignment_mean"], w))

                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B)
                    cond_cpu = sub["spatial_conditions"].cpu()
                    true_cpu = sub["images"].cpu()
                    gen_bn_cpu = gen_flat.reshape(B, N, *gen_flat.shape[1:]).cpu()
                    for i in range(n_new):
                        # candidate 0 of instance i sits at flat index i*N (see
                        # cond_rep's repeat_interleave above)
                        sq = float(acc["per_sample_squareness"][i * N])
                        al = float(acc["per_sample_alignment"][i * N])
                        panel = make_squares_panel_image(
                            cond_cpu[i], gen_bn_cpu[i, 0], true_cpu[i], squareness=sq, alignment=al
                        )
                        panels.append(
                            _wandb.Image(panel, caption=f"sample[{n_done + i}] squareness={sq:.3f} alignment={al:.1f}px")
                        )

            n_done += B_full

        result: dict = {
            "squareness_mean": _weighted_mean(weighted_squareness),
            "alignment_mean": _weighted_mean(weighted_alignment),
        }
        if panels:
            result["samples"] = panels
        return result


# ── TopoDiff (topology optimization) ──────────────────────────────────────────


class TopodiffEvalCallback(EvalCallbackBase):
    """
    Single-sample DDIM eval for TopoDiff models: draw one generated topology
    per test instance (no best-of-N — the paper's own evaluation protocol
    scores exactly one sample per instance; its guidance mechanism, not
    sample selection, is what's responsible for quality there) and score it
    with the paper's own 4 metrics (see eval/topodiff_eval.py, a thin
    wrapper around eval/topodiff_analysis.py — a verbatim vendored copy of
    the paper's own topodiff/topodiff_analysis.py):

      CE  — compliance error: FEA-computed compliance of the generated
            topology relative to the reference SIMP-optimal compliance,
            C(generated)/C(SIMP) - 1. Only computable when the dataloader's
            dataset has a precomputed reference compliance for every
            instance in the batch (TopodiffDataset.compliance_for() —
            only non-None for the test_data_level_1/2 splits, see
            datasets/topodiff_dataset.py); silently omitted from the result
            otherwise (e.g. a training-split validation dataloader still
            gets VFE/LV/FM, just no CE).
      VFE — volume fraction error: |VF(generated) - VF(target)| / VF(target).
      LV  — load violation rate: proportion of instances with no material at
            the applied load point.
      FM  — floating material rate: proportion of instances whose generated
            topology has disconnected material regions.

    The FEA solve (compute_deflection's replacement, see topodiff_eval.py)
    is CPU-bound and slow (~2s/sample on a single core) — keep num_samples
    modest for periodic training-time eval (e.g. 64-256); run a much larger
    offline pass (e.g. the full 1800-instance test_data_level_1) only for a
    one-off final evaluation, not every eval_every step.

    extra_eval_sets adds additional held-out test-set dataloaders (typically
    the paper's own test_data_level_1 in-distribution / test_data_level_2
    out-of-distribution splits), each contributing metrics suffixed
    `_{name}` (e.g. "CE_level_1", "VFE_level_2"). Each entry:
        {name, split, data_dir (optional, defaults to this callback's
         data_dir), ids (optional), num_samples (optional, defaults to this
         callback's num_samples)}

    Args:
        data_dir: root of the extracted dataset_1_diff directory (see
                  datasets/topodiff_dataset.py's module docstring).
        image_size: must match the dataset's native resolution (64).
        num_samples: number of instances to evaluate on the primary
                     dataloader.
        num_log_images: WandB panel images (primary dataloader only).
        condition_mode: must match the condition_mode the model's own
                        train_dataset/val_dataset use ("full" or "hard" —
                        see datasets/topodiff_dataset.py) — applied to the
                        primary dataloader's extra_eval_sets siblings built
                        here; the primary dataloader itself is whatever the
                        caller passes in (already built with the right mode
                        by train_trm.py from configs/data/*.yaml).
        num_workers: parallelize the per-sample FEA solve (the bottleneck,
                     ~2s/sample single-threaded CPU) across this many worker
                     processes — see eval/topodiff_eval.py's
                     evaluate_topodiff/_get_pool docstrings. 1 (default)
                     runs sequentially in the main process, matching prior
                     behavior exactly.
        extra_eval_sets: optional list of held-out test-set specs.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
        num_samples: int = 256,
        num_log_images: int = 8,
        condition_mode: str = "full",
        num_workers: int = 1,
        extra_eval_sets: Optional[list] = None,
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_log_images = num_log_images
        self.condition_mode = condition_mode
        self.num_workers = num_workers
        self._extra_specs = list(extra_eval_sets) if extra_eval_sets else []
        self._extra_dataloaders = None  # built lazily on first __call__

    def _build_extra_dataloaders(self, batch_size: int) -> list:
        from datasets.topodiff_dataset import TopodiffDataset

        loaders = []
        for spec in self._extra_specs:
            ds = TopodiffDataset(
                data_dir=spec.get("data_dir", self.data_dir),
                split=spec["split"],
                ids=spec.get("ids"),
                condition_mode=spec.get("condition_mode", self.condition_mode),
            )
            dl = DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=TopodiffDataset.collate_fn
            )
            loaders.append((spec["name"], spec.get("num_samples", self.num_samples), dl))
        return loaders

    def __call__(self, model, dataloader, accelerator, **kwargs) -> dict:
        if not accelerator.is_main_process:
            return {}
        if not hasattr(model, "_batch_to_sample"):
            import logging
            logging.getLogger(__name__).warning(
                "TopodiffEvalCallback: model has no _batch_to_sample method, skipping eval."
            )
            return {}

        if self._extra_dataloaders is None:
            self._extra_dataloaders = self._build_extra_dataloaders(dataloader.batch_size)

        result = self._eval_one(model, dataloader, accelerator, self.num_samples, suffix="", log_panels=True)
        for name, n_samples, extra_dl in self._extra_dataloaders:
            result.update(
                self._eval_one(model, extra_dl, accelerator, n_samples, suffix=f"_{name}", log_panels=False)
            )
        return result

    def _eval_one(self, model, dataloader, accelerator, n_total: int, suffix: str, log_panels: bool) -> dict:
        device = accelerator.device
        pipeline = model.sampling_pipeline
        n_log = self.num_log_images if log_panels else 0
        dataset = dataloader.dataset
        chunk = max(1, pipeline.batch_size)

        weighted_ce, weighted_vfe, weighted_lv, weighted_fm = [], [], [], []
        panels: list = []
        n_done = 0

        n_batches = (n_total + dataloader.batch_size - 1) // dataloader.batch_size
        for batch in tqdm(dataloader, "TopoDiff eval" + suffix, total=n_batches):
            if n_done >= n_total:
                break
            B_full = batch["spatial_conditions"].shape[0]

            for start in range(0, B_full, chunk):
                end = min(start + chunk, B_full)
                sub = batch.slice(start, end)
                B = end - start

                conditions = model._batch_to_sample(sub, device)
                gen = pipeline.sample_one_batch(model, conditions, device)  # (B, 1, H, W)
                gen = model.decode_for_eval(gen)

                puzzle_ids = sub["puzzle_id"].cpu().tolist()
                summaries = [dataset.summary_for(pid) for pid in puzzle_ids]
                compliance_opt = np.array([dataset.compliance_for(pid) for pid in puzzle_ids], dtype=np.float64)
                has_compliance = not np.isnan(compliance_opt).any()

                acc = evaluate_topodiff(
                    gen, summaries, compliance_opt=compliance_opt if has_compliance else None,
                    num_workers=self.num_workers,
                )

                if has_compliance:
                    weighted_ce.append((acc["CE"], B))
                weighted_vfe.append((acc["VFE"], B))
                weighted_lv.append((acc["LV"], B))
                weighted_fm.append((acc["FM"], B))

                if _wandb is not None and len(panels) < n_log:
                    n_new = min(n_log - len(panels), B)
                    cond_cpu = sub["spatial_conditions"].cpu()
                    true_cpu = sub["images"].cpu() if sub["images"] is not None else None
                    gen_cpu = gen.cpu()
                    for i in range(n_new):
                        ref = true_cpu[i] if true_cpu is not None else None
                        panel = make_topodiff_panel_image(cond_cpu[i], gen_cpu[i], ref)
                        caption = (
                            f"sample[{n_done + i}] "
                            f"CE={acc['per_sample_ce'][i]:.3f} VFE={acc['per_sample_vfe'][i]:.3f}"
                        )
                        panels.append(_wandb.Image(panel, caption=caption))

            n_done += B_full

        result: dict = {
            f"VFE{suffix}": _weighted_mean(weighted_vfe),
            f"LV{suffix}": _weighted_mean(weighted_lv),
            f"FM{suffix}": _weighted_mean(weighted_fm),
        }
        if weighted_ce:
            result[f"CE{suffix}"] = _weighted_mean(weighted_ce)
        if panels:
            result["samples"] = panels
        return result
