import dataclasses
import logging
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from hydra.utils import instantiate


Z_KEYS = ("z_L", "z_H")


@dataclasses.dataclass
class BatchRuns:
    """One batch of boards, run in one or more paired versions (sets)."""
    runs: dict                                            # {set_name: (conditions, x_t, run_timesteps)}, every set with the same B boards
    info: dict = dataclasses.field(default_factory=dict)  # per-batch extras for the metrics: masks, labels, example images, ...


class InterpretabilityAblation:
    """Runs boards batch by batch, records the thinker's z states and streams metrics over them.

    run() works in two phases, on disjoint boards:
        1. fit:      boards [0, K)     -> by default the template: the mean z per (t, z_L / z_H, update, token), over boards and sets
                                          (a subclass can fit something else here, f.ex. the path probe)
        2. metrics:  boards [K, K + N) -> metrics computed per batch and summed into accumulators; the states are freed after each batch
    So memory is one batch of states, whatever N is.

    Subclasses implement _make_runs, _accumulate_metrics_per_batch and _visualise_metrics (and _init_metrics if needed).
    """
    needs_template: bool = True
    output_subdir: str = "interpretability_ablation"

    @torch.no_grad()
    def _collect_z_states(
        self,
        model,
        conditions,
        x_t: torch.Tensor,
        run_timesteps: torch.Tensor,
        record_at: list,
        generator=None,
        keep_L_indices: list | None = None,           # f.ex. list(range(36)) + [71, 143, 215, 287] for saving memory
        storage_dtype: torch.dtype = torch.float32,   # float32: bf16 / fp16 rounding is as large as the late updates' changes
    ):
        """Run full denoising process and collect z states at specified timesteps.

        Args:
            model: The diffusion model.
            conditions: The conditions for the model.
            x_t (torch.Tensor): The initial noisy input.
            run_timesteps (torch.Tensor): The timesteps to run the denoising process.
            record_at (list): The timesteps at which to record z states.
            generator: Optional random number generator for reproducibility.
            keep_L_indices (list | None): Indices of L-level states to keep for memory efficiency.
            storage_dtype (torch.dtype): The data type to use for storing z states.

        Returns:
            dict: A dictionary containing the collected z states keyed by timestep:
                {t: {"z_L": (B, n_sup * H_cycles * L_cycles, n_tokens, hidden), "z_H": (B, n_sup * H_cycles, n_tokens, hidden),
                     "x": (B, C, H, W)}},
                every update in call order (conditional pass only); "x" is the image the thinker sees at t.
        """
        predictor = model.sampling_pipeline.predictor
        per_cycle = model.thinker.inner.config.L_cycles + 1
        x = x_t
        states = {}

        for t in run_timesteps:
            t_batch = t.expand(x.shape[0]).to(x.device)
            step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
            record = int(t.item()) in record_at

            level_outputs = []
            if record:
                handle = model.thinker.inner.L_level.register_forward_hook(
                    lambda module, inputs, output: level_outputs.append(
                        output.detach().to(device="cpu", dtype=storage_dtype)
                    )
                )

            pred_cond, _, _ = model.forward_with_carry(step_sample)
            if record:
                handle.remove()
            pred_uncond = model(model.null_condition_sample(step_sample)).pred
            pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)

            if record:
                z_L_list = [z for i, z in enumerate(level_outputs) if i % per_cycle != per_cycle - 1]
                if keep_L_indices is not None:
                    z_L_list = [z_L_list[idx] for idx in keep_L_indices]
                z_H_list = [z for i, z in enumerate(level_outputs) if i % per_cycle == per_cycle - 1]

                states[int(t.item())] = {
                    "z_L": torch.stack(z_L_list, dim=1),  # (B, n_L_kept, n_tokens, hidden)
                    "z_H": torch.stack(z_H_list, dim=1),  # (B, 48, n_tokens, hidden)
                    "x": x.detach().to(device="cpu", dtype=torch.float32),  # (B, C, H, W)
                }

            kw = {"generator": generator} if generator is not None else {}
            x = model.scheduler.step(pred, t, x, **kw).prev_sample

        return states

    def _make_runs(self, model, ds, indices: list, base_seed: int, device) -> BatchRuns | None:
        """Build the sets to run for these boards. All randomness (starting noise) must be the same for every set.

        Returns None if no board of the batch is usable.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _run_batch(self, model, batch_runs: BatchRuns, record_at: list, base_seed: int) -> dict:
        """Run every set of the batch and record its z states.

        Every set gets a fresh generator with the same seed, so the scheduler's step noise is the same for all of them:
        that is what makes the sets paired.

        Returns:
            dict: {set_name: {t: {"z_L", "z_H", "x"}}}
        """
        states = {}
        for name, (conditions, x_t, run_timesteps) in batch_runs.runs.items():
            generator = torch.Generator(device=x_t.device).manual_seed(base_seed + 1)
            states[name] = self._collect_z_states(
                model, conditions, x_t, run_timesteps, record_at, generator, keep_L_indices=self.keep_L_indices
            )
            missing = [t for t in record_at if t not in states[name]]
            if missing:
                raise SystemExit(f"[{name}] record_at {missing} not in the run's timesteps {run_timesteps.tolist()}")
        return states

    def _accumulate_template(self, states: dict, sums: dict) -> None:
        """Add one batch to the template sums: over boards AND over sets, per (t, z_L / z_H, update, token).

        WHY: Model has positional embeddings. If we took two completely different boards (Maze A and Maze B) and
        compared their vector at field 00, we would find that they are very similar, because both primarily indicate "I am field 0".
        One template shared by all sets: a template per set would also remove the average effect of what differs between the sets.

        Args:
            states (dict): {set_name: {t: {"z_L" | "z_H": (B, n_updates, n_tokens, hidden)}}}
            sums (dict): {t: {"z_L" | "z_H": {"sum": (n_updates, n_tokens, hidden), "count": int}}}, updated in place.
        """
        for set_states in states.values():
            for t, s in set_states.items():
                for k in Z_KEYS:
                    acc = sums.setdefault(t, {}).setdefault(k, {"sum": 0.0, "count": 0})
                    acc["sum"] = acc["sum"] + s[k].float().sum(dim=0)
                    acc["count"] += s[k].shape[0]

    def _accumulate_fit(self, states: dict, info: dict, acc: dict) -> None:
        """Fit phase, per batch. Default: the template sums (see _accumulate_template)."""
        self._accumulate_template(states, acc)

    def _finish_fit(self, acc: dict, device) -> None:
        """Fit phase, after the last batch. Default: self.template[t][k] = sum / count, (n_updates, n_tokens, hidden)."""
        self.template = {t: {k: a["sum"] / a["count"] for k, a in s.items()} for t, s in acc.items()}

    def _init_metrics(self) -> dict:
        return {}

    def _accumulate_metrics_per_batch(self, model, states: dict, info: dict, metrics: dict) -> None:
        """Compute the metrics of one batch per board and add them to `metrics` (in place)."""
        raise NotImplementedError("Subclasses should implement this method.")

    def _visualise_metrics(self, model, metrics: dict, output_dir: Path) -> None:
        raise NotImplementedError("Subclasses should implement this method.")

    def _batches(self, model, ds, first: int, last: int, batch_size: int, seed: int, device):
        """Yield (batch_runs, base_seed) for the boards [first, last), skipping batches with no usable board."""
        for start in range(first, last, batch_size):
            base_seed = seed + start * 977
            batch_runs = self._make_runs(model, ds, list(range(start, min(start + batch_size, last))), base_seed, device)
            if batch_runs is not None:
                yield batch_runs, base_seed

    def run(self, cfg: dict):
        interprebability_config = cfg.get("interprebability_config", {})
        num_samples = int(interprebability_config.get("num_samples", 128))          # N: boards the metrics are computed on
        template_samples = int(interprebability_config.get("template_samples", 32))  # K: boards used only for the fit phase (template / probe training)
        batch_size = int(interprebability_config.get("batch_size", 16))
        seed = int(interprebability_config.get("seed", 0))

        checkpoint = interprebability_config.get("checkpoint", None)
        model_name = interprebability_config.get("model_name", "painter_thinker_default")
        if checkpoint is None:
            raise SystemExit("Set +checkpoint=<...>.pt")

        data_parquet = str(interprebability_config.get("data_parquet"))
        output_dir = Path(str(interprebability_config.get("output_dir", "runs/interpretability/"))) / self.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.set_float32_matmul_precision("high")
        logging.basicConfig(level=logging.INFO)
        accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
        device = accelerator.device

        scheduler = instantiate(cfg.diffusion)
        model = build_model(cfg, scheduler)
        load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
        model = accelerator.prepare(model)
        model = accelerator.unwrap_model(model)
        model.eval()

        num_inference_steps = int(interprebability_config.get("num_inference_steps", model.sampling_pipeline.num_inference_steps))
        model.scheduler.set_timesteps(num_inference_steps, device=device)
        full_ts = model.scheduler.timesteps

        if not os.path.isabs(data_parquet):
            data_parquet = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), data_parquet)  # relative to trm_diffusion/

        vd = cfg.data.get("val_dataset", {})
        ds = AmazeDataset(
            dataset_path=data_parquet, split="test",
            image_size=int(cfg.data.get("image_size", 144)),
            condition_field=vd.get("condition_field", "m_original_img"),
            target_field=vd.get("target_field", "sol_img"),
            num_channels=int(vd.get("num_channels", 3)), include_metadata=True,
        )
        n_template = min(template_samples, len(ds)) if self.needs_template else 0
        n_total = min(n_template + num_samples, len(ds))
        logging.info(f"[{model_name}] dataset={data_parquet}: template boards [0, {n_template}), metric boards [{n_template}, {n_total}) of {len(ds)}")

        # Timesteps at which every z update is recorded
        record_at = [int(t) for t in interprebability_config.get("record_at", [70])]
        missing = [t for t in record_at if t not in {int(s) for s in full_ts}]
        if missing:
            raise SystemExit(f"record_at {missing} not in the denoising timesteps {full_ts.tolist()}")

        # Used by the subclasses' _make_runs
        self.interprebability_config = interprebability_config
        self.full_ts = full_ts
        keep_L_indices = interprebability_config.get("keep_L_indices", None)
        self.keep_L_indices = [int(i) for i in keep_L_indices] if keep_L_indices is not None else None

        # 1. Fit: by default the template, the mean z over the template boards (and over all sets); a subclass can fit
        #    something else on these boards instead (f.ex. a probe), via _accumulate_fit / _finish_fit
        self.template = None
        if self.needs_template:
            fit_acc = {}
            for batch_runs, base_seed in self._batches(model, ds, 0, n_template, batch_size, seed, device):
                states = self._run_batch(model, batch_runs, record_at, base_seed)
                self._accumulate_fit(states, batch_runs.info, fit_acc)
                del states
            if not fit_acc:
                raise SystemExit("No usable template board")
            self._finish_fit(fit_acc, device)
            logging.info(f"[{model_name}] fit phase done on boards [0, {n_template})")

        # 2. Metrics: per batch, then summed; the states of a batch are freed before the next one
        metrics = self._init_metrics()
        for batch_runs, base_seed in self._batches(model, ds, n_template, n_total, batch_size, seed, device):
            states = self._run_batch(model, batch_runs, record_at, base_seed)
            self._accumulate_metrics_per_batch(model, states, batch_runs.info, metrics)
            del states
            logging.info(f"[{model_name}] metrics: batch with base_seed={base_seed} done")

        torch.save(metrics, output_dir / "metrics.pt")  # the plots can be redone from this without the GPU
        self._visualise_metrics(model, metrics, output_dir)
