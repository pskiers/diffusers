"""
ABOUTME: Rebuttal exp 07 analysis 1 (Fig. 5 analogue on Maze / Queens / CLEVR): the x0 estimate at each denoising step
ABOUTME: of PaTh and of the DM, scored by the task's own eval, by % of sampling and by puzzle size / object count.

Each model samples with its own eval sampler (sampling_pipeline.sample_one_batch, models.sampling.TrajectoryRecorder
capturing the scheduler's pred_original_sample) on a subset: AMAZE the first --boards-per-combo boards of every
(geometry, size) test parquet, 1 sample each; CLEVR --n-scenes seeded val scenes (3-10 objects, the callback's
conditioning). Up to --record-steps evenly spaced steps (always first and last) are decoded and scored:
- AMAZE: AmazeMetrics (eval/amaze_eval.py), the paper's Violation (`background_violation`), Coverage and Exact.
- CLEVR: ClevrMetricsCallback's scorer (`_score_image`, DINO + SigLIP + relation check), spatial accuracy =
  satisfied / checkable relations, pooled per step.
Early x0 estimates are blurry; the scorers are tuned for clean images and may misread them.

Usage (from examples/trm_diffusion; CLEVR not on the local A100, GroundingDINO fails silently there):
    python experiments/trajectory_violations.py --task maze --model path
    python experiments/trajectory_violations.py --task clevr --model dm --n-scenes 300
    # smoke: --boards-per-combo 2 --record-steps 4 / --n-scenes 4
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from compute_table import ENTRIES, build  # noqa: E402

CHECKPOINTS = {
    "maze/path": "runs/pt_maze_final_thinker/checkpoint_final.pt",
    "maze/dm": "runs/dit_maze_final/checkpoint_final.pt",
    "queens/path": "runs/pt_queens_final_thinker/checkpoint_final.pt",
    "queens/dm": "runs/queens_dit_baseline/checkpoint_final.pt",
    "clevr/path": "runs/clevr_thinker_v1_centroidmaskfused_x0hint_t80_sliced_biggrid_forceuncond/checkpoint_step-90048.pt",
    "clevr/dm": "runs/clevr_unet_concat_painter_very_big/checkpoint_step-40000.pt",
}
OBJ_BUCKETS = ((3, 4), (5, 6), (7, 8), (9, 10))


def record_steps(num_steps: int, k: int) -> set[int]:
    """k evenly spaced step indices, always the first and the last (all steps if k >= num_steps)."""
    if k >= num_steps:
        return set(range(num_steps))
    return {round(i * (num_steps - 1) / (k - 1)) for i in range(k)}


@torch.no_grad()
def trajectory(model, conditions, device, steps: set[int], seed: int) -> list[dict]:
    """Sample with the model's own pipeline; return [{step, t, frac, images (B, C, H, W) in [0, 1]}]."""
    from models.sampling import TrajectoryRecorder

    pipe = model.sampling_pipeline
    rec = TrajectoryRecorder(steps=steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    pipe.sample_one_batch(model, conditions, device, generator=gen, recorder=rec)
    n = pipe.num_inference_steps
    return [{"step": r["step"], "t": r["t"], "frac": (r["step"] + 1) / n,
             "images": model.decode_for_eval(r["x0_pred"].to(device)).cpu()} for r in rec.records]


def run_amaze(model, cfg, task: str, args, device) -> dict:
    from amaze_generate_and_calculate_metrics import _build_amaze_dataset
    from eval.amaze_eval import MAZE_GEOMETRIES, MAZE_OOD_SCALES, MAZE_SCALES, QUEEN_OOD_SCALES, QUEEN_SCALES, AmazeMetrics

    scorer = AmazeMetrics(device=device, task=task)
    root = Path(args.data_root)
    if task == "maze":
        combos = [(f"{g}_{s}", s, root / "maze" / g / f"n{s}_test.parquet")
                  for g in MAZE_GEOMETRIES for s in [*MAZE_SCALES, *MAZE_OOD_SCALES]]
    else:
        combos = [(str(s), s, root / "queens" / f"n{s}_test.parquet") for s in [*QUEEN_SCALES, *QUEEN_OOD_SCALES]]
    steps = record_steps(model.sampling_pipeline.num_inference_steps, args.record_steps)
    per = {}  # (frac, size) -> lists of per-board metrics
    first = True
    for name, size, parquet in combos:
        assert parquet.exists(), parquet
        ds = _build_amaze_dataset(cfg, str(parquet))
        n = min(len(ds), args.boards_per_combo)
        for start in range(0, n, args.batch_size):
            puzzles = [ds[i] for i in range(start, min(start + args.batch_size, n))]
            conditions = model._batch_to_sample(ds.collate_fn(puzzles), device)
            metadata = [p.metadata or {} for p in puzzles]
            for r in trajectory(model, conditions, device, steps, args.seed + start):
                recs = scorer.compute_and_accumulate_metrics(r["images"][:, None], metadata)
                for group in (str(size), "all"):
                    d = per.setdefault((round(r["frac"], 4), group), {"violation": [], "coverage": [], "exact": []})
                    d["violation"] += [x["background_violation"] for x in recs]
                    d["coverage"] += [x["gt_cell_coverage"] for x in recs]
                    d["exact"] += [x["exact"] for x in recs]
                if first:
                    print(f"  first batch {name}: {len(puzzles)} boards, step {r['step']} (t={r['t']}), images "
                          f"{tuple(r['images'].shape)} [{r['images'].min():.2f}, {r['images'].max():.2f}], "
                          f"violation {np.mean(d['violation']):.3f}", flush=True)
                    first = False
        print(f"  [{task}/{name}] {n} boards done", flush=True)
    return {f"{frac}|{size}": {k: float(np.mean(v)) for k, v in d.items()} | {"n": len(d["exact"])}
            for (frac, size), d in sorted(per.items())}


def run_clevr(model, args, device) -> dict:
    from datasets.clevr_dataset import ORIG_H, ORIG_W
    from datasets.data_sample import collate_data_samples
    from eval.clevr_eval_callbacks import ClevrMetricsCallback, _load_scenes, _scene_to_data_sample
    from eval.evaluate_clevr import _score_image
    from PIL import Image

    cb = ClevrMetricsCallback(root_dir=args.clevr_root, mode="reduced", include_centroid_mask=True,
                              centroid_mask_presence_only=True)
    cb._load(device)
    H, l_vec, f_vec, sz_thresh = cb._calibration
    dino_proc, dino_mod, sig_proc, sig_mod, text_embeds = cb._eval_models
    scenes = random.Random(args.seed).sample(_load_scenes(args.clevr_root, "val", 3, 10), args.n_scenes)
    steps = record_steps(model.sampling_pipeline.num_inference_steps, args.record_steps)
    per = {}  # (frac, bucket) -> [satisfied, checkable]
    for start in range(0, len(scenes), args.batch_size):
        chunk = scenes[start:start + args.batch_size]
        samples = [_scene_to_data_sample(s, "reduced", include_centroid_mask=True, centroid_mask_presence_only=True,
                                         mask_size=32) for s in chunk]
        conditions = collate_data_samples(samples).to(device)
        for r in trajectory(model, conditions, device, steps, args.seed + start):
            for img, scene in zip(r["images"], chunk):
                pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize(
                    (ORIG_W, ORIG_H), Image.BILINEAR)  # as ClevrMetricsCallback
                inc = _score_image(pil, scene["objects"], scene["relationships"], H, l_vec, f_vec, sz_thresh,
                                   dino_proc, dino_mod, sig_proc, sig_mod, text_embeds)
                n_obj = len(scene["objects"])
                bucket = next(f"{a}-{b}" for a, b in OBJ_BUCKETS if a <= n_obj <= b)
                for key in (bucket, "all"):
                    d = per.setdefault((round(r["frac"], 4), key), [0, 0])
                    d[0] += inc["c_rel"]
                    d[1] += inc["t_rel"]
        print(f"  [clevr] {start + len(chunk)}/{len(scenes)} scenes; last step spatial "
              f"{per[(1.0, 'all')][0] / max(1, per[(1.0, 'all')][1]):.3f}", flush=True)
    return {f"{frac}|{key}": {"spatial_acc": c / max(1, t), "checkable_relations": t}
            for (frac, key), (c, t) in sorted(per.items())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", choices=["maze", "queens", "clevr"], required=True)
    p.add_argument("--model", choices=["path", "dm"], required=True)
    p.add_argument("--checkpoint", default=None, help="default: CHECKPOINTS[task/model]")
    p.add_argument("--boards-per-combo", type=int, default=50)
    p.add_argument("--n-scenes", type=int, default=300)
    p.add_argument("--record-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data/amaze")
    p.add_argument("--clevr-root", default="data/clevr")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    key = f"{args.task}/{args.model}"
    ckpt = args.checkpoint or CHECKPOINTS[key]
    out_dir = Path(args.out_dir or HERE / "results" / "trajectory_violations" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    t0 = time.time()

    from eval.checkpoint_utils import load_checkpoint

    extra = [f"data.clevr_root={args.clevr_root}"] if args.task == "clevr" else []
    cfg, model, _ = build(ENTRIES[key], extra, device)
    step = load_checkpoint(model, ckpt, use_ema=True, device="cpu")
    model = model.to(device).eval()
    print(f"{key}: {ENTRIES[key]['run']} from {ckpt} (step {step}); sampler "
          f"{model.sampling_pipeline.num_inference_steps} steps, CFG {model.sampling_pipeline.cfg_scale}", flush=True)
    res = run_clevr(model, args, device) if args.task == "clevr" else run_amaze(model, cfg, args.task, args, device)

    out = {"task": args.task, "model": args.model, "run": ENTRIES[key]["run"], "checkpoint": ckpt, "step": step,
           "args": vars(args), "per_step": res, "wall_clock_s": round(time.time() - t0, 1),
           "command": " ".join(sys.argv), "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(device),
           "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 cwd=HERE).stdout.strip()}
    name = f"{args.task}_{args.model}"
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=2))
    metrics = ["spatial_acc"] if args.task == "clevr" else ["violation", "coverage", "exact"]
    groups = sorted({k.split("|")[1] for k in res}, key=lambda g: (g != "all", len(g), g))
    fracs = sorted({float(k.split("|")[0]) for k in res})
    lines = [f"# {name}: the x0 estimate scored by % of sampling", "", f"{ENTRIES[key]['run']}, `{ckpt}`"]
    for metric in metrics:
        lines += ["", f"## {metric}", "", "| % sampling | " + " | ".join(groups) + " |", "|---" * (1 + len(groups)) + "|"]
        for f in fracs:
            lines.append(f"| {100 * f:.0f} | " + " | ".join(
                f"{res[f'{round(f, 4)}|{g}'][metric]:.3f}" if f"{round(f, 4)}|{g}" in res else "-" for g in groups)
                + " |")
    (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved -> {out_dir / name}.json ({out['wall_clock_s']} s)")


if __name__ == "__main__":
    main()
