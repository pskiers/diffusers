"""VAE ceiling + latent statistics for the latent Amaze painter-thinker.

1. Ceiling: encode -> decode the Amaze TEST ground-truth solutions with a frozen
   pretrained VAE and score the reconstructions with the same AmazeMetrics the
   eval callbacks use. A latent-diffusion model can never beat its VAE's
   reconstruction, so this is the upper bound of the whole latent pipeline. The
   resized-but-not-encoded ground truth is scored too, to separate what the
   resize costs from what the VAE costs.
2. Statistics: per-channel mean/std of the VAE latents of TRAIN solutions at the
   training resolution, written as {"latent_shift": [...], "latent_scale": [...]}
   for UNetPainter's per-channel latent normalisation.

Usage:
    python experiments/amaze_vae_ceiling.py --task maze --data_root data/amaze \
        --sizes 144 288 --stats_size 288 --out runs/vae_ceiling_maze.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.amaze_dataset import AmazeDataset, AmazeScaledDataset, sample_scale_and_offset, scale_and_paste  # noqa: E402
from eval.amaze_eval import AmazeMetrics, crop_to_content  # noqa: E402
from models.utility_models import load_frozen_vae  # noqa: E402


def _loader(dataset, batch_size, limit=None, seed=0):
    if limit is not None and limit < len(dataset):
        g = torch.Generator().manual_seed(seed)
        dataset = Subset(dataset, torch.randperm(len(dataset), generator=g)[:limit].tolist())
        collate = dataset.dataset.collate_fn
    else:
        collate = dataset.collate_fn
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)


def _means(scorer):
    return {k: v for k, v in scorer.return_metrics().items() if k.startswith("mean_")}


@torch.no_grad()
def ceiling(vae, task, test_path, size, batch_size, limit, device, scaled=None):
    """scaled: None, or dict(scale_min, scale_max, canvas_color, threshold, min_frac) —
    additionally score the GT after the AmazeScaledDataset transform + crop-back
    (does crop-back scoring itself reproduce the GT?) and that transformed GT
    through the VAE + crop-back (the latent pipeline's ceiling on scaled data)."""
    ds = AmazeDataset(test_path, split="test", image_size=size, include_metadata=True)
    names = ["vae_recon", "resized_gt"] + (["scaled_gt_cropped", "scaled_vae_cropped"] if scaled else [])
    scorers = {k: AmazeMetrics(device=device, task=task) for k in names}
    torch.manual_seed(1234)
    mse_sum, n = 0.0, 0
    for batch in _loader(ds, batch_size, limit):
        x = batch.images.to(device)                                    # [0,1]
        z = vae.encode(x * 2 - 1).latent_dist.mode()
        rec = ((vae.decode(z).sample + 1) / 2).clamp(0, 1)
        mse_sum += (rec - x).pow(2).flatten(1).mean(1).sum().item()
        n += x.shape[0]
        meta = list(batch.metadata)
        scorers["vae_recon"].compute_and_accumulate_metrics(rec.cpu(), meta)
        scorers["resized_gt"].compute_and_accumulate_metrics(x.cpu(), meta)
        if scaled:
            crop = dict(canvas_color=scaled["canvas_color"], threshold=scaled["threshold"], min_frac=scaled["min_frac"])
            xs = torch.stack([
                scale_and_paste(img, *sample_scale_and_offset(size, scaled["scale_min"], scaled["scale_max"]),
                                canvas_color=scaled["canvas_color"])
                for img in batch.images
            ])
            scorers["scaled_gt_cropped"].compute_and_accumulate_metrics(crop_to_content(xs, **crop), meta)
            zs = vae.encode(xs.to(device) * 2 - 1).latent_dist.mode()
            recs = ((vae.decode(zs).sample + 1) / 2).clamp(0, 1).cpu()
            scorers["scaled_vae_cropped"].compute_and_accumulate_metrics(crop_to_content(recs, **crop), meta)
    latent_hw = list(z.shape[-2:])
    out = {"image_size": size, "latent_shape": [z.shape[1], *latent_hw], "pixel_mse": mse_sum / max(n, 1)}
    out.update({k: _means(v) for k, v in scorers.items()})
    return out


@torch.no_grad()
def latent_stats(vae, train_path, size, batch_size, limit, device, scaled=None):
    if scaled:   # stats of what the painter actually trains on: the scaled targets
        ds = AmazeScaledDataset(train_path, split="train", image_size=size, include_metadata=False,
                                scale_min=scaled["scale_min"], scale_max=scaled["scale_max"],
                                canvas_color=scaled["canvas_color"])
    else:
        ds = AmazeDataset(train_path, split="train", image_size=size, include_metadata=False)
    s = s2 = None
    count = 0
    for batch in _loader(ds, batch_size, limit, seed=1):
        z = vae.encode(batch.images.to(device) * 2 - 1).latent_dist.sample().double()
        per_c = z.transpose(0, 1).flatten(1)                          # (C, B*H*W)
        s = per_c.sum(1) if s is None else s + per_c.sum(1)
        s2 = per_c.pow(2).sum(1) if s2 is None else s2 + per_c.pow(2).sum(1)
        count += per_c.shape[1]
    mean = s / count
    std = (s2 / count - mean.pow(2)).clamp(min=1e-12).sqrt()
    return [round(v, 6) for v in mean.tolist()], [round(v, 6) for v in std.tolist()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["maze", "queens"], required=True)
    p.add_argument("--data_root", default="data/amaze")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--sizes", type=int, nargs="+", default=[144, 288])
    p.add_argument("--stats_size", type=int, default=288)
    p.add_argument("--n_test", type=int, default=None, help="limit test puzzles (default: all)")
    p.add_argument("--n_stats", type=int, default=4096, help="train images for latent statistics")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--scaled", action="store_true", help="AmazeScaledDataset checks + stats on scaled targets")
    p.add_argument("--scale_min", type=float, default=0.8)
    p.add_argument("--scale_max", type=float, default=0.9)
    p.add_argument("--canvas_color", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--bbox_threshold", type=float, default=0.05)
    p.add_argument("--bbox_min_frac", type=float, default=0.1)
    args = p.parse_args()
    scaled = None
    if args.scaled:
        scaled = dict(scale_min=args.scale_min, scale_max=args.scale_max, canvas_color=tuple(args.canvas_color),
                      threshold=args.bbox_threshold, min_frac=args.bbox_min_frac)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_frozen_vae(args.vae).to(device).eval()
    task_dir = os.path.join(args.data_root, args.task)
    test_path = os.path.join(task_dir, "maze_dataset_test.parquet")

    result = {"task": args.task, "vae": args.vae, "scaled": scaled, "ceiling": []}
    for size in args.sizes:
        r = ceiling(vae, args.task, test_path, size, args.batch_size, args.n_test, device, scaled)
        result["ceiling"].append(r)
        print(f"[{args.task}] {size}px latent={r['latent_shape']} pixel_mse={r['pixel_mse']:.5f}", flush=True)
        for k in ("resized_gt", "vae_recon", "scaled_gt_cropped", "scaled_vae_cropped"):
            if k in r:
                print(f"   {k:<19}: {json.dumps(r[k])}", flush=True)

    train_path = os.path.join(task_dir, f"maze_dataset_train_{args.stats_size}.parquet")
    shift, scale = latent_stats(vae, train_path, args.stats_size, args.batch_size, args.n_stats, device, scaled)
    result.update({"stats_size": args.stats_size, "latent_shift": shift, "latent_scale": scale})
    print(f"[{args.task}] latent_shift={shift} latent_scale={scale}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
