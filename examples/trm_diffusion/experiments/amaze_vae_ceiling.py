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

from datasets.amaze_dataset import AmazeDataset  # noqa: E402
from eval.amaze_eval import AmazeMetrics  # noqa: E402
from models.utility_models import load_frozen_vae  # noqa: E402


def _loader(dataset, batch_size, limit=None, seed=0):
    if limit is not None and limit < len(dataset):
        g = torch.Generator().manual_seed(seed)
        dataset = Subset(dataset, torch.randperm(len(dataset), generator=g)[:limit].tolist())
        collate = dataset.dataset.collate_fn
    else:
        collate = dataset.collate_fn
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)


@torch.no_grad()
def ceiling(vae, task, test_path, size, batch_size, limit, device):
    ds = AmazeDataset(test_path, split="test", image_size=size, include_metadata=True)
    recon_scorer = AmazeMetrics(device=device, task=task)
    resized_scorer = AmazeMetrics(device=device, task=task)
    mse_sum, n = 0.0, 0
    for batch in _loader(ds, batch_size, limit):
        x = batch.images.to(device)                                    # [0,1]
        z = vae.encode(x * 2 - 1).latent_dist.mode()
        rec = ((vae.decode(z).sample + 1) / 2).clamp(0, 1)
        mse_sum += (rec - x).pow(2).flatten(1).mean(1).sum().item()
        n += x.shape[0]
        meta = list(batch.metadata)
        recon_scorer.compute_and_accumulate_metrics(rec.cpu(), meta)
        resized_scorer.compute_and_accumulate_metrics(x.cpu(), meta)
    latent_hw = list(z.shape[-2:])
    return {
        "image_size": size,
        "latent_shape": [z.shape[1], *latent_hw],
        "pixel_mse": mse_sum / max(n, 1),
        "vae_recon": {k: v for k, v in recon_scorer.return_metrics().items() if k.startswith("mean_")},
        "resized_gt": {k: v for k, v in resized_scorer.return_metrics().items() if k.startswith("mean_")},
    }


@torch.no_grad()
def latent_stats(vae, train_path, size, batch_size, limit, device):
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
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_frozen_vae(args.vae).to(device).eval()
    task_dir = os.path.join(args.data_root, args.task)
    test_path = os.path.join(task_dir, "maze_dataset_test.parquet")

    result = {"task": args.task, "vae": args.vae, "ceiling": []}
    for size in args.sizes:
        r = ceiling(vae, args.task, test_path, size, args.batch_size, args.n_test, device)
        result["ceiling"].append(r)
        print(f"[{args.task}] {size}px latent={r['latent_shape']} pixel_mse={r['pixel_mse']:.5f}\n"
              f"   VAE recon : {json.dumps(r['vae_recon'])}\n"
              f"   resized GT: {json.dumps(r['resized_gt'])}", flush=True)

    train_path = os.path.join(task_dir, f"maze_dataset_train_{args.stats_size}.parquet")
    shift, scale = latent_stats(vae, train_path, args.stats_size, args.batch_size, args.n_stats, device)
    result.update({"stats_size": args.stats_size, "latent_shift": shift, "latent_scale": scale})
    print(f"[{args.task}] latent_shift={shift} latent_scale={scale}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
