"""paper_plots/eval_mnist_joint_vs_separate_diversity.py — quantitative
companion to plot_mnist_joint_vs_separate_diversity.py. Uses the same
extracted, hand-labelled cells (by default all 81 cells per grid, including
the given cells, which both models redraw rather than copy) and measures
within-digit diversity per model:

  * LPIPS (AlexNet): mean pairwise LPIPS between same-digit samples
    (cells upsampled 16 -> 64 px, replicated to 3 channels).
  * Pixel L2: mean pairwise L2 between L2-normalised cells.
  * Both of the above also on binarised cells (>0.5), to reduce the effect
    of stroke thickness / intensity.
  * Ink: mean total pixel intensity per cell (stroke-thickness proxy).

Each diversity metric is averaged over digits 1-9. Significance: one-sided
permutation test on that class-averaged gap (separate - joint). Model labels
are shuffled within each digit class, keeping per-class sample counts fixed.

Usage (run from examples/trm_diffusion/):
    python paper_plots/eval_mnist_joint_vs_separate_diversity.py \\
      --out_tex paper_plots/mnist_joint_vs_separate_diversity_table.tex
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lpips
import numpy as np
import torch
import torch.nn.functional as F

from plot_mnist_joint_vs_separate_diversity import MODELS, gather


def pairwise_mean(d: np.ndarray, idx: np.ndarray) -> float:
    sub = d[np.ix_(idx, idx)]
    n = len(idx)
    return float(sub.sum() / (n * (n - 1)))


def l2_matrix(cells: list[np.ndarray]) -> np.ndarray:
    x = np.stack([c.ravel() / (np.linalg.norm(c) + 1e-8) for c in cells])
    return np.linalg.norm(x[:, None] - x[None], axis=-1)


def lpips_matrix(net, cells: list[np.ndarray]) -> np.ndarray:
    x = torch.from_numpy(np.stack(cells))[:, None].float()
    x = F.interpolate(x, size=64, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) * 2 - 1
    n = len(cells)
    ii, jj = np.triu_indices(n, 1)
    with torch.no_grad():
        v = net(x[ii], x[jj]).flatten().numpy()
    m = np.zeros((n, n))
    m[ii, jj] = v
    return m + m.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_tex", default=None)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_givens", action="store_true",
                    help="drop given cells (by default they are kept: the model redraws them too)")
    args = ap.parse_args()

    data = {m: gather(m, include_givens=not args.exclude_givens) for m in MODELS}
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    binarize = lambda c: (c > 0.5).astype(np.float32)
    metrics = {
        "LPIPS": lambda cs: lpips_matrix(net, cs),
        "LPIPS (bin.)": lambda cs: lpips_matrix(net, [binarize(c) for c in cs]),
        "Pixel $L_2$": l2_matrix,
        "Pixel $L_2$ (bin.)": lambda cs: l2_matrix([binarize(c) for c in cs]),
    }

    # Per digit: pooled cells (joint first), split index, one distance matrix per metric.
    per_digit = []
    for d in range(1, 10):
        cells = data["joint"][d] + data["separate"][d]
        per_digit.append((len(data["joint"][d]), {k: f(cells) for k, f in metrics.items()},
                          np.array([c.sum() for c in cells])))
    print("cells per model: " + ", ".join(f"{m}={sum(len(v) for v in data[m].values())}" for m in MODELS))

    def gaps(key, perms):
        return np.array([pairwise_mean(mats[key], p[nj:]) - pairwise_mean(mats[key], p[:nj])
                         for (nj, mats, _), p in zip(per_digit, perms)])

    sizes = [len(ink) for _, _, ink in per_digit]
    identity = [np.arange(n) for n in sizes]
    rng = np.random.default_rng(args.seed)
    perms = [[rng.permutation(n) for n in sizes] for _ in range(args.n_perm)]

    results = {}
    for key in metrics:
        per = {m: np.array([pairwise_mean(mats[key], np.arange(nj) if m == "joint" else np.arange(nj, len(ink)))
                            for nj, mats, ink in per_digit]) for m in MODELS}
        obs = gaps(key, identity).mean()
        null = np.array([gaps(key, p).mean() for p in perms])
        pval = (1 + (null >= obs).sum()) / (1 + args.n_perm)
        wins = int((per["separate"] > per["joint"]).sum())
        results[key] = dict(joint=per["joint"].mean(), separate=per["separate"].mean(), p=pval, wins=wins)
        print(f"{key:20s} joint={per['joint'].mean():.3f} separate={per['separate'].mean():.3f} "
              f"sep>joint on {wins}/9 digits, permutation p={pval:.4g}")
        print("    per digit (j/s): " + " ".join(f"{d}:{a:.3f}/{b:.3f}" for d, a, b in
                                             zip(range(1, 10), per["joint"], per["separate"])))
    ink = {m: np.mean([ink[:nj].mean() if m == "joint" else ink[nj:].mean() for nj, _, ink in per_digit])
           for m in MODELS}
    print(f"{'Ink (mean)':20s} " + " ".join(f"{m}={v:.1f}" for m, v in ink.items()))

    if args.out_tex:
        keys = list(metrics)
        lines = [r"\begin{tabular}{l" + "c" * len(keys) + "}", r"\toprule",
                 "Model & " + " & ".join(f"{k} $\\uparrow$" for k in keys) + r" \\", r"\midrule"]
        for m, name in MODELS.items():
            cells = []
            for k in keys:
                v = results[k][m]
                best = v == max(results[k][mm] for mm in MODELS)
                cells.append(f"\\textbf{{{v:.3f}}}" if best else f"{v:.3f}")
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines += [r"\midrule", "$p$-value & " + " & ".join(
            f"{results[k]['p']:.3f}" if results[k]["p"] >= 1e-3 else r"$<$0.001" for k in keys) + r" \\",
                  r"\bottomrule", r"\end{tabular}"]
        with open(args.out_tex, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("saved", args.out_tex)


if __name__ == "__main__":
    main()
