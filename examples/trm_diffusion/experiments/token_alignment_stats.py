"""TEMPORARY (Claude): cell/token alignment per shape and size -- who fails: the token labels, the thinker or the painter?

Inputs: runs/claude_shapes_sizes/{plain,path}/<shape>_n<size>.pt (shapes_sizes.py: plain TRM 40k and PaTh on the amaze
test sets; same boards in both) + the test parquets (cell maps), boards selected exactly as shapes_sizes.py (first
boards whose maze graph is a valid tree).
Image 144 px, 12 x 12 tokens of 12 px. Token labels (as in training / the probes): path if >= 10% of the token's pixels
are blue, off if none, else ignored.

Per shape and size:
  geometry  cells per maze; median cell side (sqrt of the area at 144 px) in tokens; distinct cells per token (tokens inside
            the maze); share of tokens covering >= 2 / >= 3 cells; share of in-maze tokens with an ignored label; share of path
            tokens with only 10-25 % blue (borderline)
  plain     its own output vs the labels: exact boards; wrong tokens per board (extra / missing); share of extra tokens
            touching a path token; exact if those are forgiven; share of errors on tokens covering >= 2 cells
  PaTh thinker   the cross-fit path probe on the thinker's z_L (last of 72 recorded z_L updates, t = 70, no painter): same
  PaTh drawn     the generated image (shapes_sizes.py): exact boards; cell error rate by cell-area tercile
Heatmaps (one figure per shape, rows = sizes): an example maze with the token grid and the number of cells per token;
mean cells per token; plain error rate; PaTh-thinker error rate; share of ignored labels -- per token position.
Output: <out_dir>/token_stats.csv, token_stats.md, tokens_<shape>.png, tokens_summary.png
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from shapes_sizes import maze_graph

G, P = 12, 12
SHAPES = ["square", "hexagon", "triangle", "circle"]
SIZES = [5, 7, 8, 9, 10, 11, 13, 16]


def adj(mask):
    m = torch.as_tensor(mask).view(-1, 1, G, G).float()
    k = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).float().view(1, 1, 3, 3)
    return (F.conv2d(m, k, padding=1) > 0).view(torch.as_tensor(mask).shape)


def blue_frac(sol):                                                     # (N, 3, 144, 144) uint8 -> (N, 144)
    s = sol.float() / 255
    b = ((s[:, 2] > 0.5) & (s[:, 0] < 0.5) & (s[:, 2] - s[:, 0] > 0.2)).float()
    return b.view(-1, G, P, G, P).mean((2, 4)).reshape(-1, G * G)


def errors(pred, lab):
    known, path = lab >= 0, lab == 1
    wrong = (pred != path) & known
    extra, miss = wrong & (lab == 0), wrong & path
    ex_adj = extra & adj(path)
    return dict(exact=float((wrong.sum(1) == 0).float().mean()), wrong=float(wrong.sum(1).float().mean()),
                extra=float(extra.sum(1).float().mean()), miss=float(miss.sum(1).float().mean()),
                extra_adj=float(ex_adj.sum() / max(1, int(extra.sum()))),
                exact_forgiven=float(((wrong & ~ex_adj).sum(1) == 0).float().mean())), wrong, known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/claude_shapes_sizes"); ap.add_argument("--data_root", default="data/amaze/maze")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, maps = [], {}
    for shape in SHAPES:
        for size in SIZES:
            fp, fq = Path(a.runs) / "plain" / f"{shape}_n{size}.pt", Path(a.runs) / "path" / f"{shape}_n{size}.pt"
            pq = Path(a.data_root) / shape / f"n{size}_test.parquet"
            if not (fp.exists() and fq.exists() and pq.exists()):
                print(f"skip {shape} n{size}"); continue
            rp, rq = torch.load(fp, weights_only=False), torch.load(fq, weights_only=False)
            ds = AmazeDataset(dataset_path=str(pq), split="test", image_size=144, condition_field="m_original_img",
                              target_field="sol_img", num_channels=3, include_metadata=True)
            cmaps = []
            for i in range(len(ds)):
                try:                                                     # the same selection as shapes_sizes.py
                    g = maze_graph(ds[i].metadata)
                except Exception:
                    continue
                if g["ok"]:
                    cmaps.append(cv2.resize(g["ids"].astype(np.float32), (144, 144), interpolation=cv2.INTER_NEAREST).astype(np.int64))
                if len(cmaps) == rp["n"]:
                    break
            N = rp["n"]
            assert len(cmaps) == N, (shape, size, len(cmaps), N)
            lab = rp["labels"].float()
            # geometry
            ncell = np.zeros((N, G * G)); inside = np.zeros((N, G * G), bool); areas = []
            for i, cm in enumerate(cmaps):
                blk = cm.reshape(G, P, G, P).transpose(0, 2, 1, 3).reshape(G * G, P * P)
                for t in range(G * G):
                    u = np.unique(blk[t]); u = u[u > 0]
                    ncell[i, t] = len(u); inside[i, t] = len(u) > 0
                _, cnt = np.unique(cm[cm > 0], return_counts=True); areas += list(cnt)
            bf = blue_frac(rp["sol"])
            path_tok = lab == 1
            ep, wp, kp = errors(rp["pred"].bool(), lab)
            pthink = rq["p"].float()[:, -1] > 0.5
            eq, wq, kq = errors(pthink, rq["labels"].float())
            multi = torch.as_tensor(ncell >= 2)
            ce = rq["cell_err"]                                           # (board, cell, area, on solution, wrong)
            terc = np.quantile(ce[:, 2], [1 / 3, 2 / 3]) if len(ce) else [0, 0]
            cerr = [ce[(ce[:, 2] <= terc[0]), 4].mean(), ce[(ce[:, 2] > terc[0]) & (ce[:, 2] <= terc[1]), 4].mean(), ce[ce[:, 2] > terc[1], 4].mean()]
            row = dict(shape=shape, n=size, boards=N, cells=len(areas) / N, cell_side_tok=float(np.sqrt(np.median(areas)) / P),
                       cells_per_tok=float(ncell[inside].mean()), tok_ge2=float((ncell[inside] >= 2).mean()), tok_ge3=float((ncell[inside] >= 3).mean()),
                       ignored=float((lab[torch.as_tensor(inside)] < 0).float().mean()),
                       borderline=float(((bf >= 0.1) & (bf < 0.25))[path_tok].float().mean()),
                       plain_exact=ep["exact"], plain_forgiven=ep["exact_forgiven"], plain_wrong=ep["wrong"], plain_extra=ep["extra"],
                       plain_miss=ep["miss"], plain_extra_adj=ep["extra_adj"],
                       plain_err_on_multi=float((wp & multi).sum() / max(1, int(wp.sum()))),
                       path_think_exact=eq["exact"], path_think_forgiven=eq["exact_forgiven"], path_think_wrong=eq["wrong"],
                       path_think_extra_adj=eq["extra_adj"], path_think_err_on_multi=float((wq & multi).sum() / max(1, int(wq.sum()))),
                       path_drawn_exact=float(np.mean(rq["exact"])), cell_err_small=float(cerr[0]), cell_err_mid=float(cerr[1]), cell_err_large=float(cerr[2]))
            rows.append(row)
            maps[(shape, size)] = dict(cells=ncell.mean(0).reshape(G, G),
                                       plain=(wp.float().sum(0) / kp.float().sum(0).clamp_min(1)).numpy().reshape(G, G),
                                       think=(wq.float().sum(0) / kq.float().sum(0).clamp_min(1)).numpy().reshape(G, G),
                                       ignored=((lab < 0) & torch.as_tensor(inside)).float().mean(0).numpy().reshape(G, G),
                                       example=rp["maze"][0].permute(1, 2, 0).numpy(), ex_cells=ncell[0].reshape(G, G),
                                       ex_wrong_plain=wp[0].numpy().reshape(G, G), ex_wrong_think=wq[0].numpy().reshape(G, G))
            print(f"{shape:8s} n{size:2d}: cells/token {row['cells_per_tok']:.2f}, >=2 cells {row['tok_ge2']:.0%} | plain {row['plain_exact']:.0%} "
                  f"(forgiven {row['plain_forgiven']:.0%}) | PaTh thinker {row['path_think_exact']:.0%} (forgiven {row['path_think_forgiven']:.0%}) "
                  f"| PaTh drawn {row['path_drawn_exact']:.0%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "token_stats.csv", index=False)
    pct = lambda x: f"{100 * x:.0f}%"
    with open(out / "token_stats.md", "w") as f:
        f.write("| shape | n | cells | cell side (tokens) | cells per token | tokens with ≥2 cells | ignored labels | borderline path tokens "
                "| plain exact | plain, adjacent extras forgiven | plain errors on ≥2-cell tokens | PaTh thinker exact | PaTh thinker, forgiven "
                "| PaTh drawn exact | PaTh cell errors small / mid / large cells |\n|" + "---|" * 15 + "\n")
        for r in rows:
            f.write(f"| {r['shape']} | {r['n']} | {r['cells']:.0f} | {r['cell_side_tok']:.2f} | {r['cells_per_tok']:.2f} | {pct(r['tok_ge2'])} | "
                    f"{pct(r['ignored'])} | {pct(r['borderline'])} | {pct(r['plain_exact'])} | {pct(r['plain_forgiven'])} | {pct(r['plain_err_on_multi'])} | "
                    f"{pct(r['path_think_exact'])} | {pct(r['path_think_forgiven'])} | {pct(r['path_drawn_exact'])} | "
                    f"{pct(r['cell_err_small'])} / {pct(r['cell_err_mid'])} / {pct(r['cell_err_large'])} |\n")

    # ---------------------------------------------------------------- heatmaps per shape
    for shape in SHAPES:
        sizes = [s for s in SIZES if (shape, s) in maps]
        if not sizes:
            continue
        fig, ax = plt.subplots(len(sizes), 5, figsize=(17, 3.3 * len(sizes)))
        ax = np.atleast_2d(ax)
        for r, s in enumerate(sizes):
            M = maps[(shape, s)]
            ax[r, 0].imshow(M["example"])
            ax[r, 0].imshow(np.kron(M["ex_cells"], np.ones((P, P))), cmap="magma", alpha=0.35, vmin=0, vmax=4)
            for k in range(1, G):
                ax[r, 0].axhline(k * P - 0.5, color="c", lw=0.4); ax[r, 0].axvline(k * P - 0.5, color="c", lw=0.4)
            ys, xs = np.nonzero(M["ex_wrong_plain"])
            ax[r, 0].scatter(xs * P + P / 2, ys * P + P / 2, marker="x", c="r", s=25, label="plain error")
            ax[r, 0].set_title(f"{shape} n{s}: board 0, token grid, cells per token (shade)", fontsize=8)
            for c, (key, title, vmax, cmap) in enumerate((("cells", "mean cells per token", 4, "magma"),
                                                         ("plain", "plain TRM: token error rate", None, "Reds"),
                                                         ("think", "PaTh thinker (probe): token error rate", None, "Reds"),
                                                         ("ignored", "ignored labels (10%-boundary)", None, "Blues")), start=1):
                v = M[key]
                im = ax[r, c].imshow(v, cmap=cmap, vmin=0, vmax=vmax if vmax else max(0.05, float(v.max())))
                ax[r, c].set_title(f"{title} (max {v.max():.2f})", fontsize=8); plt.colorbar(im, ax=ax[r, c], fraction=0.046)
            for c in range(5):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        fig.tight_layout(); fig.savefig(out / f"tokens_{shape}.png", dpi=90); plt.close(fig)

    # ---------------------------------------------------------------- summary lines
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    for i, shape in enumerate(SHAPES):
        d = df[df["shape"] == shape]
        if d.empty:
            continue
        for col, lab_, st in (("plain_exact", "plain", "C0-"), ("plain_forgiven", "plain, adjacent extras forgiven", "C0--"),
                              ("path_think_exact", "PaTh thinker", "C1-"), ("path_think_forgiven", "PaTh thinker, forgiven", "C1--"),
                              ("path_drawn_exact", "PaTh drawn", "C3-")):
            ax[i].plot(d["n"], d[col], st, marker="o", label=lab_)
        ax2 = ax[i].twinx(); ax2.bar(d["n"], d["tok_ge2"], width=0.6, alpha=0.15, color="k"); ax2.set_ylim(0, 1)
        ax2.set_ylabel("tokens covering ≥ 2 cells (bars)", fontsize=8)
        ax[i].axvline(10, color="0.7", ls=":"); ax[i].set(title=shape, xlabel="maze size n", ylabel="boards exact", ylim=(-0.02, 1.02))
        if i == 0:
            ax[i].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "tokens_summary.png", dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
