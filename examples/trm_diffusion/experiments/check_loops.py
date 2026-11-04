"""TEMPORARY check (Claude): do the models run dead-end filling? Give them 8x8 mazes with one wall removed (= one loop).

The training mazes are perfect (a tree, one solution). Removing one inner wall between two adjacent cells adds one loop.
Two kinds per board (random wall of each kind, same seed for both models):
  side  - the loop lies off the solution (in a side branch): the shortest path is still the old solution, but dead-end
          filling cannot remove the loop nor the corridor that connects it to the solution ("lollipop").
  route - the loop contains solution cells: a second route between S and G. Dead-end filling keeps both routes,
          the right answer is the shorter one.
Per board and model we read which cells are drawn and compare with
  SP  = shortest path (the right answer),  DEF = what dead-end filling leaves.
PaTh: full sampling (20 denoising steps, 1 attempt), drawn cells as in the evaluator (predicted_cells).
plain: compared token by token with the answers rendered like the training targets (blue line through the cell centres,
token labels as in training: 1 path, 0 off, -1 ignored); only token-level numbers (tokens straddle cells).
Output: <out_dir>/loops_<kind>.pt (+ loops_examples_<kind>.png), printed summary.
"""
import dataclasses
import random
from collections import deque
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation
from interp_prune_mech import CONFIG_DIR

KINDS = ["none", "side", "route"]


def tree(ms):
    nb = {rc: [] for rc in ms.rc2id}
    walls = []
    for (r, c) in ms.rc2id:
        for v in ((r, c + 1), (r + 1, c)):
            if v in ms.rc2id:
                if mcl._is_open(ms.morig, r, c, *v, ms.cell_px):
                    nb[(r, c)].append(v); nb[v].append((r, c))
                else:
                    walls.append(((r, c), v))
    return nb, walls


def bfs_path(nb, a, b):
    prev = {a: None}; dq = deque([a])
    while dq:
        u = dq.popleft()
        for v in nb[u]:
            if v not in prev:
                prev[v] = u; dq.append(v)
    out, u = [], b
    while u is not None:
        out.append(u); u = prev[u]
    return out[::-1]


def dead_end_fill(nb, s, g):
    alive = set(nb)
    changed = True
    while changed:
        changed = False
        for p in list(alive):
            if p not in (s, g) and sum(v in alive for v in nb[p]) <= 1:
                alive.discard(p); changed = True
    return alive


def remove_wall(img, ms, u, v):
    """Paint the wall segment between adjacent cells u, v white (native resolution), keeping the corner posts."""
    out = img.copy()
    cp = ms.cell_px
    dark = img.max(2) < 100
    (r, c), (r2, c2) = sorted([u, v])
    if r == r2:                                                   # vertical wall at x = (c + 1) * cp
        x, y0, y1 = int((c + 1) * cp), int(r * cp), int((r + 1) * cp)
        ym = (y0 + y1) // 2
        cols = [x + d for d in range(-12, 13) if 0 <= x + d < img.shape[1] and dark[ym, x + d]]
        if cols:
            out[y0 + 14:y1 - 4, min(cols) - 1:max(cols) + 2] = 255
    else:                                                         # horizontal wall at y = (r + 1) * cp
        y, x0, x1 = int((r + 1) * cp), int(c * cp), int((c + 1) * cp)
        xm = (x0 + x1) // 2
        rows = [y + d for d in range(-12, 13) if 0 <= y + d < img.shape[0] and dark[y + d, xm]]
        if rows:
            out[min(rows) - 1:max(rows) + 2, x0 + 14:x1 - 4] = 255
    return out


def variants(ms, rng, side_pick="random", min_side=0):
    """side_pick: which removed wall makes the side loop - "random" (default, as in all earlier runs), or the one giving the
    "longest" / "shortest" cycle (in cells); boards whose chosen side loop is shorter than min_side give None."""
    nb, walls = tree(ms)
    rc = [ms.id2rc[p] for p in ms.path]
    s, g, sol = rc[0], rc[-1], set(rc)
    if sum(len(x) for x in nb.values()) // 2 != len(nb) - 1:
        return None                                               # not a perfect maze
    side, route = [], []
    for u, v in walls:
        cyc = set(bfs_path(nb, u, v))
        (route if len(cyc & sol) >= 2 else side if not (cyc & sol) else []).append((u, v))
    out = {"none": (ms.morig, nb, None)}
    for name, pool in (("side", side), ("route", route)):
        if not pool:
            return None
        if name == "side" and side_pick != "random":
            key = lambda w: (len(bfs_path(nb, *w)), w)
            u, v = max(pool, key=key) if side_pick == "longest" else min(pool, key=key)
        else:
            u, v = rng.choice(pool)
        if name == "side" and len(bfs_path(nb, u, v)) < min_side:
            return None
        nb2 = {k: list(x) for k, x in nb.items()}
        nb2[u].append(v); nb2[v].append(u)
        out[name] = (remove_wall(ms.morig, ms, u, v), nb2, (u, v))
    res = {}
    for name, (img, nbk, wall) in out.items():
        sp = set(bfs_path(nbk, s, g))
        res[name] = dict(img=img, wall=wall, sp=sp, def_=dead_end_fill(nbk, s, g), sol=sol, nb=nbk)
    return res


def render_labels(ds, ms, img, nbk, cells):
    """Token labels (144,) of `cells` drawn like the training targets: blue line between connected cells' centres."""
    out = img.copy()
    for a in cells:
        for b in nbk[a]:
            if b in cells and a < b:
                cv2.line(out, mcl._center(a, ms.cell_px), mcl._center(b, ms.cell_px), ms.blue_color, ms.stroke)
    t = ds._image_to_tensor(Image.fromarray(out))[None]
    return ZPathEncodingAblation._token_labels(t, 12, 0.1)[0].numpy()


def cell_tokens(ms, rc):
    """Tokens touched by the segment from the cell centre to the middle of each of its edges (where a line can run)."""
    cp = 144 / ms.width
    cy, cx = (rc[0] + 0.5) * cp, (rc[1] + 0.5) * cp
    pts = [(cy + f * dy * cp / 2, cx + f * dx * cp / 2) for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)) for f in (0, 0.5, 0.95)]
    return sorted({int(min(y, 143) // 12) * 12 + int(min(x, 143) // 12) for y, x in pts})


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_boards, bs, seed = int(ic.get("num_boards", 60)), int(ic.get("batch_size", 12)), int(ic.get("seed", 0))
    kind = str(ic.get("model_kind", "path"))
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)
    if kind == "plain":
        from train_plain_trm import PlainTRM
        ck = torch.load(str(ic.plain_checkpoint), map_location="cpu", weights_only=False)
        plain = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16))); plain.load_state_dict(ck["ema"])
        dev_p = "cpu" if device.type == "mps" else device                 # MPS lacks the encoder's adaptive pool
        plain = plain.to(dev_p).eval()
    else:
        model = build_model(cfg, instantiate(cfg.diffusion))
        load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
        model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()

    rng = random.Random(seed)
    items = []                                                    # (board, variant name, sample, info)
    i = 0
    while len({b for b, *_ in items}) < n_boards and i < len(ds):
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is not None and ms.width == 8:
            vs = variants(ms, rng)
            if vs is not None:
                for name in KINDS:
                    s2 = dataclasses.replace(smp, spatial_conditions=ds._image_to_tensor(Image.fromarray(vs[name]["img"])))
                    items.append((i, name, s2, ms, vs[name]))
        i += 1

    rows = []
    for k0 in range(0, len(items), bs):
        chunk = items[k0:k0 + bs]
        with torch.no_grad():
            if kind == "plain":
                maze = torch.stack([s.spatial_conditions for _, _, s, _, _ in chunk]).to(dev_p)
                p_tok = torch.softmax(plain.think(maze).float(), -1)[..., 1].cpu().numpy()     # (B, 144)
                outs = [None] * len(chunk)
            else:
                cond = model._batch_to_sample(ds.collate_fn([s for _, _, s, _, _ in chunk]), device)
                gen = model.sampling_pipeline.sample_one_batch(model, cond, device,
                                                               generator=torch.Generator(device=device).manual_seed(seed + k0))
                outs = model.decode_for_eval(gen).cpu().clamp(0, 1)
        for j, (b, name, s, ms, v) in enumerate(chunk):
            if kind == "plain":
                pred = p_tok[j] > 0.5
                L_sp = render_labels(ds, ms, v["img"], v["nb"], v["sp"])
                L_def = render_labels(ds, ms, v["img"], v["nb"], v["def_"])
                drawn = set()
                tok = dict(tok_exact_sp=bool((pred == (L_sp == 1))[L_sp >= 0].all()),
                           tok_match_def=bool((pred == (L_def == 1))[L_def >= 0].all()),
                           tok_extra_kept=float(pred[(L_def == 1) & (L_sp == 0)].mean()) if ((L_def == 1) & (L_sp == 0)).any() else np.nan,
                           tok_other_wrong=int((pred & (L_def == 0)).sum()), p_tok=p_tok[j])
                img_out = None
            else:
                img = (outs[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_n = cv2.resize(img, ms.morig.shape[1::-1], interpolation=cv2.INTER_LINEAR)
                drawn = {ms.id2rc[c] for c in mcl.predicted_cells(img_n, ms) if c in ms.id2rc}
                img_out = img
            extra = v["def_"] - v["sp"]
            if kind != "plain":
                tok = {}
            rows.append(dict(**tok, board=b, variant=name, wall=v["wall"], drawn=drawn, sp=v["sp"], def_=v["def_"], sol=v["sol"],
                             exact=drawn == v["sp"], matches_def=drawn == v["def_"],
                             extra_drawn=len(drawn & extra) / len(extra) if extra else np.nan,
                             other_drawn=len(drawn - v["def_"]), sp_cov=len(drawn & v["sp"]) / len(v["sp"]),
                             img_in=(s.spatial_conditions.permute(1, 2, 0).numpy() * 255).astype(np.uint8), img_out=img_out))
        print(f"[{kind}] {min(k0 + bs, len(items))}/{len(items)}", flush=True)

    torch.save(rows, out_dir / f"loops_{kind}.pt")                         # incl. input / output images (144 px)
    print(f"\n[{kind}] {len({r['board'] for r in rows})} boards")
    for name in KINDS:
        R = [r for r in rows if r["variant"] == name]
        if kind == "plain":
            print(f"  {name:5s} n={len(R)} tokens: exact (= shortest path) {np.mean([r['tok_exact_sp'] for r in R]):4.0%}  "
                  f"= dead-end filling {np.mean([r['tok_match_def'] for r in R]):4.0%}  "
                  f"loop tokens kept {np.nanmean([r['tok_extra_kept'] for r in R]):4.0%}  "
                  f"other wrong tokens per board {np.mean([r['tok_other_wrong'] for r in R]):.2f}")
            continue
        print(f"  {name:5s} n={len(R)}: exact (= shortest path) {np.mean([r['exact'] for r in R]):4.0%}  "
              f"= what dead-end filling leaves {np.mean([r['matches_def'] for r in R]):4.0%}  "
              f"loop cells kept {np.nanmean([r['extra_drawn'] for r in R]):4.0%}  "
              f"other wrong cells per board {np.mean([r['other_drawn'] for r in R]):.2f}  "
              f"shortest path covered {np.mean([r['sp_cov'] for r in R]):4.0%}")


    # examples: 4 boards x (none, side, route): input with the removed wall marked, and what the model drew
    ex = [b for b in dict.fromkeys(r["board"] for r in rows)][:4]
    fig, axes = plt.subplots(len(ex), 3, figsize=(9.5, 3.3 * len(ex)), squeeze=False)
    for a, b in enumerate(ex):
        for c, name in enumerate(KINDS):
            r = next(x for x in rows if x["board"] == b and x["variant"] == name)
            ax = axes[a, c]
            ax.imshow(r["img_in"] if r["img_out"] is None else r["img_out"])
            cp = 144 / 8
            if kind == "plain":
                for t in np.nonzero(r["p_tok"] > 0.5)[0]:
                    ax.add_patch(plt.Rectangle(((t % 12) * 12 + 3, (t // 12) * 12 + 3), 6, 6, color="#2a78d6", alpha=0.8))
            for rc in r["def_"] - r["sp"]:
                ax.add_patch(plt.Rectangle((rc[1] * cp + 1, rc[0] * cp + 1), cp - 2, cp - 2, fill=False, ec="#eb6834", lw=1.4))
            if r["wall"] is not None:
                (r1, c1), (r2, c2) = r["wall"]
                ax.plot([(c1 + c2 + 1) / 2 * cp], [(r1 + r2 + 1) / 2 * cp], marker="o", ms=7, mfc="none", mec="#c3218f", mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            ok = r["tok_exact_sp"] if kind == "plain" else r["exact"]
            ax.set_title(f"board {b}, {name}: {'solved' if ok else 'wrong'}", fontsize=9)
    fig.suptitle(f"{'PaTh' if kind == 'path' else 'plain TRM'}: one wall removed (magenta circle); orange boxes = cells that "
                 f"dead-end filling cannot remove but are not on the shortest path", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / f"loops_examples_{kind}.png", dpi=130)


if __name__ == "__main__":
    main()
