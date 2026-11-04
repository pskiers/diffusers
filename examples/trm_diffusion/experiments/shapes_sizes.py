"""TEMPORARY (Claude): PaTh / plain TRM on every shape and size of the amaze test sets.

Per test set (<data_root>/<shape>/n<size>_test.parquet, 100 mazes):
  accuracy  PaTh: full sampling (20 steps, 1 attempt), drawn cells as the evaluator (blue in the cell), exact = the drawn
            cells are the solution cells; per-cell errors with the cell's area (for the circle question).
            plain: its own output, token level against the target labels (ignored tokens skipped).
  pruning   the MLP probe of ZPathEncodingAblation on the last z_L (2-fold cross-fit inside the set), applied to the first
            `n_upd` z_L updates (PaTh: empty board noised to t = 70; plain: the maze). Dropped as in _plot_corrections
            (p > 0.8 twice in a row, < 0.2 at the end of the window). For dropped tokens of side-branch cells: drop update
            vs cells from the dead-end tip, and how often two neighbouring dropped cells drop at the same update.
  maze graph for every shape from the cell map: two cells are connected if their regions touch and the boundary is not
            covered by a wall (dark pixels, dilated by 4 px). Checked: a tree, and the solution path is connected.
Output: <out_dir>/<kind>/<shape>_n<size>.pt and summary_<kind>.txt
"""
import dataclasses
import json
from collections import deque
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.stats import spearmanr

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation as Z
from factory import build_model
from interp_prune_mech import CONFIG_DIR


def maze_graph(m):
    meta = json.loads(m["metadata"]) if isinstance(m["metadata"], str) else m["metadata"]
    ids = mcl.decode_cell_ids(m["cell_map"])
    dark = cv2.dilate((mcl.to_rgb(m["m_original_img"]).max(2) < 128).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    P = {}
    for a, c, da, dc in ((ids[:, :-1], ids[:, 1:], dark[:, :-1], dark[:, 1:]), (ids[:-1], ids[1:], dark[:-1], dark[1:])):
        msk = (a != c) & (a > 0) & (c > 0)
        for x, y, d in zip(a[msk], c[msk], (da | dc)[msk]):
            P.setdefault((int(min(x, y)), int(max(x, y))), []).append(bool(d))
    cells = [int(c) for c in np.unique(ids) if c > 0]
    nb = {c: [] for c in cells}
    for (a, b), v in P.items():
        if len(v) >= 10 and np.mean(v) < 0.5:
            nb[a].append(b); nb[b].append(a)
    path = [int(c) for c in meta["path_cell_ids"]]
    ok = (sum(map(len, nb.values())) // 2 == len(cells) - 1) and all(b in nb[a] for a, b in zip(path[:-1], path[1:]))
    sol = set(path)
    tips = [c for c in cells if c not in sol and len(nb[c]) == 1]
    d_tip = {c: 0 for c in tips}; dq = deque(tips)
    while dq:
        u = dq.popleft()
        for v in nb[u]:
            if v not in sol and v not in d_tip:
                d_tip[v] = d_tip[u] + 1; dq.append(v)
    area = {c: int(n) for c, n in zip(*np.unique(ids, return_counts=True)) if c > 0}
    H, W = ids.shape
    tok_cell = np.array([int(ids[min(H - 1, int((r * 12 + 6) * H / 144)), min(W - 1, int((c * 12 + 6) * W / 144))])
                         for r in range(12) for c in range(12)])
    return dict(ok=ok, nb=nb, sol=sol, d_tip=d_tip, area=area, tok_cell=tok_cell, ids=ids)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    kind = str(ic.get("model_kind", "path"))
    out_dir = Path(str(ic.out_dir)) / kind; out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(str(ic.data_root))
    shapes = list(ic.get("shapes", ["square", "circle", "hexagon", "triangle"]))
    sizes = [int(s) for s in ic.get("sizes", [5, 7, 8, 9, 10, 11, 13, 16])]
    n_upd, window, bs, seed = int(ic.get("n_upd", 72)), int(ic.get("window", 36)), int(ic.get("batch_size", 50)), 0
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    if kind == "plain":
        from train_plain_trm import PlainTRM
        ck = torch.load(str(ic.plain_checkpoint), map_location="cpu", weights_only=False)
        plain = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16))); plain.load_state_dict(ck["ema"]); plain = plain.to(device).eval()
        thinker = plain.thinker
    else:
        model = build_model(cfg, instantiate(cfg.diffusion))
        load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
        model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
        model.scheduler.set_timesteps(20, device=device)
        thinker = model.thinker
    inner = thinker.inner
    per = inner.config.L_cycles + 1
    st = {"mode": None, "call": 0, "k": 0, "out": [], "probes": None}

    def hook(module, args, kwargs, output):
        if st["mode"] is None:
            return None
        st["call"] += 1
        if st["call"] % per == 0:
            return None
        st["k"] += 1
        if st["mode"] == "last":
            st["out"] = [output.detach().float()]
        elif st["k"] <= n_upd:
            B, T, D = output.shape
            x = output.float().reshape(-1, D)
            st["out"].append(torch.stack([torch.sigmoid(p(x)).reshape(B, T) for p in st["probes"]]).half().cpu())
        return None
    inner.L_level.register_forward_hook(hook, with_kwargs=True)

    def t70(batch, mode):
        st.update(mode=mode, call=0, k=0, out=[])
        with torch.no_grad():
            if kind == "plain":
                logits = plain.think(batch.spatial_conditions.to(device))
                st["final"] = torch.softmax(logits.float(), -1)[..., 1].cpu()
            else:
                cond = model._batch_to_sample(batch, device)
                empty = batch.spatial_conditions.to(device)
                noise = torch.randn(empty.shape, generator=torch.Generator(device=device).manual_seed(seed), device=device)
                tt = torch.full((empty.shape[0],), 70, device=device, dtype=torch.long)
                model.forward_with_carry(dataclasses.replace(cond, x_noisy=model.scheduler.add_noise(empty, noise, tt), timesteps=tt))
        st["mode"] = None
        return st["out"][-1].cpu() if mode == "last" else torch.stack(st["out"], 2)       # (2, B, n_upd, T)

    lines = []
    for shape in shapes:
        for size in sizes:
            p = root / shape / f"n{size}_test.parquet"
            if not p.is_file():
                continue
            ds = AmazeDataset(dataset_path=str(p), split="test", image_size=144, condition_field="m_original_img",
                              target_field="sol_img", num_channels=3, include_metadata=True)
            samples, graphs = [], []
            for i in range(len(ds)):
                s = ds[i]
                try:
                    g = maze_graph(s.metadata)
                except Exception:
                    continue
                if g["ok"]:
                    samples.append(s); graphs.append(g)
                if len(samples) >= int(ic.get("max_mazes", 100)):
                    break
            N = len(samples)
            chunks = [list(range(k, min(k + bs, N))) for k in range(0, N, bs)]
            batches = [AmazeDataset.collate_fn([samples[i] for i in ch]) for ch in chunks]
            labels = torch.cat([Z._token_labels(b.images, 12, 0.1) for b in batches])
            Zl = torch.cat([t70(b, "last") for b in batches])
            fold = np.arange(N) % 2
            probes = []
            for f in range(2):
                sel = np.nonzero(fold == f)[0]
                y = labels[sel].flatten(); keep = y >= 0
                probes.append(Z._train_probe(Zl[sel].flatten(0, 1)[keep], y[keep], use_nonlinearity=True, steps=3000, device=device))
            st["probes"] = probes
            Pp, fin = [], []
            for b in batches:
                Pp.append(t70(b, "apply"))
                if kind == "plain":
                    fin.append(st["final"])
            Pp = torch.cat(Pp, 1)                                                              # (2, N, n_upd, T)
            P = torch.stack([Pp[1 - fold[i], i] for i in range(N)]).float()                     # the probe not fit on it

            # accuracy
            rec = dict(shape=shape, size=size, n=N, labels=labels.half(), sol=torch.cat([(b.images * 255).round().byte() for b in batches]),
                       maze=torch.cat([(b.spatial_conditions * 255).round().byte() for b in batches]), p=P.half())
            if kind == "plain":
                pred = torch.cat(fin) > 0.5
                exact = [bool((pred[i] == (labels[i] == 1))[labels[i] >= 0].all()) for i in range(N)]
                rec["pred"] = pred
            else:
                outs = []
                for k0, b in zip(range(0, N, bs), batches):
                    cond = model._batch_to_sample(b, device)
                    with torch.no_grad():
                        gen = model.sampling_pipeline.sample_one_batch(model, cond, device, generator=torch.Generator(device=device).manual_seed(seed + k0))
                    outs.append((model.decode_for_eval(gen).cpu().clamp(0, 1) * 255).round().byte())
                outs = torch.cat(outs); rec["out"] = outs
                exact, cell_err = [], []
                for i, g in enumerate(graphs):
                    img = cv2.resize(outs[i].permute(1, 2, 0).numpy(), g["ids"].shape[::-1], interpolation=cv2.INTER_LINEAR)
                    bm = mcl.blue_mask(img, morph_open=True)
                    drawn = set(int(c) for c in np.unique(g["ids"][bm]) if c > 0)
                    exact.append(drawn == g["sol"])
                    for c, a in g["area"].items():
                        cell_err.append((i, c, a, c in g["sol"], (c in drawn) != (c in g["sol"])))
                rec["cell_err"] = np.array(cell_err, dtype=np.int64)
            rec["exact"] = np.array(exact)

            # pruning: drop update of tokens, side-branch cells only
            w = P[:, :window]
            high2 = (w[:, 1:] > 0.8) & (w[:, :-1] > 0.8)
            dropped = high2.any(1) & (w[:, -2:] < 0.2).all(1)
            crossed = (w[:, :-1] >= 0.2) & (w[:, 1:] < 0.2)
            steps = torch.arange(1, window).view(1, -1, 1)
            drop_upd = torch.where(crossed, steps, torch.zeros_like(steps)).amax(1)
            rec.update(dropped=dropped, drop_upd=drop_upd)
            dt, du, same, pairs, tok_dtip = [], [], 0, 0, np.full((N, 144), -1)
            for i, g in enumerate(graphs):
                cell_drop = {}
                for t in range(144):
                    c = int(g["tok_cell"][t])
                    if c in g["d_tip"]:
                        tok_dtip[i, t] = g["d_tip"][c]
                        if dropped[i, t] and labels[i, t] == 0:
                            dt.append(g["d_tip"][c]); du.append(int(drop_upd[i, t]))
                            cell_drop[c] = min(cell_drop.get(c, 99), int(drop_upd[i, t]))
                for c in cell_drop:
                    for n_ in g["nb"][c]:
                        if n_ in cell_drop and c < n_:
                            pairs += 1; same += cell_drop[c] == cell_drop[n_]
            rec["tok_dtip"] = tok_dtip
            rho = spearmanr(dt, du)[0] if len(dt) > 5 else np.nan
            slope = np.polyfit(dt, du, 1)[0] if len(set(dt)) > 1 else np.nan
            rec["order"] = dict(n_drops=len(dt), rho=rho, upd_per_cell=slope, same_frac=same / max(pairs, 1))
            torch.save(rec, out_dir / f"{shape}_n{size}.pt")
            line = (f"{kind:5s} {shape:8s} n{size:2d}: {N:3d} mazes, exact {np.mean(exact):5.1%} | side-branch drops {len(dt):4d}, "
                    f"drop vs cells from tip rho {rho:+.2f}, {slope:.2f} z_L updates per cell, neighbours dropping together "
                    f"{same / max(pairs, 1):.0%}")
            print(line, flush=True); lines.append(line)
    (out_dir / f"summary_{kind}.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
