"""TEMPORARY check (Claude): several starts, one goal. 8x8 test mazes with extra start markers (red dots, drawn like the
real start marker) on random cells off the solution, at least 2 cells away from it.

Right answer = the path from every start to the goal (their union, a subtree of the maze). Dead-end filling gives exactly
this: a cell with a marker is never a dead end. Variants per maze: none (the original), starts2 (1 extra start),
starts4 (3 extra starts). Same seed for both models.
PaTh: full sampling (20 steps, 1 attempt), drawn cells as the evaluator. plain: token level against the answers rendered
like the training targets. Plus the call-by-call view as in check_loops_calls.py (PaTh probe at t = 70, plain own head).
Output: <out_dir>/multistart_<kind>.pt, printed summary.
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

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from check_loops import bfs_path, render_labels, tree
from interp_prune_mech import CONFIG_DIR

VARIANTS = {"none": 0, "starts2": 1, "starts4": 3}


def start_marker(ms):
    """Centre and radius of the real start dot (the red blob nearest to the start cell)."""
    s = ms.morig.astype(int)
    red = ((s[..., 0] > 150) & (s[..., 1] < 110) & (s[..., 2] < 110)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(red)
    sc = np.array(mcl._center(ms.id2rc[ms.path[0]], ms.cell_px))
    k = 1 + int(np.argmin([np.linalg.norm(cent[j] - sc) for j in range(1, n)]))
    color = tuple(int(v) for v in ms.morig[lab == k].mean(0))
    return float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi)), color


def variants(ms, rng):
    nb, _ = tree(ms)
    rc = [ms.id2rc[p] for p in ms.path]
    s, g, sol = rc[0], rc[-1], set(rc)
    dist = {c: 0 for c in sol}; dq = deque(sol)
    while dq:
        u = dq.popleft()
        for v in nb[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; dq.append(v)
    pool = [c for c in nb if dist.get(c, 0) >= 2]
    if len(pool) < 3:
        return None
    extra_all = rng.sample(pool, 3)
    r, color = start_marker(ms)
    out = {}
    for name, k in VARIANTS.items():
        extra = extra_all[:k]
        img = ms.morig.copy()
        for c in extra:
            cv2.circle(img, mcl._center(c, ms.cell_px), int(round(r)), color, -1, lineType=cv2.LINE_AA)
        ans = set(sol)
        for c in extra:
            ans |= set(bfs_path(nb, c, g))
        out[name] = dict(img=img, nb=nb, ans=ans, sol=sol, extra=extra)
    return out


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    kind, bs, seed = str(ic.get("model_kind", "path")), int(ic.get("batch_size", 33)), int(ic.get("seed", 0))
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)
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
        ref = torch.load(str(ic.ref_pt), weights_only=False)
        w, b = ref["w"].float().to(device), float(ref["b"])
    inner = thinker.inner
    per = inner.config.L_cycles + 1

    rng, items = random.Random(seed), []
    for i in range(len(ds)):
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is None or ms.width != 8:
            continue
        vs = variants(ms, rng)
        if vs is None:
            continue
        for name, v in vs.items():
            items.append(dict(board=i, variant=name, ms=ms, v=v,
                              smp=dataclasses.replace(smp, spatial_conditions=ds._image_to_tensor(Image.fromarray(v["img"])))))

    st = {"on": False, "call": 0, "rec": []}

    def hook(module, args, kwargs, output):
        if not st["on"]:
            return None
        st["call"] += 1
        is_zh = st["call"] % per == 0
        if kind == "plain" and is_zh:
            st["rec"].append(torch.softmax(inner.lm_head(output).float(), -1)[..., 1].cpu())
        elif kind == "path" and not is_zh:
            st["rec"].append((output.float() @ w + b).cpu())
        return None
    inner.L_level.register_forward_hook(hook, with_kwargs=True)

    rows = []
    for k0 in range(0, len(items), bs):
        chunk = items[k0:k0 + bs]
        batch = AmazeDataset.collate_fn([it["smp"] for it in chunk])
        st.update(on=True, call=0, rec=[])
        with torch.no_grad():
            if kind == "plain":
                p_tok = torch.softmax(plain.think(batch.spatial_conditions.to(device)).float(), -1)[..., 1].cpu().numpy()
            else:
                cond = model._batch_to_sample(batch, device)
                empty = batch.spatial_conditions.to(device)
                noise = torch.randn(empty.shape, generator=torch.Generator(device=device).manual_seed(seed + k0), device=device)
                tt = torch.full((len(chunk),), 70, device=device, dtype=torch.long)
                model.forward_with_carry(dataclasses.replace(cond, x_noisy=model.scheduler.add_noise(empty, noise, tt), timesteps=tt))
        st["on"] = False
        trace = torch.stack(st["rec"], 1).half()
        if kind == "path":
            with torch.no_grad():
                gen = model.sampling_pipeline.sample_one_batch(model, cond, device, generator=torch.Generator(device=device).manual_seed(seed + k0))
            outs = (model.decode_for_eval(gen).cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        for j, it in enumerate(chunk):
            ms, v = it["ms"], it["v"]
            L_ans = render_labels(ds, ms, v["img"], v["nb"], v["ans"])
            L_sol = render_labels(ds, ms, v["img"], v["nb"], v["sol"])
            branch = (L_ans == 1) & (L_sol == 0)                         # tokens of the paths from the extra starts
            r = dict(board=it["board"], variant=it["variant"], extra=v["extra"], ans=v["ans"], sol=v["sol"], trace=trace[j],
                     set_branch=branch, set_sol=L_sol == 1,
                     img_in=(it["smp"].spatial_conditions.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            if kind == "plain":
                pred = p_tok[j] > 0.5
                r.update(p_tok=p_tok[j], exact=bool((pred == (L_ans == 1))[L_ans >= 0].all()),
                         only_old=bool((pred == (L_sol == 1))[L_sol >= 0].all()) and it["variant"] != "none",
                         branch_kept=float(pred[branch].mean()) if branch.any() else np.nan)
            else:
                img_n = cv2.resize(outs[j], ms.morig.shape[1::-1], interpolation=cv2.INTER_LINEAR)
                drawn = {ms.id2rc[c] for c in mcl.predicted_cells(img_n, ms) if c in ms.id2rc}
                extra_cells = v["ans"] - v["sol"]
                r.update(img_out=outs[j], drawn=drawn, exact=drawn == v["ans"], only_old=drawn == v["sol"] and it["variant"] != "none",
                         branch_kept=len(drawn & extra_cells) / len(extra_cells) if extra_cells else np.nan)
            rows.append(r)
        print(f"[{kind}] {len(rows)}/{len(items)}", flush=True)
    torch.save(rows, out_dir / f"multistart_{kind}.pt")
    thr = 0.0 if kind == "path" else 0.5
    print(f"\n[{kind}] {len({r['board'] for r in rows})} boards")
    for name in VARIANTS:
        R = [r for r in rows if r["variant"] == name]
        T = [r["trace"][:, r["set_branch"]].float().numpy() > thr for r in R if r["set_branch"].any()]
        tr = (f" | branch tokens path: ever {np.mean(np.concatenate([t.any(0) for t in T])):.0%}, end "
              f"{np.mean(np.concatenate([t[-1] for t in T])):.0%}") if T else ""
        print(f"  {name:8s} n={len(R)}: right answer {np.mean([r['exact'] for r in R]):4.0%}  only the original path "
              f"{np.mean([r['only_old'] for r in R]):4.0%}  extra-start paths drawn {np.nanmean([r['branch_kept'] for r in R]) if name != 'none' else float('nan'):4.0%}" + tr)


if __name__ == "__main__":
    main()
