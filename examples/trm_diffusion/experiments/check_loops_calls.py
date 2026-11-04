"""TEMPORARY check (Claude): what happens inside the models on the loop mazes of check_loops.py, call by call.

Same boards and removed walls as check_loops.py (seed 0). Per maze, the thinker's opinion on every token at every update:
  PaTh:  the analysis probe (ref_pt = prune_mech2.pt: w, b, trained on the last z_L at t = 70) on z_L after every z_L
         update, in the same setting as the pruning analysis (empty board noised to t = 70, conditional pass).
  plain: its own output head on z_H after every z_H update (no probe).
Token sets (rendered like the training targets): sp = shortest path, extra = kept by dead-end filling but not on the
shortest path, old = the old solution's tokens that are not on the shortest path, gap = tokens on the line through the opening.
Output: <out_dir>/loops_calls_<kind>.pt and a printed summary per loop type.
"""
import dataclasses
import random
from pathlib import Path

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
from check_loops import KINDS, render_labels, variants
from interp_prune_mech import CONFIG_DIR


def loop_type(name, v):
    if name in ("none", "side"):
        return name
    if v["def_"] == v["sol"]:
        return "shortcut"
    return "detour_unchanged" if v["sp"] == v["sol"] else "detour_shorter"


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_boards, bs, seed = int(ic.get("num_boards", 100)), int(ic.get("batch_size", 33)), int(ic.get("seed", 0))
    kind = str(ic.get("model_kind", "path"))
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

    rng = random.Random(seed)                                         # same walls as check_loops.py
    items = []
    i = 0
    while len({x[0] for x in items}) < n_boards and i < len(ds):
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is not None and ms.width == 8:
            vs = variants(ms, rng)
            if vs is not None:
                for name in KINDS:
                    v = vs[name]
                    cond_img = ds._image_to_tensor(Image.fromarray(v["img"]))
                    L = {k: render_labels(ds, ms, v["img"], v["nb"], cells) for k, cells in
                         (("sp", v["sp"]), ("def", v["def_"]), ("sol", v["sol"]))}
                    gap = (render_labels(ds, ms, v["img"], v["nb"], set(v["wall"])) == 1) if v["wall"] else np.zeros(144, bool)
                    sets = dict(sp=L["sp"] == 1, extra=(L["def"] == 1) & (L["sp"] == 0), old=(L["sol"] == 1) & (L["sp"] == 0),
                                gap=gap & (L["sp"] != 1) if v["wall"] else gap, gap_all=gap)
                    items.append((i, name, loop_type(name, v), dataclasses.replace(smp, spatial_conditions=cond_img), sets))
        i += 1

    st = {"on": False, "call": 0, "rec": []}

    def hook(module, args, kwargs, output):
        if not st["on"]:
            return None
        st["call"] += 1
        is_zh = st["call"] % per == 0
        if kind == "plain" and is_zh:
            st["rec"].append(torch.softmax(inner.lm_head(output).float(), -1)[..., 1].cpu())        # p(path), own head
        elif kind == "path" and not is_zh:
            st["rec"].append((output.float() @ w + b).cpu())                                        # probe logit
        return None
    inner.L_level.register_forward_hook(hook, with_kwargs=True)

    rows = []
    for k0 in range(0, len(items), bs):
        chunk = items[k0:k0 + bs]
        st.update(on=True, call=0, rec=[])
        with torch.no_grad():
            if kind == "plain":
                plain.think(torch.stack([s.spatial_conditions for _, _, _, s, _ in chunk]).to(device))
            else:
                batch = AmazeDataset.collate_fn([s for _, _, _, s, _ in chunk])
                cond = model._batch_to_sample(batch, device)
                empty = batch.spatial_conditions.to(device)                       # the (edited) maze, no path
                noise = torch.randn(empty.shape, generator=torch.Generator(device=device).manual_seed(seed + k0), device=device)
                tt = torch.full((len(chunk),), 70, device=device, dtype=torch.long)
                model.forward_with_carry(dataclasses.replace(cond, x_noisy=model.scheduler.add_noise(empty, noise, tt), timesteps=tt))
        st["on"] = False
        tr = torch.stack(st["rec"], 1).half()                                     # (B, n_updates, 144)
        for j, (bd, name, lt, _, sets) in enumerate(chunk):
            rows.append(dict(board=bd, variant=name, type=lt, trace=tr[j], **{f"set_{k}": v for k, v in sets.items()}))
        print(f"[{kind}] {len(rows)}/{len(items)}", flush=True)
    torch.save(rows, out_dir / f"loops_calls_{kind}.pt")

    hi = 1.39 if kind == "path" else 0.8                                          # logit of p = 0.8 / p = 0.8
    path_of = (lambda x: x > 0) if kind == "path" else (lambda x: x > 0.5)
    unit = "z_L update" if kind == "path" else "z_H update"
    print(f"\n[{kind}] share of tokens read as path ({unit}s 1, 3, 6, 12, 24, last); ever confident path; path at the end")
    for lt in ("none", "side", "shortcut", "detour_unchanged", "detour_shorter"):
        R = [r for r in rows if r["type"] == lt]
        if not R:
            continue
        for sk in ("sp", "extra", "old", "gap"):
            T = [r["trace"][:, r[f"set_{sk}"]].float().numpy() for r in R if r[f"set_{sk}"].any()]
            if not T:
                continue
            X = np.concatenate(T, 1)                                              # (n_updates, n_tokens)
            idx = [0, 2, 5, 11, 23, X.shape[0] - 1]
            print(f"  {lt:16s} {sk:5s} n_tok={X.shape[1]:4d}: " + " ".join(f"{path_of(X[k]).mean():4.0%}" for k in idx)
                  + f" | ever confident {np.mean((X > hi).any(0)):4.0%} | end {path_of(X[-1]).mean():4.0%}")


if __name__ == "__main__":
    main()
