"""TEMPORARY (Claude): does PaTh's thinker keep the lollipop because of the noisy image x_t or because of the maze it is
conditioned on? Mismatched inputs, thinker only (conditional pass of one denoising step, as interp_loops_first.py).

Mazes as interp_loops_first.py (maze_size, side_pick, min_side, seed 0). Images per board:
  maze_none / maze_loop      the unedited maze / the maze with the side loop (one wall removed), no path
  loop_sp / loop_def         the loop maze with the shortest path drawn / with everything dead-end filling keeps drawn
                             (= the shortest path + the lollipop), drawn like the training targets
Conditions at t in {99, 70, 30} (x_t = the image noised to t with a fixed seed per batch):
  base           conditioning maze_loop, x_t from maze_loop
  xt_unedited    conditioning maze_loop, x_t from maze_none
  cond_unedited  conditioning maze_none, x_t from maze_loop
  xt_sp          conditioning maze_loop, x_t from loop_sp
  xt_def         conditioning maze_loop, x_t from loop_def
Readout: linear probe on the last z_L, fit per t on (maze_none, maze_none) runs, 2-fold over boards.
E = lollipop tokens (of the loop maze). Output: <out_dir>/loops_xt.pt, printed summary.
"""
import dataclasses
import logging
import random
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
from check_loops import render_labels, variants
from interp_prune_mech import CONFIG_DIR
from interp_prune_mech3 import train_linear

CONDS = {"base": ("maze_loop", "maze_loop"), "xt_unedited": ("maze_loop", "maze_none"), "cond_unedited": ("maze_none", "maze_loop"),
         "xt_sp": ("maze_loop", "loop_sp"), "xt_def": ("maze_loop", "loop_def")}
SHOW = [5, 11, 17, 35]


def draw(ms, img, nbk, cells):
    out = img.copy()
    for a in cells:
        for b in nbk[a]:
            if b in cells and a < b:
                cv2.line(out, mcl._center(a, ms.cell_px), mcl._center(b, ms.cell_px), ms.blue_color, ms.stroke)
    return out


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_boards, bs, seed = int(ic.get("num_boards", 100)), int(ic.get("batch_size", 33)), int(ic.get("seed", 0))
    maze_size, side_pick, min_side = int(ic.get("maze_size", 8)), str(ic.get("side_pick", "random")), int(ic.get("min_side", 0))
    ts = [int(x) for x in str(ic.get("ts", "99,70,30")).split(",")]
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
    model.scheduler.set_timesteps(20, device=device)
    inner = model.thinker.inner
    per = inner.config.L_cycles + 1

    rng = random.Random(seed)
    boards = []
    i = 0
    while i < len(ds) and len(boards) < n_boards:
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is not None and ms.width == maze_size:
            vs = variants(ms, rng, side_pick, min_side)
            if vs is not None:
                v, n0 = vs["side"], vs["none"]
                lab = {k: render_labels(ds, ms, vs[k]["img"], vs[k]["nb"], vs[k]["sp"]) for k in ("none", "side")}
                E = (render_labels(ds, ms, v["img"], v["nb"], v["def_"]) == 1) & (lab["side"] == 0)
                T_ = lambda im: ds._image_to_tensor(Image.fromarray(im))
                imgs = dict(maze_none=T_(n0["img"]), maze_loop=T_(v["img"]), loop_sp=T_(draw(ms, v["img"], v["nb"], v["sp"])),
                            loop_def=T_(draw(ms, v["img"], v["nb"], v["def_"])))
                boards.append(dict(board=i, fold=len(boards) % 2, E=E, lab_none=lab["none"], lab_side=lab["side"], smp=smp, imgs=imgs))
        i += 1
    logging.info(f"{len(boards)} boards")

    st = {"on": False}

    def pre_hook(module, args):
        if st["on"]:
            st["call"] += 1

    def level_hook(module, args, kwargs, output):
        if st["on"] and st["call"] % per != 0:
            o = output.float()
            if st["mode"] == "fit":
                st["last"] = o
            else:
                st["logits"].append(((o * st["W"][:, None, :]).sum(-1) + st["Bv"][:, None]).half().cpu())
        return None

    inner.L_level.register_forward_pre_hook(pre_hook)
    inner.L_level.register_forward_hook(level_hook, with_kwargs=True)

    def run(chunk, k0, cond_key, xt_key, t, mode, probes=None):
        if probes is not None:
            folds = torch.tensor([b["fold"] for b in chunk], device=device)
            st["W"] = torch.stack(probes[0])[1 - folds]; st["Bv"] = torch.tensor(probes[1], device=device)[1 - folds]
        st.update(on=True, mode=mode, call=0, logits=[])
        try:
            with torch.no_grad():
                smps = [dataclasses.replace(b["smp"], spatial_conditions=b["imgs"][cond_key]) for b in chunk]
                batch = AmazeDataset.collate_fn(smps)
                c_ = model._batch_to_sample(batch, device)
                base = torch.stack([b["imgs"][xt_key] for b in chunk]).to(device)
                noise = torch.randn(base.shape, generator=torch.Generator(device=device).manual_seed(seed + k0), device=device)
                tt = torch.full((len(chunk),), t, device=device, dtype=torch.long)
                model.forward_with_carry(dataclasses.replace(c_, x_noisy=model.scheduler.add_noise(base, noise, tt), timesteps=tt))
        finally:
            st["on"] = False

    res = {}
    for t in ts:
        X, Y = [], []
        for k0 in range(0, len(boards), bs):
            chunk = boards[k0:k0 + bs]
            run(chunk, k0, "maze_none", "maze_none", t, "fit")
            X.append(st["last"].cpu()); Y.append(torch.tensor(np.stack([b["lab_none"] for b in chunk])))
        X, Y = torch.cat(X), torch.cat(Y)
        F_ = torch.tensor([b["fold"] for b in boards])
        pw, pb = [], []
        for f in (0, 1):
            w_, b_ = train_linear(X[F_ == f].flatten(0, 1), Y[F_ == f].flatten(), int(ic.get("probe_steps", 3000)), device)
            pw.append(w_.to(device)); pb.append(float(b_))
        del X
        for name, (ck, xk) in CONDS.items():
            L = []
            for k0 in range(0, len(boards), bs):
                run(boards[k0:k0 + bs], k0, ck, xk, t, "rec", (pw, pb))
                L.append(torch.stack(st["logits"], 1))
            res[(t, name)] = torch.cat(L)                                         # (n_boards, U, T)
            logging.info(f"t={t} {name}")
    torch.save(dict(res=res, boards=[{k: b[k] for k in ("board", "fold", "E", "lab_none", "lab_side")} for b in boards]),
               out_dir / "loops_xt.pt")

    E = torch.tensor(np.stack([b["E"] for b in boards]))
    lab = torch.tensor(np.stack([b["lab_side"] for b in boards]))
    sp, known = lab == 1, lab >= 0
    print(f"\n[path] n{maze_size} side_pick={side_pick} min_side={min_side}: {len(boards)} boards, {int(E.sum())} lollipop tokens")
    print(f"share of lollipop tokens read as path at z_L updates {SHOW} | end || shortest path covered at the end | boards exact (vs the loop maze's answer)")
    for t in ts:
        for name in CONDS:
            L = res[(t, name)].float()
            fin = L[:, -1] > 0
            exact = ((fin == sp) | ~known).all(1).float().mean()
            print(f"   t={t:2d} {name:13s} " + " ".join(f"{(L[:, k][E] > 0).float().mean():4.0%}" for k in SHOW)
                  + f" | {fin[E].float().mean():4.0%} || {fin[sp].float().mean():4.0%} | {exact:4.0%}")


if __name__ == "__main__":
    main()
