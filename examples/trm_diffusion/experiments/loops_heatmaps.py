"""TEMPORARY (Claude): probe_corrections-style heatmaps on the loop mazes of check_loops.py (seed 0), PaTh or plain TRM.

Probe as in ZPathEncodingAblation: its MLP probe (_train_probe, use_nonlinearity=True) on the LAST z_L update, trained on
the unedited mazes (2-fold: even boards -> applied to odd boards' mazes and the other way round), then applied to every
z_L update of the loop mazes. PaTh: empty board noised to t = 70, conditional pass (as the probe analysis). plain: maze only.
Figure (one per model): 2 mazes of each loop kind (the first ones, not picked by result). Top: the edited maze with the
right answer, removed wall (magenta circle), orange box = dropped (p > 0.8 twice in a row, then < 0.2 at the end of the
window, as in _plot_corrections), magenta box = still path at the end of the pass but not on the shortest path.
Bottom: z_L update of the drop (viridis), light blue = shortest path.
Output: <out_dir>/loops_heatmap_<kind>.png, loops_heatmap_<kind>.pt
"""
import dataclasses
import random
from pathlib import Path

import cv2
import hydra
import matplotlib
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation as Z
from factory import build_model
from check_loops import KINDS, render_labels, variants
from check_loops_calls import loop_type
from interp_prune_mech import CONFIG_DIR

TYPES = ["side", "shortcut", "detour_unchanged", "detour_shorter"]
SHORT = {"side": "A. loop away", "shortcut": "B. shortcut", "detour_unchanged": "C. longer new route",
         "detour_shorter": "D. shorter new route"}


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    kind, seed, bs = str(ic.get("model_kind", "path")), int(ic.get("seed", 0)), int(ic.get("batch_size", 44))
    window = int(ic.get("window", 36))
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
    inner = thinker.inner
    per = inner.config.L_cycles + 1

    rng, items = random.Random(seed), []                          # same walls as check_loops.py
    for i in range(len(ds)):
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is not None and ms.width == 8:
            vs = variants(ms, rng)
            if vs is not None:
                for name in KINDS:
                    v = vs[name]
                    items.append(dict(board=i, variant=name, type=loop_type(name, v), ms=ms, v=v,
                                      smp=dataclasses.replace(smp, spatial_conditions=ds._image_to_tensor(Image.fromarray(v["img"])))))
    boards = sorted({it["board"] for it in items})
    fold = {b: k % 2 for k, b in enumerate(boards)}

    st = {"mode": None, "call": 0, "out": [], "probes": None}

    def hook(module, args, kwargs, output):
        st["call"] += 1
        if st["call"] % per == 0:
            return None
        if st["mode"] == "last":
            st["out"] = [output.detach().float()]
        else:
            B, T, D = output.shape
            x = output.float().reshape(-1, D)
            st["out"].append(torch.stack([torch.sigmoid(p(x)).reshape(B, T) for p in st["probes"]], 0).half().cpu())
        return None
    inner.L_level.register_forward_hook(hook, with_kwargs=True)

    def run(chunk, mode):
        st.update(mode=mode, call=0, out=[])
        with torch.no_grad():
            if kind == "plain":
                plain.think(torch.stack([it["smp"].spatial_conditions for it in chunk]).to(device))
            else:
                batch = AmazeDataset.collate_fn([it["smp"] for it in chunk])
                cond = model._batch_to_sample(batch, device)
                empty = batch.spatial_conditions.to(device)
                noise = torch.randn(empty.shape, generator=torch.Generator(device=device).manual_seed(seed + chunk[0]["board"]), device=device)
                tt = torch.full((len(chunk),), 70, device=device, dtype=torch.long)
                model.forward_with_carry(dataclasses.replace(cond, x_noisy=model.scheduler.add_noise(empty, noise, tt), timesteps=tt))
        return st["out"][-1].cpu() if mode == "last" else torch.stack(st["out"], 2)        # (2, B, n_upd, T)

    # probes: trained on the last z_L of the unedited mazes, one per fold
    plain_items = [it for it in items if it["variant"] == "none"]
    Zl = torch.cat([run(plain_items[k:k + bs], "last") for k in range(0, len(plain_items), bs)])
    lab = Z._token_labels(torch.stack([it["smp"].images for it in plain_items]), 12, 0.1)
    probes = []
    for f in range(2):
        sel = [k for k, it in enumerate(plain_items) if fold[it["board"]] == f]
        y = lab[sel].flatten(); keep = y >= 0
        probes.append(Z._train_probe(Zl[sel].flatten(0, 1)[keep], y[keep], use_nonlinearity=True, steps=3000, device=device))
    st["probes"] = probes
    P = []
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]
        pp = run(chunk, "apply")
        for j, it in enumerate(chunk):
            P.append(pp[1 - fold[it["board"]], j])                                     # the probe NOT trained on this board
        print(f"[{kind}] {min(k + bs, len(items))}/{len(items)}", flush=True)
    P = torch.stack(P).float()                                                         # (n_items, n_upd, 144)

    # dropped (as _plot_corrections), drop update, still path at the end
    w = P[:, :window]
    high2 = (w[:, 1:] > 0.8) & (w[:, :-1] > 0.8)
    dropped = high2.any(1) & (w[:, -2:] < 0.2).all(1)
    crossed = (w[:, :-1] >= 0.2) & (w[:, 1:] < 0.2)
    steps = torch.arange(1, window).view(1, -1, 1)
    drop_update = torch.where(crossed, steps, torch.zeros_like(steps)).amax(1)
    end_path = P[:, -1] > 0.5
    rec = []
    for k, it in enumerate(items):
        ms, v = it["ms"], it["v"]
        L_sp = render_labels(ds, ms, v["img"], v["nb"], v["sp"])
        rec.append(dict(board=it["board"], variant=it["variant"], type=it["type"], p=P[k].half(), L_sp=L_sp,
                        dropped=dropped[k].numpy(), drop_update=drop_update[k].numpy(),
                        kept_wrong=(end_path[k].numpy() & (L_sp == 0))))
    torch.save(rec, out_dir / f"loops_heatmap_{kind}.pt")

    cols = []
    for t in TYPES:
        cols += [k for k, it in enumerate(items) if it["type"] == t][:2]
    fig, axes = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 4.9), squeeze=False)
    for c, k in enumerate(cols):
        it, r = items[k], rec[k]
        ms, v = it["ms"], it["v"]
        img = v["img"].copy()
        for a in v["sp"]:
            for b in v["nb"][a]:
                if b in v["sp"] and a < b:
                    cv2.line(img, mcl._center(a, ms.cell_px), mcl._center(b, ms.cell_px), ms.blue_color, ms.stroke)
        img = np.asarray(Image.fromarray(img).resize((144, 144), Image.BICUBIC))
        ax = axes[0][c]
        ax.imshow(img)
        for tok in np.nonzero(r["dropped"])[0]:
            ax.add_patch(Rectangle(((tok % 12) * 12, (tok // 12) * 12), 12, 12, fill=False, edgecolor="tab:orange", linewidth=1.2))
        for tok in np.nonzero(r["kept_wrong"])[0]:
            ax.add_patch(Rectangle(((tok % 12) * 12 + 1.5, (tok // 12) * 12 + 1.5), 9, 9, fill=False, edgecolor="#c3218f", linewidth=1.4))
        (r1, c1), (r2, c2) = v["wall"]
        cp = 144 / 8
        ax.add_patch(Circle(((c1 + c2 + 1) / 2 * cp, (r1 + r2 + 1) / 2 * cp), cp * 0.45, fill=False, ec="#c3218f", lw=1.8, ls="--"))
        ax.set_title(f"{SHORT[it['type']]}\nboard {it['board']}: {int(r['dropped'].sum())} dropped, {int(r['kept_wrong'].sum())} extra", fontsize=8)
        ax = axes[1][c]
        ax.imshow((r["L_sp"] == 1).reshape(12, 12), cmap="Blues", vmin=0, vmax=4)
        dm = np.ma.masked_where(~r["dropped"].reshape(12, 12), r["drop_update"].reshape(12, 12))
        im = ax.imshow(dm, cmap="viridis", vmin=1, vmax=window - 1)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[1][0].set_ylabel(f"z_L update of the drop\n(dark = 1 ... yellow = {window - 1})", fontsize=7)
    fig.colorbar(im, ax=axes[1].tolist(), fraction=0.02)
    who = "PaTh thinker (t=70)" if kind == "path" else "plain TRM"
    fig.suptitle(f"{who}, mazes with one removed wall (dashed circle): self-corrections in the first {window} z_L updates. "
                 "Orange box: dropped (p > 0.8, then < 0.2); magenta box: still path at the end but not on the shortest path",
                 fontsize=9)
    fig.savefig(out_dir / f"loops_heatmap_{kind}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
