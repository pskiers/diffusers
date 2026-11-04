"""TEMPORARY (Claude): does z_H carry the decision to drop a lollipop (side loop)? PaTh thinker or plain TRM.

Mazes as interp_loops_first.py: square test set of size maze_size, side loop from check_loops.variants (side_pick,
min_side, seed 0); per board the unedited maze ("none") and the maze with the lollipop ("side").
E = lollipop tokens (kept by dead-end filling, not on the shortest path). Readout: linear probe on the last z_L of the
unedited mazes, 2-fold over boards (as interp_loops_first.py).
Every z_H update happens in an L_level call c with c % (L_cycles + 1) == 0 (calls 7, 14, 21, ...). Conditions on the
lollipop maze (the unedited maze is run first with the same noise, and its z_H after every z_H update is recorded):
  base          no change
  skip_zH7/14   the z_H update at call 7 / 14 is skipped for the whole board (z_H stays as it was)
  swap_E        after every z_H update, the E tokens' z_H := the unedited maze's z_H at the same update
  swap_rest     the same for every token except E
  swap_all      the same for every token (z_H taken entirely from the unedited maze; z_L sees the lollipop maze)
  freeze_E14    the E tokens' z_H is not updated from call 14 on
Output: <out_dir>/loops_zh_<kind>.pt / .png, printed summary.
"""
import dataclasses
import logging
import random
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

CONDS = {"base": {}, "skip_zH7": {"skip": 7}, "skip_zH14": {"skip": 14}, "swap_E": {"swap": "E"},
         "swap_rest": {"swap": "rest"}, "swap_all": {"swap": "all"}, "freeze_E14": {"freeze": 14}}
SHOW = [5, 11, 17, 23, 35]


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_boards, bs, seed = int(ic.get("num_boards", 100)), int(ic.get("batch_size", 33)), int(ic.get("seed", 0))
    maze_size, side_pick, min_side = int(ic.get("maze_size", 8)), str(ic.get("side_pick", "random")), int(ic.get("min_side", 0))
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    kind = str(ic.get("model_kind", "path"))
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3, include_metadata=True)
    if kind == "plain":
        from train_plain_trm import PlainTRM
        ck = torch.load(str(ic.plain_checkpoint), map_location="cpu", weights_only=False)
        plain = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16)), noisy=bool(ck["args"].get("noisy", False)))
        plain.load_state_dict(ck["ema"]); plain = plain.to(device).eval()
        thinker = plain.thinker
        logging.info(f"[plain] {ic.plain_checkpoint}: step {ck['step']}, noisy={plain.noisy}, loss={ck['args'].get('loss', 'ce')}")
    else:
        model = build_model(cfg, instantiate(cfg.diffusion))
        load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
        model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
        model.scheduler.set_timesteps(20, device=device)
        thinker = model.thinker
    inner = thinker.inner
    per = inner.config.L_cycles + 1

    # ---------------------------------------------------------------- mazes (same as interp_loops_first.py)
    rng = random.Random(seed)
    boards = []
    i = 0
    while i < len(ds) and len(boards) < n_boards:
        smp = ds[i]
        ms = mcl.build_maze_sample(smp.metadata)
        if ms is not None and ms.width == maze_size:
            vs = variants(ms, rng, side_pick, min_side)
            if vs is not None:
                lab = {k: render_labels(ds, ms, vs[k]["img"], vs[k]["nb"], vs[k]["sp"]) for k in ("none", "side")}
                E = (render_labels(ds, ms, vs["side"]["img"], vs["side"]["nb"], vs["side"]["def_"]) == 1) & (lab["side"] == 0)
                mk = lambda k: dataclasses.replace(smp, spatial_conditions=ds._image_to_tensor(Image.fromarray(vs[k]["img"])))
                boards.append(dict(board=i, fold=len(boards) % 2, E=E, lab_none=lab["none"], lab_side=lab["side"],
                                   none=mk("none"), side=mk("side")))
        i += 1
    logging.info(f"{len(boards)} boards")

    # ---------------------------------------------------------------- hooks
    st = {"on": False}

    def dot(x):
        return (x.float() * st["W"][:, None, :]).sum(-1)

    def pre_hook(module, args):
        if st["on"]:
            st["call"] += 1

    def level_hook(module, args, kwargs, output):
        if not st["on"]:
            return None
        c = st["call"]
        z_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
        if c % per != 0:                                                    # z_L update
            if st["mode"] == "fit":
                st["last_zl"] = output.detach().float()
            else:
                st["logits"].append((dot(output) + st["Bv"][:, None]).cpu())
            return None
        st["zh_k"] += 1                                                     # z_H update
        if st["mode"] == "rec_none":
            st["zh_ref"].append(output.detach().clone())
            return None
        if st["mode"] != "cond":
            return None
        cd = st["cond"]
        if cd.get("skip") == c:
            return z_in
        if "swap" in cd:
            m = {"E": st["E"], "rest": ~st["E"], "all": torch.ones_like(st["E"])}[cd["swap"]]
            out = output.clone()
            out[m] = st["zh_ref"][st["zh_k"]][m].to(out.dtype)
            return out
        if "freeze" in cd and c >= cd["freeze"]:
            out = output.clone()
            out[st["E"]] = z_in[st["E"]].to(out.dtype)
            return out
        return None

    inner.L_level.register_forward_pre_hook(pre_hook)
    inner.L_level.register_forward_hook(level_hook, with_kwargs=True)

    def run(chunk, which, k0, mode, cond=None):
        if mode != "fit":
            folds = torch.tensor([b["fold"] for b in chunk], device=device)
            st["W"] = torch.stack(st["probes_w"])[1 - folds]
            st["Bv"] = torch.tensor(st["probes_b"], device=device)[1 - folds]
        st.update(on=True, mode=mode, cond=cond, call=0, zh_k=-1, logits=[])
        if mode == "rec_none":
            st["zh_ref"] = []
        try:
            with torch.no_grad():
                smps = [b[which] for b in chunk]
                if kind == "plain":
                    torch.manual_seed(seed + k0)                                # same pure-noise x_t for the noisy variant
                    plain.think(torch.stack([s.spatial_conditions for s in smps]).to(device))
                else:
                    batch = AmazeDataset.collate_fn(smps)
                    c_ = model._batch_to_sample(batch, device)
                    empty = batch.spatial_conditions.to(device)
                    noise = torch.randn(empty.shape, generator=torch.Generator(device=device).manual_seed(seed + k0), device=device)
                    tt = torch.full((len(smps),), 70, device=device, dtype=torch.long)
                    model.forward_with_carry(dataclasses.replace(c_, x_noisy=model.scheduler.add_noise(empty, noise, tt), timesteps=tt))
        finally:
            st["on"] = False

    # ---------------------------------------------------------------- probes (2-fold, unedited mazes)
    X, Y = [], []
    for k0 in range(0, len(boards), bs):
        chunk = boards[k0:k0 + bs]
        run(chunk, "none", k0, "fit")
        X.append(st["last_zl"].cpu()); Y.append(torch.tensor(np.stack([b["lab_none"] for b in chunk])))
    X, Y = torch.cat(X), torch.cat(Y)
    F_ = torch.tensor([b["fold"] for b in boards])
    st["probes_w"], st["probes_b"] = [], []
    for f in (0, 1):
        w_, b_ = train_linear(X[F_ == f].flatten(0, 1), Y[F_ == f].flatten(), int(ic.get("probe_steps", 3000)), device)
        st["probes_w"].append(w_.to(device)); st["probes_b"].append(float(b_))
    del X

    # ---------------------------------------------------------------- conditions
    res = {name: [] for name in CONDS}
    res_none = []
    for k0 in range(0, len(boards), bs):
        chunk = boards[k0:k0 + bs]
        run(chunk, "none", k0, "rec_none")
        res_none.append(torch.stack(st["logits"], 1).half())
        st["E"] = torch.tensor(np.stack([b["E"] for b in chunk]), device=device)
        for name, cd in CONDS.items():
            run(chunk, "side", k0, "cond", cd)
            res[name].append(torch.stack(st["logits"], 1).half())               # (B, n_zL_updates, T)
        logging.info(f"{k0 + len(chunk)}/{len(boards)} boards")
    res = {k: torch.cat(v) for k, v in res.items()}
    res_none = torch.cat(res_none)
    torch.save(dict(res=res, res_none=res_none, boards=[{k: b[k] for k in ("board", "fold", "E", "lab_none", "lab_side")} for b in boards],
                    kind=kind, maze_size=maze_size, side_pick=side_pick), out_dir / f"loops_zh_{kind}.pt")

    # ---------------------------------------------------------------- summary
    E = torch.tensor(np.stack([b["E"] for b in boards]))
    lab = torch.tensor(np.stack([b["lab_side"] for b in boards]))
    sp, known = lab == 1, lab >= 0
    U = res["base"].shape[1]

    def drop_update(L):                                                     # (U, n_tok) -> first update after the last > 0
        above = (L > 0)
        last = torch.where(above.any(0), U - 1 - above.flip(0).float().argmax(0), torch.full((L.shape[1],), -1))
        return last + 1

    print(f"\n[{kind}] n{maze_size}, side_pick={side_pick}, min_side={min_side}: {len(boards)} boards, {int(E.sum())} lollipop tokens; "
          f"z_L updates {U} (z_L update 6k is the first after the z_H update at call 7(k+1))")
    print(f"share of lollipop tokens read as path at z_L updates {SHOW} | at the end | median drop update of the tokens ever read as path "
          f"| shortest path covered at the end | boards exact at the end")
    Lnone = res_none.float()
    print(f"   {'unedited maze':13s} " + " ".join(f"{(Lnone[:, k][E] > 0).float().mean():4.0%}" for k in SHOW)
          + f" | {(Lnone[:, -1][E] > 0).float().mean():4.0%}")
    for name in CONDS:
        L = res[name].float()
        shares = " ".join(f"{(L[:, k][E] > 0).float().mean():4.0%}" for k in SHOW)
        Ltok = L.permute(1, 0, 2)[:, E]                                     # (U, n_tok)
        ever = (Ltok > 0).any(0)
        du = drop_update(Ltok)[ever]
        du = du[du < U]
        fin = L[:, -1] > 0
        exact = ((fin == sp) | ~known).all(1).float().mean()
        print(f"   {name:13s} {shares} | {(fin[E]).float().mean():4.0%} | {du.float().median() if len(du) else float('nan'):5.1f} "
              f"| {(fin[sp]).float().mean():4.0%} | {exact:4.0%}")

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot([(Lnone[:, k][E] > 0).float().mean() for k in range(60)], "k--", label="unedited maze")
    for name in CONDS:
        L = res[name].float()
        ax.plot([(L[:, k][E] > 0).float().mean() for k in range(60)], label=name)
    for x in range(6, 60, 6):
        ax.axvline(x - 0.5, color="0.85", lw=0.8)
    ax.set(xlabel="z_L update (grey lines: z_H updates)", ylabel="lollipop tokens read as path",
           title=f"{kind}, n{maze_size} {side_pick} side loops: z_H interventions")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out_dir / f"loops_zh_{kind}.png", dpi=120)
    print(f"saved {out_dir}/loops_zh_{kind}.pt / .png")


if __name__ == "__main__":
    main()
