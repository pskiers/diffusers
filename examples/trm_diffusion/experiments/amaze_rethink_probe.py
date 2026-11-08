"""
ABOUTME: Rebuttal exp 08 on Maze / Queens: a linear probe from the PaTh thinker's answer state z_H (token at each cell
ABOUTME: centre) to per-cell labels, then probe accuracy and within-step rethinks (Fig. 9 / Tab. 20 analogues).

Mirrors the Sudoku App. K probe (experiments/train_digit_probe.py) and the Fig. 9 rethink analysis
(experiments/z_scratchpad_probe.py, whose L_level hook and (k, h, l) structuring are reused):
1. Train trajectories (real CFG sampling, the model's own sampler settings): z_H after every reasoning iteration of
   every denoising step (conditional branch), read at the token holding each cell's centre; label = that cell in the
   trajectory's OWN final image, read by the AMAZE scorer (experiments/amaze_boards.py). A random --row-frac of the
   (step, iteration, cell) rows is kept and a logistic probe is trained on them (10% of train boards held out).
2. Test trajectories (disjoint boards): at every step every (k, h, l) position is decoded through the probe:
   y = z_H after each H-cycle, and each fast state z_L through one hypothetical slow update (z_scratchpad_probe's
   recipe). Reported: probe accuracy by step and iteration vs the final-image labels, with the control floor (the
   probe on the initial state, the same for every board); violations of the decoded grid (amaze_boards.violations,
   no ground truth); dip ("rethink") per group = the group's worst z more violated than the y it started from, rate
   and magnitude given a dip, by % of sampling (Fig. 9); W3a: the same restricted to steps where the probe is >= 95%
   accurate, and confident-cell dips: z keeps its decoded value only on cells where the probe is confident
   (|logit| > --conf-margin) at both the start y and that z, every other cell takes the start y's value, so a dip
   can only come from confident changes (not from cells near the probe's decision boundary).

Data: Maze test parquets of square / hexagon / triangle (circle excluded: its cell map does not align with the drawn
walls, so passages cannot be read; see amaze_boards.py), the first --test-per-combo boards of each (geometry, size)
for testing and the next --train-per-combo for training. Queens: the 50 boards per size split the same way, each
train board sampled --samples-per-train-board times. The train parquets carry no cell maps.

Usage (from examples/trm_diffusion):
    python experiments/amaze_rethink_probe.py --task maze --train-per-combo 43 --test-per-combo 11
    python experiments/amaze_rethink_probe.py --task queens --train-per-combo 36 --test-per-combo 14 \
        --samples-per-train-board 4
    # smoke: --sizes 8 --train-per-combo 2 --test-per-combo 1 --probe-updates 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from amaze_boards import build_board, labels_from_image, violations  # noqa: E402
from compute_table import ENTRIES, build  # noqa: E402
from trajectory_violations import CHECKPOINTS  # noqa: E402
from z_scratchpad_probe import _LLevelCapture, _structure_calls  # noqa: E402

MAZE_GEOS = ("square", "hexagon", "triangle")


def load_boards(task: str, data_root: Path, sizes: list[int], scorer, start: int, count: int) -> list[dict]:
    from datasets.amaze_dataset import AmazeDataset

    paths = ([(f"{g}_{s}", s, data_root / "maze" / g / f"n{s}_test.parquet") for g in MAZE_GEOS for s in sizes]
             if task == "maze" else [(str(s), s, data_root / "queens" / f"n{s}_test.parquet") for s in sizes])
    out = []
    for name, size, p in paths:
        ds = AmazeDataset(str(p), split="test", image_size=144, condition_field="m_original_img",
                          target_field="sol_img", num_channels=3, include_metadata=True)
        for i in range(start, min(start + count, len(ds))):
            item = ds[i]
            out.append({"combo": name, "size": size, "item": item, "board": build_board(item.metadata, task, scorer)})
    return out


@torch.no_grad()
def sample(model, conditions, x_init, capture: str):
    """The model's own CFG sampler (sampling_pipeline settings), conditional branch instrumented.
    capture='zH': per step the list of z_H after each reasoning iteration; 'calls': per step the structured
    (k, h) groups of every L_level call. Returns (final x, per-step captures)."""
    pipe = model.sampling_pipeline
    scale = pipe.cfg_scale
    model.scheduler.set_timesteps(pipe.num_inference_steps, device=x_init.device)
    inner = model.thinker.inner
    x, caps = x_init.clone(), []
    for t in model.scheduler.timesteps:
        s = dataclasses.replace(conditions, x_noisy=x, timesteps=t.expand(x.shape[0]).to(x.device))
        if capture == "zH":
            zs: list = []
            pred_c, _, _ = model.forward_with_carry(s, zH_out=zs)
            caps.append(zs)
        else:
            with _LLevelCapture(inner) as cap:
                pred_c, _, _ = model.forward_with_carry(s)
            caps.append(_structure_calls(cap.calls, model.n_sup, inner.config.H_cycles, inner.config.L_cycles))
        noise_pred = pred_c.pred
        if scale != 1.0:
            pred_u, _, _ = model.forward_with_carry(model.null_condition_sample(s))
            noise_pred = pred_u.pred + scale * (noise_pred - pred_u.pred)
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    return x, caps


def hypothetical_state(inner, z_H_before: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
    """z_scratchpad_probe._hypothetical_decode without the lm_head: one slow update from this fast state."""
    cos_sin = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None
    return inner.L_level(z_H_before, z_L, cos_sin=cos_sin)


def batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", choices=["maze", "queens"], required=True)
    p.add_argument("--sizes", default=None, help="comma list (default: maze 3,5,7,8,9,11,13,16; queens 4-10)")
    p.add_argument("--train-per-combo", type=int, default=43)
    p.add_argument("--test-per-combo", type=int, default=11)
    p.add_argument("--samples-per-train-board", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--row-frac", type=float, default=0.1, help="fraction of train (step, iter, cell) rows kept")
    p.add_argument("--probe-updates", type=int, default=4000, help="Adam updates of the probe (minibatch 8192)")
    p.add_argument("--probe-acc-gate", type=float, default=0.95)
    p.add_argument("--conf-margin", type=float, default=2.0, help="|probe logit| for a confident cell (2: p > 0.88)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="data/amaze")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else (
        [3, 5, 7, 8, 9, 11, 13, 16] if args.task == "maze" else list(range(4, 11)))
    out_dir = Path(args.out_dir or HERE / "results" / "rethink_probe" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from eval.amaze_eval import AmazeMetrics
    from eval.checkpoint_utils import load_checkpoint

    key = f"{args.task}/path"
    cfg, model, _ = build(ENTRIES[key], [], device)
    step = load_checkpoint(model, CHECKPOINTS[key], use_ema=True, device="cpu")
    model = model.to(device).eval()
    inner = model.thinker.inner
    H, L, n_sup = inner.config.H_cycles, inner.config.L_cycles, model.n_sup
    n_steps = model.sampling_pipeline.num_inference_steps
    scorer = AmazeMetrics(device=device, task=args.task)
    print(f"{key}: {ENTRIES[key]['run']} (step {step}); sampler {n_steps} steps, CFG {model.sampling_pipeline.cfg_scale};"
          f" n_sup {n_sup}, H {H}, L {L}", flush=True)

    test = load_boards(args.task, Path(args.data_root), sizes, scorer, 0, args.test_per_combo)
    train = load_boards(args.task, Path(args.data_root), sizes, scorer, args.test_per_combo, args.train_per_combo)
    train = [b for b in train for _ in range(args.samples_per_train_board)]
    print(f"boards: {len(train)} train trajectories, {len(test)} test boards; first board {test[0]['combo']}: "
          f"{test[0]['board'].n_cells} cells, tokens {test[0]['board'].token_idx[:6].tolist()}", flush=True)

    def conditions_of(chunk):
        from datasets.amaze_dataset import AmazeDataset
        items = [c["item"] for c in chunk]
        return model._batch_to_sample(AmazeDataset.collate_fn(items), device)

    def x_init_of(n):
        return torch.randn(n, *model.noise_shape, device=device) * getattr(model.scheduler, "init_noise_sigma", 1.0)

    # sanity: the instrumented loop is the model's own sampler (same RNG stream -> same image)
    chunk = test[:2]
    cond, x0 = conditions_of(chunk), x_init_of(len(chunk))
    torch.manual_seed(123)
    ref = model.sampling_pipeline.sample_one_batch(model, cond, device, initial_noise=x0)
    torch.manual_seed(123)
    mine, _ = sample(model, cond, x0, "zH")
    print(f"  sampler check: max |instrumented - pipeline| = {(ref - mine).abs().max().item():.3g}", flush=True)
    assert torch.allclose(ref, mine, atol=1e-5), "instrumented sampler differs from the pipeline"

    # ── 1. training rows ──────────────────────────────────────────────────────
    rows_x, rows_y, rows_board = [], [], []
    for bi, chunk in enumerate(batches(train, args.batch_size)):
        final, caps = sample(model, conditions_of(chunk), x_init_of(len(chunk)), "zH")
        imgs = model.decode_for_eval(final).cpu()
        for j, c in enumerate(chunk):
            b = c["board"]
            lab = labels_from_image(imgs[j], b, scorer)
            feats = torch.stack([z[j, b.token_idx.to(device)] for zs in caps for z in zs])  # (steps*n_sup, N, D)
            feats = feats.reshape(-1, feats.shape[-1])
            keep = torch.from_numpy(rng.random(feats.shape[0]) < args.row_frac)
            rows_x.append(feats[keep.to(device)].half().cpu())
            rows_y.append(lab.repeat(n_steps * n_sup)[keep])
            rows_board.append(torch.full((int(keep.sum()),), bi * args.batch_size + j))
        if bi == 0:
            print(f"  first train batch: {len(chunk)} trajectories, {len(caps)} steps x {len(caps[0])} iters, "
                  f"z_H {tuple(caps[0][0].shape)}, label positive rate {rows_y[0].mean().item():.3f}", flush=True)
    X, Y, BID = torch.cat(rows_x), torch.cat(rows_y), torch.cat(rows_board)
    n_train_traj = len(train)
    val_mask = BID >= int(0.9 * n_train_traj)
    print(f"  {X.shape[0]} probe rows ({val_mask.sum().item()} held out), positive rate {Y.mean():.3f}", flush=True)

    # linear probe on train-standardized features (standardization is itself linear)
    Xtr, Ytr = X[~val_mask], Y[~val_mask]
    mu = Xtr.float().mean(0).to(device)
    sd = Xtr.float().std(0).clamp(min=1e-6).to(device)
    probe = torch.nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for it in range(args.probe_updates):
        idx = torch.randint(0, Xtr.shape[0], (8192,))
        logit = probe((Xtr[idx].to(device).float() - mu) / sd).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, Ytr[idx].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it in (0, args.probe_updates - 1):
            print(f"  probe update {it}: loss {loss.item():.4f}", flush=True)

    @torch.no_grad()
    def probe_logit(feats: torch.Tensor) -> torch.Tensor:
        return probe((feats.float() - mu) / sd).squeeze(-1)

    def predict(feats: torch.Tensor) -> torch.Tensor:
        return probe_logit(feats) > 0

    def acc_on(mask):
        xs, ys = X[mask], Y[mask]
        pr = torch.cat([predict(xs[i:i + 65536].to(device)).cpu() for i in range(0, len(xs), 65536)])
        acc = (pr == ys.bool()).float().mean().item()
        bal = 0.5 * ((pr & ys.bool()).sum() / ys.sum().clamp(min=1) + (~pr & ~ys.bool()).sum() / (1 - ys).sum().clamp(min=1))
        return acc, float(bal)

    probe_train, probe_val = acc_on(~val_mask), acc_on(val_mask)
    print(f"  probe: train acc/bal-acc {probe_train}, held-out {probe_val}", flush=True)

    # ── 2. test trajectories ─────────────────────────────────────────────────
    z_init = model.get_initial_states(1)[0].to(device)  # (1, seq, D): the same for every board
    stats = {"acc": np.zeros((n_steps, n_sup)), "acc_n": np.zeros((n_steps, n_sup)), "floor": [0, 0]}
    per_board = []  # per test board: violations / dips / start-y predictions per step
    for chunk in batches(test, args.batch_size):
        final, caps = sample(model, conditions_of(chunk), x_init_of(len(chunk)), "calls")
        imgs = model.decode_for_eval(final).cpu()
        labs = [labels_from_image(imgs[j], c["board"], scorer) for j, c in enumerate(chunk)]
        rec = [{"combo": c["combo"], "size": c["size"], "steps": [],
                "label_viol": float(violations(labs[j][None], c["board"]).item())} for j, c in enumerate(chunk)]
        for s, groups in enumerate(caps):
            # per position (B, seq, D): y0, then per (k, h): L hypothetical z states and the y after it
            y0 = groups[0][0]["z_H_before"]
            z_pos, y_pos = [], []
            for k in range(n_sup):
                for h in range(H):
                    g = groups[k][h]
                    z_pos += [hypothetical_state(inner, g["z_H_before"], g["z_L"][l]) for l in range(L)]
                    y_pos.append(g["z_H_after"])
            for j, c in enumerate(chunk):
                b, lab = c["board"], labs[j].to(device)
                tok = b.token_idx.to(device)
                ly = probe_logit(torch.stack([y0[j, tok]] + [y[j, tok] for y in y_pos]))  # (1 + n_sup*H, N)
                lz = probe_logit(torch.stack([z[j, tok] for z in z_pos]))                 # (n_sup*H*L, N)
                ys, zs = ly > 0, lz > 0
                vy, vz = violations(ys, b), violations(zs, b).reshape(n_sup * H, L)
                prev = vy[:-1]                                                           # start y of each group
                dip = vz.max(1).values > prev
                mag = (vz.max(1).values - prev).clamp(min=0)
                # confident-cell dips: z's value only where the probe is confident at both start y and z
                G, N = n_sup * H, b.n_cells
                start_pred = ys[:-1][:, None, :].expand(G, L, N)
                start_conf = (ly[:-1].abs() > args.conf_margin)[:, None, :].expand(G, L, N)
                z_conf = lz.reshape(G, L, N).abs() > args.conf_margin
                z_mixed = torch.where(start_conf & z_conf, zs.reshape(G, L, N), start_pred)
                vzc = violations(z_mixed.reshape(G * L, N), b).reshape(G, L)
                dip_c = vzc.max(1).values > prev
                mag_c = (vzc.max(1).values - prev).clamp(min=0)
                iter_end = ys[1:][H - 1::H]                                              # y at the end of each iteration
                stats["acc"][s] += (iter_end == lab.bool()).float().mean(1).cpu().numpy()
                stats["acc_n"][s] += 1
                rec[j]["steps"].append({"viol_y": vy.cpu().numpy(), "viol_z": vz.cpu().numpy(),
                                        "dip": dip.cpu().numpy(), "mag": mag.cpu().numpy(),
                                        "dip_conf": dip_c.cpu().numpy(), "mag_conf": mag_c.cpu().numpy(),
                                        "conf_frac": float((ly.abs() > args.conf_margin).float().mean()),
                                        "acc_end": float((ys[-1] == lab.bool()).float().mean())})
                if s == 0:
                    fl = predict(z_init[0, tok])
                    stats["floor"][0] += float((fl == lab.bool()).float().mean())
                    stats["floor"][1] += 1
        per_board += [(r, c["board"]) for r, c in zip(rec, chunk)]
        print(f"  test {len(per_board)}/{len(test)} boards", flush=True)

    acc_step_iter = stats["acc"] / np.maximum(stats["acc_n"], 1)                 # (steps, n_sup)
    acc_step = acc_step_iter[:, -1]
    err_step = 1 - acc_step
    gated = acc_step >= args.probe_acc_gate

    def dip_table(filter_fn):
        out = []
        for s in range(n_steps):
            d = [x for r, _ in per_board if filter_fn(r) for x in [r["steps"][s]]]
            dips = np.concatenate([x["dip"] for x in d])
            mags = np.concatenate([x["mag"] for x in d])
            dc = np.concatenate([x["dip_conf"] for x in d])
            mc = np.concatenate([x["mag_conf"] for x in d])
            vy = np.mean([x["viol_y"][1:].mean() for x in d])
            out.append({"frac": (s + 1) / n_steps, "dip_rate": float(dips.mean()),
                        "mag_given_dip": float(mags[dips].mean()) if dips.any() else float("nan"),
                        "dip_rate_conf": float(dc.mean()),
                        "mag_given_dip_conf": float(mc[dc].mean()) if dc.any() else float("nan"),
                        "conf_frac": float(np.mean([x["conf_frac"] for x in d])),
                        "viol_y": float(vy), "probe_acc": float(acc_step[s]), "gated": bool(gated[s])})
        return out

    overall = dip_table(lambda r: True)
    by_size = {str(sz): dip_table(lambda r, sz=sz: r["size"] == sz) for sz in sizes}
    gated_steps = [o for o in overall if o["gated"]]
    summary = {
        "dip_rate_all_steps": float(np.mean([o["dip_rate"] for o in overall])),
        "dip_rate_gated_steps": float(np.mean([o["dip_rate"] for o in gated_steps])) if gated_steps else None,
        "dip_rate_conf_all_steps": float(np.mean([o["dip_rate_conf"] for o in overall])),
        "dip_rate_conf_gated_steps": float(np.mean([o["dip_rate_conf"] for o in gated_steps])) if gated_steps else None,
        "n_gated_steps": int(gated.sum()),
        "probe_floor_acc": stats["floor"][0] / max(1, stats["floor"][1]),
        "final_image_label_viol_mean": float(np.mean([r["label_viol"] for r, _ in per_board])),
    }
    res = {"task": args.task, "run": ENTRIES[key]["run"], "checkpoint": CHECKPOINTS[key], "step": step,
           "args": vars(args), "n_train_trajectories": n_train_traj, "n_test_boards": len(per_board),
           "probe_rows": int(X.shape[0]), "probe_train_acc_balacc": probe_train, "probe_heldout_acc_balacc": probe_val,
           "probe_acc_by_step_iter": acc_step_iter.tolist(),
           "overall": overall, "by_size": by_size, "summary": summary, "wall_clock_s": round(time.time() - t0, 1),
           "command": " ".join(sys.argv), "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(device),
           "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 cwd=HERE).stdout.strip()}
    (out_dir / f"{args.task}.json").write_text(json.dumps(res, indent=2))
    its = sorted({0, n_sup // 4 - 1, n_sup // 2 - 1, n_sup - 1})
    lines = [f"# Exp 08 rethinks, {args.task}: {ENTRIES[key]['run']}", "",
             f"{n_train_traj} train trajectories, {len(per_board)} test boards; probe held-out acc / balanced acc "
             f"{probe_val[0]:.3f} / {probe_val[1]:.3f}; control floor (initial state) {summary['probe_floor_acc']:.3f};"
             f" violations of the final-image labels themselves {summary['final_image_label_viol_mean']:.2f}.",
             f"Dip rate {summary['dip_rate_all_steps']:.3f} (all steps), {summary['dip_rate_gated_steps']} on the "
             f"{summary['n_gated_steps']} steps with probe acc >= {args.probe_acc_gate}; confident cells only "
             f"(|logit| > {args.conf_margin}): {summary['dip_rate_conf_all_steps']:.3f} "
             f"(gated {summary['dip_rate_conf_gated_steps']}).", "",
             "| % sampling " + " ".join(f"| probe acc it {i + 1}" for i in its) + " | viol (y) | dip rate | "
             "mag given dip | dip rate conf | mag given dip conf | conf cells |", "|---" * (7 + len(its)) + "|"]
    for s, o in enumerate(overall):
        lines.append(f"| {100 * o['frac']:.0f} | " + " | ".join(f"{acc_step_iter[s, i]:.3f}" for i in its)
                     + f" | {o['viol_y']:.2f} | {o['dip_rate']:.3f} | {o['mag_given_dip']:.2f} | {o['dip_rate_conf']:.3f}"
                     f" | {o['mag_given_dip_conf']:.2f} | {o['conf_frac']:.3f} |")
    lines += ["", "Dip rate by size (mean over steps): " + ", ".join(
        f"{sz}: {np.mean([o['dip_rate'] for o in t]):.3f}" for sz, t in by_size.items())]
    (out_dir / f"{args.task}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved -> {out_dir / args.task}.json ({res['wall_clock_s']} s)")


if __name__ == "__main__":
    main()
