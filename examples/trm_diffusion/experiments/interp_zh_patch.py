"""TEMPORARY diagnostic (cluster only): does the first z_H update trigger the thinker's self-corrections in step 1?

Clean boards, same boards / noise as before; one conditional thinker pass at t=70. Reasoning step 1 = 21 L_level
calls (3 H-cycles x (6 z_L updates + 1 z_H update)); a forward hook records every call's output and patches one:
  normal
  skip_zh1   first z_H update (call 6) skipped: its output replaced by its input (z_H before the update)
  other_zh1  first z_H update replaced by the next board's (board i gets board i-1's, same batch)
  skip_zh2   second z_H update (call 13) skipped
  skip_zl7   control: the first z_L update of H-cycle 2 (call 7) skipped
Per z_L update u (18), path-vs-off probes trained on the normal run (cross-fitted: even <-> odd boards) give every
token's path probability in every run. Counted per run: strict drops (truly off, p>0.8 then p<0.2 later, off at the
end of step 1), when they happen, overlap with the normal run's drops, accuracy after each H-cycle and at the end of
the pass (final z_H probe); for other_zh1, how far the plan follows the donor board.
"""
import dataclasses

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import interp_hv_swap as hs
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

MODES = ("normal", "skip_zh1", "other_zh1", "skip_zh2", "skip_zl7")
HI, LO = 0.8, 0.2


def fit_probe(X, y, train):
    m, s = X[train].mean(0), X[train].std(0) + 1e-6
    Xa = (X[train] - m) / s
    w = torch.zeros(X.shape[1] + 1, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=300)

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(Xa @ w[:-1] + w[-1], y[train]) + 1e-3 * w[:-1].pow(2).sum()
        l.backward()
        return l
    opt.step(cl)
    w = w.detach()
    return lambda Z: ((Z - m) / s) @ w[:-1] + w[-1]


def crossfit(X_train_mode, Xs, y, A):
    """Probes fitted on the normal run's states; returns {mode: p} with p for odd boards from the even-board probe
    and vice versa."""
    pA, pB = fit_probe(X_train_mode, y, A), fit_probe(X_train_mode, y, ~A)
    return {k: torch.where(A, torch.sigmoid(pB(X)), torch.sigmoid(pA(X))) for k, X in Xs.items()}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model))
    model.eval()
    model.scheduler.set_timesteps(int(ic.get("num_inference_steps", 20)), device=device)
    tc = model.thinker.inner.config
    per, n_calls = tc.L_cycles + 1, tc.H_cycles * (tc.L_cycles + 1)
    zl_calls = [c for c in range(n_calls) if c % per != per - 1]
    print(f"step 1 = {n_calls} calls; z_H updates at {[c for c in range(n_calls) if c % per == per - 1]}", flush=True)
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    rec = {m: {"zl": [], "zh_final": []} for m in MODES}
    gts, donor_gt = [], []
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B = b["B"]
        gt = torch.from_numpy(b["cats"] != "off")
        gts.append(gt); donor_gt.append(gt.roll(1, 0))
        s = dataclasses.replace(b["conditions"], x_noisy=b["x_t"], timesteps=torch.full((B,), 70, device=device, dtype=torch.long))
        for mode in MODES:
            state = {"n": 0, "out": []}

            def hook(module, inputs, output, mode=mode, state=state):
                c = state["n"]; state["n"] += 1
                if c >= n_calls:
                    return None
                new = output
                if (mode == "skip_zh1" and c == per - 1) or (mode == "skip_zh2" and c == 2 * per - 1) or (mode == "skip_zl7" and c == per):
                    new = inputs[0]
                elif mode == "other_zh1" and c == per - 1:
                    new = output.roll(1, 0)
                state["out"].append(new.detach().half().cpu())
                return None if new is output else new
            h = model.thinker.inner.L_level.register_forward_hook(hook)
            with torch.no_grad():
                _, z_H, _ = model.forward_with_carry(s)
            h.remove()
            rec[mode]["zl"].append(torch.stack([state["out"][c] for c in zl_calls], 1))     # (B, 18, T, D)
            rec[mode]["zh_final"].append(z_H.half().cpu())
        print(f"batch start={start}: {B} boards", flush=True)

    gt = torch.cat(gts); dgt = torch.cat(donor_gt)
    N, T = gt.shape
    y = gt.reshape(-1).float()
    A = torch.from_numpy(np.repeat(np.arange(N) % 2 == 0, T))
    Z = {m: torch.cat(rec[m]["zl"]).float() for m in MODES}                               # (N, 18, T, D)
    U = Z["normal"].shape[1]
    P = {m: torch.zeros(N, T, U) for m in MODES}
    for u in range(U):
        p = crossfit(Z["normal"][:, u].reshape(N * T, -1), {m: Z[m][:, u].reshape(N * T, -1) for m in MODES}, y, A)
        for m in MODES:
            P[m][..., u] = p[m].reshape(N, T)
    ZH = {m: torch.cat(rec[m]["zh_final"]).float().reshape(N * T, -1) for m in MODES}
    pf = crossfit(ZH["normal"], ZH, y, A)

    def drops(Pm):
        Pm = Pm.numpy(); g = gt.numpy()
        out = np.zeros((N, T), bool); when = np.full((N, T), -1)
        for i in range(N):
            for t in range(T):
                if g[i, t] or Pm[i, t, -1] >= 0.5:
                    continue
                hi = np.nonzero(Pm[i, t] > HI)[0]
                if len(hi):
                    later = [u for u in np.nonzero(Pm[i, t] < LO)[0] if u > hi[0]]
                    if later:
                        out[i, t] = True; when[i, t] = later[0]
        return out, when
    base, _ = drops(P["normal"])
    print(f"\n{N} boards, {int((~gt).sum())} off tokens; z_L updates 1..{U} (H-cycle 1 = 1..{per - 1}, 2 = {per}..{2 * (per - 1)}, ...)")
    for m in MODES:
        dm, wm = drops(P[m])
        acc = lambda u: float(((P[m][..., u] > 0.5) == gt).float().mean())
        hist = np.bincount(wm[dm], minlength=U)
        print(f"  {m:9s}: strict drops {int(dm.sum()):5d} (normal's drops also dropped here {(dm & base).sum() / max(base.sum(), 1):.0%}, "
              f"new {int((dm & ~base).sum())}) | acc after H-cycle 1/2/3 {acc(per - 2):.3f} {acc(2 * per - 3):.3f} {acc(U - 1):.3f} | "
              f"end of pass {float(((pf[m] > 0.5) == y.bool()).float().mean()):.3f}")
        print(f"             drops at z_L update 1..{U}: " + " ".join(str(x) for x in hist))
    # other_zh1: does the plan follow the donor board where the two boards disagree?
    dis = gt != dgt
    for m in ("normal", "other_zh1"):
        traj = [float(((P[m][..., u] > 0.5) == dgt)[dis].float().mean()) for u in range(U)]
        fin = float(((pf[m].reshape(N, T) > 0.5) == dgt)[dis].float().mean())
        print(f"  {m:9s}: where own and donor GT differ ({int(dis.sum())} tokens), share of decisions = donor's, per z_L update: "
              + " ".join(f"{v:.2f}" for v in traj) + f" | end of pass {fin:.2f}")
    torch.save({"P": P, "p_final": {m: v.reshape(N, T) for m, v in pf.items()}, "gt": gt, "donor_gt": dgt}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
