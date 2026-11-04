"""TEMPORARY diagnostic (cluster only): how the thinker's state evolves over ALL its block calls in one pass.

Clean boards, same boards / noise as before; one conditional thinker pass at each t in AT_T. A forward hook on L_level
records every call (n_sup reasoning steps x H_cycles x (L_cycles z_L updates + 1 z_H update) = 336 calls).
Per call, saved (small):
  rel_change   ‖z_c − z_prev‖ / ‖z_c‖ per board, where z_prev is the previous state of the SAME kind (z_L or z_H)
  tok_change   the same per token (board-mean kept per call and token)
  p            path probability per token, from a probe trained at that call (cross-fitted: even <-> odd boards)
  acc          held-out accuracy of that probe's decisions vs the GT path
plus GT labels and the maze / solution images for plotting.
"""
import dataclasses

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import interp_hv_swap as hs
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

AT_T = (70, 5)


def fit_probe(X, y, train, device):
    X, y = X.to(device), y.to(device)
    m, s = X[train].mean(0), X[train].std(0) + 1e-6
    Xa = (X[train] - m) / s
    w = torch.zeros(X.shape[1] + 1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(Xa @ w[:-1] + w[-1], y[train]) + 1e-3 * w[:-1].pow(2).sum()
        l.backward()
        return l
    opt.step(cl)
    w = w.detach()
    return lambda Z: (((Z.to(device) - m) / s) @ w[:-1] + w[-1]).cpu()


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
    per = tc.L_cycles + 1
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    states = {t: [] for t in AT_T}
    gts, sols, mazes, cleans = [], [], [], []
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B = b["B"]
        gts.append(torch.from_numpy(b["cats"] != "off")); sols.append(b["sol_u8"]); mazes.append(b["maze_u8"]); cleans.append(b["clean_u8"])
        # the normal CFG trajectory from t=70; trace the conditional thinker pass at each t in AT_T
        run_ts = model.scheduler.timesteps[model.scheduler.timesteps <= 70]
        predictor = model.sampling_pipeline.predictor
        gen = torch.Generator(device=device).manual_seed(b["base_seed"] + 1)
        xx = b["x_t"]
        with torch.no_grad():
            for tt in run_ts:
                s = dataclasses.replace(b["conditions"], x_noisy=xx, timesteps=tt.expand(B).to(device))
                out, h = [], None
                if int(tt) in AT_T:
                    h = model.thinker.inner.L_level.register_forward_hook(lambda m, i, o, out=out: out.append(o.detach().half().cpu()))
                pred_cond, _, _ = model.forward_with_carry(s)
                if h is not None:
                    h.remove()
                    states[int(tt)].append(torch.stack(out, 1))            # (B, n_calls, T, D)
                pred_uncond = model(model.null_condition_sample(s)).pred
                pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                xx = model.scheduler.step(pred, tt, xx, generator=gen).prev_sample
        print(f"batch start={start}: {B} boards", flush=True)

    gt = torch.cat(gts); N, T = gt.shape
    y = gt.reshape(-1).float()
    A = torch.from_numpy(np.repeat(np.arange(N) % 2 == 0, T))
    res = {"gt": gt, "sol": torch.cat(sols), "maze": torch.cat(mazes), "clean": torch.cat(cleans), "per": per,
           "n_sup": None, "kinds": None}
    for t in AT_T:
        S = torch.cat(states[t])                                           # (N, C, T, D) fp16
        C = S.shape[1]
        kinds = np.array(["z_H" if c % per == per - 1 else "z_L" for c in range(C)])
        res["kinds"], res["n_sup"] = kinds, C // (tc.H_cycles * per)
        rel = torch.zeros(N, C); tok = torch.zeros(C, T); P = torch.zeros(N, C, T, dtype=torch.float16); acc = torch.zeros(C)
        last = {}
        for c in range(C):
            z = S[:, c].float()
            k = kinds[c]
            if k in last:
                dz = z - last[k]
                rel[:, c] = dz.flatten(1).norm(dim=1) / z.flatten(1).norm(dim=1)
                tok[c] = (dz.norm(dim=-1) / z.norm(dim=-1)).mean(0)
            else:
                rel[:, c] = float("nan"); tok[c] = float("nan")
            last[k] = z
            X = z.reshape(N * T, -1)
            pA, pB = fit_probe(X, y, A, device), fit_probe(X, y, ~A, device)
            logit = torch.where(A, pB(X), pA(X))
            P[:, c] = torch.sigmoid(logit).reshape(N, T).half()
            acc[c] = ((logit > 0) == y.bool()).float().mean()
            if c % 21 == 20:
                print(f"t={t} call {c}: acc {acc[c]:.3f}  median rel change {rel[:, c].nanmedian():.4f}", flush=True)
        res[t] = {"rel_change": rel, "tok_change": tok, "p": P, "acc": acc}
    torch.save(res, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
