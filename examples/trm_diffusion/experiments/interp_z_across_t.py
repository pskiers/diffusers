"""TEMPORARY diagnostic (Claude, amaze interpretability): how does the thinker's read-out state change across denoising steps?

Standard generation from pure noise (20 steps, CFG), all 100 square 8x8 test boards, two noise seeds. Recorded at every
denoising step t: the final z_H of the conditional pass (call 336, the state lm_head reads) and the logits.
Per board, the board-specific part r_t = z_t - mu_t (mu_t = mean over boards at that t, per token position).
  (a) similarity of the same board's state between steps: mean cosine of r_t and r_t' (per token, averaged), 20 x 20;
      the same for the raw state; relative change between consecutive steps.
  (b) baseline: the same board and step with the other noise seed (how much of a difference is "just noise").
  (c) the plan at each step: linear probe path vs off per token (fit on even boards at that step, scored on odd);
      probes transferred between steps (fit at t, applied at t'): does the code for "path" stay the same?
  (d) does the plan decoded at step t already equal the plan at the last step, and what is finally drawn?
  (e) cosine between the path - off mean-difference directions of different steps.
Saves the metrics and the states (fp16) for offline analysis.
"""
import dataclasses

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import interp_size_ablation as sa
import plan_analysis as pa
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

G, IMG, T, BATCH = 12, 144, 144, 50
SEEDS = (1000, 2000)


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
    ts = model.scheduler.timesteps; tl = [int(t) for t in ts]; nT = len(tl)
    predictor = model.sampling_pipeline.predictor
    inner = model.thinker.inner
    D = inner.config.hidden_size
    W = inner.lm_head.weight.detach().float()
    st = {"on": False, "c": 0, "buf": None, "j": 0}

    def hook(module, args, output):
        if st["on"]:
            if st["c"] == 335:
                st["buf"][:, st["j"]] = output.detach().to("cpu", torch.float16)
            st["c"] += 1
    inner.L_level.register_forward_hook(hook)

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=IMG, num_channels=3, include_metadata=False,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"))
    N = min(100, len(ds))
    Z = {s: torch.empty((N, nT, T, D), dtype=torch.float16) for s in SEEDS}
    drawn = {s: np.zeros((N, T), bool) for s in SEEDS}
    gt = np.zeros((N, T), bool)
    for i0 in range(0, N, BATCH):
        batch = AmazeDataset.collate_fn([ds[i] for i in range(i0, min(i0 + BATCH, N))])
        cond = model._batch_to_sample(batch, device); B = batch.images.shape[0]
        sol = (batch.images.float().clamp(0, 1) * 255).byte().cpu(); maze = (batch.spatial_conditions.float().clamp(0, 1) * 255).byte().cpu()
        gt[i0:i0 + B] = np.stack([pa.token_labels(sol[i], maze[i])[0] != "off" for i in range(B)])
        for seed in SEEDS:
            x = torch.randn((B, 3, IMG, IMG), generator=torch.Generator(device=device).manual_seed(seed + i0), device=device)
            gen = torch.Generator(device=device).manual_seed(seed + 7 + i0)
            buf = torch.empty((B, nT, T, D), dtype=torch.float16)
            with torch.no_grad():
                for j, t in enumerate(ts):
                    s_ = dataclasses.replace(cond, x_noisy=x, timesteps=t.expand(B).to(device))
                    st.update(on=True, c=0, buf=buf, j=j)
                    pred_cond, _, _ = model.forward_with_carry(s_)
                    st["on"] = False
                    pred_uncond = model(model.null_condition_sample(s_)).pred
                    pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
                    x = model.scheduler.step(pred, t, x, generator=gen).prev_sample
                out = model.decode_for_eval(x).clamp(0, 1)
            Z[seed][i0:i0 + B] = buf
            o8 = (out * 255).byte().cpu()
            drawn[seed][i0:i0 + B] = np.stack([pa.token_labels(o8[i], o8[i])[0] != "off" for i in range(B)])
            print(f"  boards {i0}..{i0 + B}, seed {seed}: recorded {nT} steps", flush=True)

    s0, s1 = SEEDS
    gt_t = torch.from_numpy(gt)
    even = torch.from_numpy(np.arange(N) % 2 == 0); odd = ~even
    res = {"tl": tl, "gt": gt, "drawn": drawn}

    def cosmat(A, Bm):
        """A, Bm: (N, nT, T, D) on CPU fp16 -> (nT, nT) mean over boards and tokens of cos(A[:, i], Bm[:, j])."""
        out = torch.zeros(nT, nT)
        An = [torch.nn.functional.normalize(A[:, i].float().to(device), dim=-1) for i in range(nT)]
        Bn = An if Bm is A else [torch.nn.functional.normalize(Bm[:, i].float().to(device), dim=-1) for i in range(nT)]
        for i in range(nT):
            for j in range(nT):
                out[i, j] = float((An[i] * Bn[j]).sum(-1).mean())
        return out
    with torch.no_grad():
        mu = {s: Z[s].float().mean(0) for s in SEEDS}                                   # (nT, T, D)
        R = {s: (Z[s].float() - mu[s]).half() for s in SEEDS}
        res["cos_raw"] = cosmat(Z[s0], Z[s0]); res["cos_spec"] = cosmat(R[s0], R[s0])
        res["cos_seed_spec"] = cosmat(R[s0], R[s1])                                      # diagonal = same step, other seed
        res["cos_seed_raw"] = cosmat(Z[s0], Z[s1])
        rel = []
        for j in range(1, nT):
            a, b = Z[s0][:, j].float(), Z[s0][:, j - 1].float()
            rel.append(float(((a - b).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1)).median()))
        res["rel_change"] = rel
        res["spec_share"] = [float(R[s0][:, j].float().pow(2).sum() / Z[s0][:, j].float().pow(2).sum()) for j in range(nT)]
        dirs = torch.stack([Z[s0][:, j].float()[gt_t].mean(0) - Z[s0][:, j].float()[~gt_t].mean(0) for j in range(nT)])   # (nT, D)
        dn = torch.nn.functional.normalize(dirs, dim=-1)
        res["dir_cos"] = dn @ dn.T; res["gap"] = dirs.norm(dim=-1)
        logit_dirs = dirs @ W.cpu().T
        res["logit_dir_cos"] = torch.nn.functional.normalize(logit_dirs, dim=-1) @ torch.nn.functional.normalize(logit_dirs, dim=-1).T

    # (c) probes per step, fit on even boards, scored on odd; transfer between steps
    y = gt_t.reshape(-1).float().to(device)
    tr_mask = torch.from_numpy(np.repeat(even.numpy(), T)).to(device)
    te_mask = ~tr_mask
    probes, dec = [], np.zeros((nT, N, T), bool)
    for j in range(nT):
        X = Z[s0][:, j].float().reshape(-1, D).to(device)
        p = sa.fit_probe(X, y, tr_mask); probes.append(p)
        dec[j] = (p(X) > 0).reshape(N, T).cpu().numpy()
    acc = np.zeros((nT, nT))
    for i in range(nT):
        for j in range(nT):
            X = Z[s0][:, j].float().reshape(-1, D).to(device)
            acc[i, j] = float(((probes[i](X) > 0).float() == y)[te_mask].float().mean())
    res["probe_transfer_acc"] = acc                                                       # row: fit at t_i, column: applied at t_j
    oddn = odd.numpy()
    res["plan_vs_last"] = [float((dec[j][oddn] == dec[-1][oddn]).mean()) for j in range(nT)]
    res["plan_vs_drawn"] = [float((dec[j][oddn] == drawn[s0][oddn]).mean()) for j in range(nT)]
    res["plan_vs_gt"] = [float((dec[j][oddn] == gt[oddn]).mean()) for j in range(nT)]
    res["boards_exact"] = [float((dec[j][oddn] == gt[oddn]).all(1).mean()) for j in range(nT)]
    res["drawn_vs_gt"] = float((drawn[s0] == gt).mean())

    f = lambda v: " ".join(f"{t}:{x:.3f}" for t, x in zip(tl, v))
    print("\n(a) cosine of the same board's board-specific state between step t and the LAST step: " + f(res["cos_spec"][:, -1].tolist()))
    print("    ... and between consecutive steps: " + " ".join(f"{tl[j]}->{tl[j + 1]}:{float(res['cos_spec'][j, j + 1]):.3f}" for j in range(nT - 1)))
    print("    raw state, step t vs last: " + f(res["cos_raw"][:, -1].tolist()))
    print("    relative change between consecutive steps (median): " + " ".join(f"{tl[j + 1]}:{v:.3f}" for j, v in enumerate(rel)))
    print("    share of the state's energy that is board-specific: " + f(res["spec_share"]))
    print("(b) same board and step, OTHER noise seed: cosine of the board-specific part: " + f(torch.diagonal(res["cos_seed_spec"]).tolist()))
    print("(c) probe accuracy at its own step (odd boards): " + f([acc[j, j] for j in range(nT)]))
    print("    probe fit at t=%d applied at every step: " % tl[0] + f(acc[0].tolist()))
    print("    probe fit at t=%d applied at every step: " % tl[-1] + f(acc[-1].tolist()))
    print("(d) decoded plan at step t equal to the decoded plan at the last step (share of tokens): " + f(res["plan_vs_last"]))
    print("    ... equal to what is finally drawn: " + f(res["plan_vs_drawn"]) + f" | drawn vs solution: {res['drawn_vs_gt']:.3f}")
    print("    ... boards whose decoded plan is exactly the solution: " + f(res["boards_exact"]))
    print("(e) cosine of the path - off direction at step t with the one at the last step: " + f(res["dir_cos"][:, -1].tolist())
          + " | gap size: " + f(res["gap"].tolist()))
    print("    the same in logit space: " + f(res["logit_dir_cos"][:, -1].tolist()))
    torch.save(res, str(ic.out))
    torch.save({"Z": Z, "tl": tl, "gt": gt}, str(ic.out).replace(".pt", "_states.pt"))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
