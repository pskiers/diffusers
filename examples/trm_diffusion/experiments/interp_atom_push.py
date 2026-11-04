"""TEMPORARY diagnostic (cluster only): causal test of SAE atoms of the final z_H (interp_sae_name.py).

Atoms: the 8 most frequent (frequency ≥ 0.5%) plus the best detector of each candidate feature (wall_tip,
tip_wrapped3, tip_wrapped4, marker_edge). Push = beta · (typical activation of the atom) · scale · unit decoder
direction, added to z_H at EVERY z_H update of every reasoning step, conditional pass, every denoising step, at
fixed target tokens per board: 6 random off-path + 6 random path tokens. Controls: the same tokens, a random
direction with the median atom-push norm. beta in {3, 8}.
Outcome per target patch (12x12 px), vs the normal output of the same board: change in blue pixels (path ink),
dark pixels (walls), mean |Δ| per pixel; and globally the remaining path drawn and off-path cells drawn.
Odd boards reported (all boards were pushed; nothing here is fitted on boards).
"""
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import hv_features as hf
import interp_hv_swap as hs
import interp_sae_z as sz
import interp_z_edit as ze
import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

BETAS = (3.0, 8.0)


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
    run_ts = model.scheduler.timesteps[model.scheduler.timesteps <= 70]
    tc = model.thinker.inner.config
    per = tc.L_cycles + 1
    is_zH = lambda c: c % per == per - 1

    s = torch.load(f"{ic.sae_dir}/sae_zH_final.pt", weights_only=False)
    a = torch.load(f"{ic.sae_dir}/atoms_zH_final.pt", weights_only=False)
    sae = sz.TopKSAE(512, s["m"], s["k"]); sae.load_state_dict(s["sae"])
    Dn = (sae.D / sae.D.norm(dim=0, keepdim=True)).T.detach()                  # (m, 512), unit
    info, act_mean = a["info"], a["act_mean"]
    freq_atoms = sorted([j for j in info if j != "_best"], key=lambda j: -info[j]["freq"])[:8]
    cand = {f: js[0] for f, js in info["_best"].items()}
    atoms = list(dict.fromkeys(freq_atoms + list(cand.values())))
    label = {j: f"freq {info[j]['freq']:.3f}" for j in freq_atoms}
    for f, j in cand.items():
        label[j] = (label.get(j, "") + f" best {f}").strip()
    vec = {j: float(act_mean[j]) * s["scale"] * Dn[j] for j in atoms}
    med = float(np.median([float(v.norm()) for v in vec.values()]))
    g = torch.Generator().manual_seed(0)
    r = torch.randn(512, generator=g); vec["random"] = r / r.norm() * med
    print("atoms: " + "; ".join(f"{j} ({label[j]}, push norm {float(vec[j].norm()):.2f})" for j in atoms) + f"; random norm {med:.2f}", flush=True)

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    runs = [("normal", None, 0.0)] + [(j, j, b_) for j in atoms + ["random"] for b_ in BETAS]
    rows, n_board, saved = [], 0, []
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        gt = b["cats"] != "off"
        tgt = np.zeros((B, 144), bool); kind = np.full((B, 144), "", dtype=object)
        for i in range(B):
            rng = np.random.default_rng(1000 + n_board + i)
            for lab, pool in (("off", np.nonzero(~gt[i])[0]), ("path", np.nonzero(gt[i])[0])):
                pick = rng.choice(pool, size=min(6, len(pool)), replace=False)
                tgt[i, pick] = True; kind[i, pick] = lab
        m = torch.from_numpy(tgt).to(device)[..., None].float()
        outs = {}
        for name, j, beta in runs:
            v = None if j is None else (beta * vec[j]).to(device)

            def fn(t, c, out, v=v):
                if v is None or not is_zH(c):
                    return None
                return (out.float() + m * v).to(out.dtype)
            out = ze.denoise(model, b["conditions"], b["x_t"], run_ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), fn, B)
            outs[(name, beta)] = out
        base = outs[("normal", 0.0)]
        for (name, beta), out in outs.items():
            for i, (ms, r_) in enumerate(zip(mss, recs)):
                o = (out[i] * 255).permute(1, 2, 0).cpu().numpy(); o0 = (base[i] * 255).permute(1, 2, 0).cpu().numpy()
                bl, bl0 = mcl.blue_mask(o.astype(np.uint8)), mcl.blue_mask(o0.astype(np.uint8))
                dk, dk0 = o.mean(-1) < hf.WALL_DARK, o0.mean(-1) < hf.WALL_DARK
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                for lab in ("off", "path"):
                    toks = np.nonzero(kind[i] == lab)[0]
                    if not len(toks):
                        continue
                    db, dd, da = [], [], []
                    for tk in toks:
                        rr, cc = divmod(int(tk), 12); sl = (slice(rr * 12, rr * 12 + 12), slice(cc * 12, cc * 12 + 12))
                        db.append(int(bl[sl].sum()) - int(bl0[sl].sum())); dd.append(int(dk[sl].sum()) - int(dk0[sl].sum()))
                        da.append(float(np.abs(o[sl] - o0[sl]).mean()))
                    rows.append({"run": name, "beta": beta, "tokens": lab, "held_out": bool((n_board + i) % 2 == 1),
                                 "d_blue": float(np.mean(db)), "d_dark": float(np.mean(dd)), "abs_diff": float(np.mean(da)),
                                 "remaining_drawn": len(r_["remaining"] & pred) / len(r_["remaining"]) if r_["remaining"] else None,
                                 "off_path": len(pred - r_["path"])})
        if len(saved) < 2:
            saved.append({"out": {f"{k[0]}_{k[1]:g}": (v * 255).byte().cpu() for k, v in outs.items()}, "tgt": tgt, "kind": kind,
                          "sol": b["sol_u8"]})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)

    print("\nheld-out boards (odd); per target patch vs the normal output: Δ blue px, Δ dark px, mean |Δ| (0–255); "
          "then remaining path drawn / off-path cells")
    for name, j, beta in runs:
        line = []
        for lab in ("off", "path"):
            sel = [r_ for r_ in rows if r_["run"] == name and r_["beta"] == beta and r_["tokens"] == lab and r_["held_out"]]
            line.append(f"{lab}-tokens Δblue {np.mean([x['d_blue'] for x in sel]):+6.1f} Δdark {np.mean([x['d_dark'] for x in sel]):+6.1f} "
                        f"|Δ| {np.mean([x['abs_diff'] for x in sel]):5.1f}")
        selg = [r_ for r_ in rows if r_["run"] == name and r_["beta"] == beta and r_["tokens"] == "off" and r_["held_out"]]
        lab_ = label.get(j, "") if j not in (None, "random") else ""
        print(f"  {str(name):>7s} {lab_:28s} beta={beta:g} | " + " | ".join(line)
              + f" | remaining {np.mean([x['remaining_drawn'] for x in selg if x['remaining_drawn'] is not None]):.2f} off-path {np.mean([x['off_path'] for x in selg]):.1f}")
    torch.save({"rows": rows, "atoms": atoms, "label": label, "saved": saved}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
