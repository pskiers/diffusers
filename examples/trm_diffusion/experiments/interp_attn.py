"""TEMPORARY diagnostic (cluster only): E2 — what inside the thinker's block produces each change of z?

The block (L_level) = 2 post-norm transformer layers (8 attention heads each, 1-D RoPE over the row-major 12x12 grid,
SwiGLU MLP); the same block makes every z_L and z_H update. Two parts:
(1) Attention maps, recomputed exactly from each attention module's input (q, k with RoPE; softmax(qk/sqrt(d))), in the
    conditional pass at t=70 during reasoning step 1 (calls 1–21), clean boards as before. Saved in full for the first
    batch (16 boards: calls x layers x heads x 144 x 144); for all boards, per (call, layer, head) the attention mass
    by grid distance between query and key token (Chebyshev 0, 1, 2, 3, 4, ≥5) and on the 4 grid neighbours.
(2) Head ablation (causal): one head's output zeroed (before o_proj) at every call of the conditional pass at every
    denoising step; with the full budget and with n_sup=1 (no later reasoning steps that could repair it).
    Outcome = what the painter draws (odd boards): solved, remaining path drawn, off-path cells.
"""
import dataclasses

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import amaze_interpretability as ai
import interp_hv_swap as hs
import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from models.trm.layers import apply_rotary_pos_emb

G = 12
rr, cc = np.divmod(np.arange(G * G), G)
CHEB = np.minimum(np.maximum(np.abs(rr[:, None] - rr[None]), np.abs(cc[:, None] - cc[None])), 5)
NB4 = (np.abs(rr[:, None] - rr[None]) + np.abs(cc[:, None] - cc[None])) == 1


@torch.no_grad()
def denoise(model, conditions, x, ts, generator, n_sup=None):
    predictor = model.sampling_pipeline.predictor
    for t in ts:
        s = dataclasses.replace(conditions, x_noisy=x, timesteps=t.expand(x.shape[0]).to(x.device))
        model._cur_t = int(t)
        model._in_cond = True
        pred_cond, _, _ = model.forward_with_carry(s, n_sup=n_sup)
        model._in_cond = False
        pred_uncond = model(model.null_condition_sample(s)).pred
        pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)
        x = model.scheduler.step(pred, t, x, generator=generator).prev_sample
    return model.decode_for_eval(x).clamp(0, 1)


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
    ts = model.scheduler.timesteps[model.scheduler.timesteps <= 70]
    inner = model.thinker.inner
    layers = inner.L_level.layers
    n_layers, n_heads = len(layers), layers[0].self_attn.num_heads
    hd = layers[0].self_attn.head_dim
    tc = inner.config
    n_step1 = tc.H_cycles * (tc.L_cycles + 1)
    print(f"{n_layers} layers x {n_heads} heads, head_dim {hd}; recording calls 1..{n_step1} at t=70", flush=True)

    st = {"call": 0, "rec": False, "maps": None, "stats": None, "ablate": None}
    model._in_cond = False; model._cur_t = None

    def count(module, args):
        if model._in_cond:
            st["call"] += 1
    inner.L_level.register_forward_pre_hook(count)

    def attn_hook(l):
        def f(module, args, kwargs, output):
            if not (st["rec"] and model._in_cond and model._cur_t == 70 and st["call"] <= n_step1):
                return None
            hs_ = kwargs["hidden_states"]; cos_sin = kwargs.get("cos_sin")
            B, S, _ = hs_.shape
            qkv = module.qkv_proj(hs_).view(B, S, module.num_heads + 2 * module.num_key_value_heads, module.head_dim)
            q, k = qkv[:, :, :module.num_heads], qkv[:, :, module.num_heads:module.num_heads + module.num_key_value_heads]
            if cos_sin is not None:
                q, k = apply_rotary_pos_emb(q, k, *cos_sin)
            a = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / module.head_dim ** 0.5, -1)   # (B,H,S,S)
            c = st["call"] - 1
            st["stats"].append((c, l, a))
            return None
        return f
    for l, layer in enumerate(layers):
        layer.self_attn.register_forward_hook(attn_hook(l), with_kwargs=True)

    def ablate_pre(l):
        def f(module, args):
            if st["ablate"] is None or not model._in_cond or st["ablate"][0] != l:
                return None
            x = args[0].clone()
            h = st["ablate"][1]
            x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        return f
    for l, layer in enumerate(layers):
        layer.self_attn.o_proj.register_forward_pre_hook(ablate_pre(l))

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    cheb = torch.from_numpy(CHEB).to(device); nb4 = torch.from_numpy(NB4).to(device).float()
    dist_mass = torch.zeros(n_step1, n_layers, n_heads, 6, device=device); nb_mass = torch.zeros(n_step1, n_layers, n_heads, device=device)
    nq = 0
    full_maps, rows, n_board = None, [], 0
    runs = [("normal", None, None), ("normal_nsup1", None, 1)] + [(f"L{l}H{h}", (l, h), ns) for ns in (None, 1)
                                                                    for l in range(n_layers) for h in range(n_heads)]
    for start in range(0, n_total, batch_size):
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B, mss, recs = b["B"], b["mss"], b["recs"]
        for name, abl, ns in runs:
            st.update(call=0, ablate=abl, rec=(name == "normal"), stats=[])

            # reset the call counter at the start of every conditional pass
            def fwc_wrap(fn=model.forward_with_carry):
                def g(*a, **k):
                    st["call"] = 0
                    return fn(*a, **k)
                return g
            orig = model.forward_with_carry
            model.forward_with_carry = fwc_wrap(orig)
            out = denoise(model, b["conditions"], b["x_t"], ts, torch.Generator(device=device).manual_seed(b["base_seed"] + 1), ns)
            model.forward_with_carry = orig
            if name == "normal":
                for c, l, a in st["stats"]:
                    for dd in range(6):
                        dist_mass[c, l, :, dd] += (a * (cheb == dd)).sum(-1).sum((0, 2))
                    nb_mass[c, l] += (a * nb4).sum(-1).sum((0, 2))
                nq += B * G * G
                if full_maps is None:
                    full_maps = torch.zeros(B, n_step1, n_layers, n_heads, G * G, G * G, dtype=torch.float16)
                    for c, l, a in st["stats"]:
                        full_maps[:, c, l] = a.half().cpu()
            for i, (ms, r_) in enumerate(zip(mss, recs)):
                pred = mcl.predicted_cells(ai._to_native(out[i], ms), ms)
                inner_cells = r_["path"] - {ms.path[0], ms.path[-1]}
                rows.append({"run": name, "n_sup": ns, "board": n_board + i, "held_out": (n_board + i) % 2 == 1,
                             "solved": inner_cells <= pred and not (pred - r_["path"]),
                             "remaining_drawn": len(r_["remaining"] & pred) / len(r_["remaining"]) if r_["remaining"] else None,
                             "off_path": len(pred - r_["path"])})
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)
    dist_mass /= nq; nb_mass /= nq
    print("\nattention locality at t=70, averaged over calls 1–21 (mass per Chebyshev distance 0/1/2/3/4/≥5 | on the 4 grid neighbours):")
    for l in range(n_layers):
        for h in range(n_heads):
            m = dist_mass[:, l, h].mean(0).cpu().numpy()
            print(f"  L{l}H{h}: " + " ".join(f"{x:.2f}" for x in m) + f" | 4-neighbours {float(nb_mass[:, l, h].mean()):.2f}")
    print("  (uniform attention would give: " + " ".join(f"{(CHEB == dd).sum() / CHEB.size:.2f}" for dd in range(6))
          + f" | {NB4.sum() / NB4.size:.3f})")
    print("\nhead ablation (one head zeroed at every call, conditional pass, every denoising step), held-out boards:")
    for name, abl, ns in runs:
        sel = [r for r in rows if r["run"] == name and r["held_out"]]
        print(f"  {name:13s} budget {'n_sup=1' if ns == 1 else 'full'} | solved {np.mean([r['solved'] for r in sel]):.2f}  "
              f"remaining drawn {np.mean([r['remaining_drawn'] for r in sel if r['remaining_drawn'] is not None]):.3f}  "
              f"off-path {np.mean([r['off_path'] for r in sel]):.2f}")
    torch.save({"rows": rows, "dist_mass": dist_mass.cpu(), "nb_mass": nb_mass.cpu(), "full_maps_first_batch": full_maps}, str(ic.out))
    print("saved", ic.out)


if __name__ == "__main__":
    main()
