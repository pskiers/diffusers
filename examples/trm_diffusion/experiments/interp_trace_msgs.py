"""TEMPORARY diagnostic (cluster only): E2b — the thinker's "messages" during reasoning step 1 (t=70, conditional pass).

For every call c of step 1 and attention layer l, head h, target token t and source token s, the exact contribution
of s to t through head h is  a[h,t,s] * W_O^h v^h_s  (a 512-dim vector). Projected on the path direction u_c (unit;
mean path state − mean off state at call c, even boards), it becomes one number:
    edge[c,l,h,t,s] = a[h,t,s] * (u_c · W_O^h v^h_s)       > 0: s pushes t towards "path", < 0: towards "off".
Saved for the first batch (16 boards) together with, per token and call: the projection on u_c of the state after the
call, of the whole attention output and of the MLP output of each layer. Exact for the attention step; the post-norms
and the MLP make the end-to-end effect approximate (to be checked causally).
For all boards: each head's output (after o_proj) at selected calls; held-out probes (fit even boards, score odd)
test what the messages encode: the token's own label, its 4 neighbours' labels (left/right/up/down = sequence
offsets −1/+1/−12/+12), how many neighbours are path, H / V stroke.
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
from models.trm.layers import apply_rotary_pos_emb

G = 12
PROBE_CALLS = (0, 2, 5, 7, 9, 12, 20)          # 0-based calls of step 1 (z_L 1, 3, 6, z_L 7, 9, 12; z_H at 20)


def neighbour_labels(gt):
    """gt: (B, 144) bool -> dict of (B, 144) float targets."""
    g = gt.reshape(-1, G, G).float()
    pad = torch.nn.functional.pad(g, (1, 1, 1, 1))
    left, right, up, down = pad[:, 1:-1, :-2], pad[:, 1:-1, 2:], pad[:, :-2, 1:-1], pad[:, 2:, 1:-1]
    f = lambda x: x.reshape(-1, G * G)
    return {"self": f(g), "left": f(left), "right": f(right), "up": f(up), "down": f(down),
            "n_nb_path": f(left + right + up + down)}


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
    inner = model.thinker.inner
    layers = inner.L_level.layers
    nL, nH, hd = len(layers), layers[0].self_attn.num_heads, layers[0].self_attn.head_dim
    tc = inner.config
    n_step1 = tc.H_cycles * (tc.L_cycles + 1)
    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=hs.IMG,
                      condition_field=vd.get("condition_field", "m_original_img"), target_field=vd.get("target_field", "sol_img"),
                      num_channels=3, include_metadata=True)
    n_total, batch_size, seed = min(100, len(ds)), 16, 0
    t70 = torch.tensor(70, device=device)

    st = {"call": -1, "on": False, "mode": None, "u": None, "batch_rec": None, "heads": None}

    def pre(module, args):
        if st["on"]:
            st["call"] += 1
    inner.L_level.register_forward_pre_hook(pre)

    # phase 1: path direction per call (state after the call), even boards
    sums = {}
    def out_hook(module, args, output):
        if st["on"] and st["mode"] == "dir" and st["call"] < n_step1:
            o = output.float(); m = st["mask"]
            sp, so = (o * m[0]).sum((0, 1)), (o * m[1]).sum((0, 1))
            if st["call"] in sums:
                sums[st["call"]][0] += sp; sums[st["call"]][1] += so
            else:
                sums[st["call"]] = [sp, so]
        if st["on"] and st["mode"] == "rec" and st["call"] < n_step1 and st["batch_rec"] is not None:
            st["batch_rec"]["state_proj"][:, st["call"]] = (output.float() @ st["u"][st["call"]]).cpu()
        return None
    inner.L_level.register_forward_hook(out_hook)

    def run_t70(b):
        s = dataclasses.replace(b["conditions"], x_noisy=b["x_t"], timesteps=t70.expand(b["B"]))
        st.update(call=-1, on=True)
        with torch.no_grad():
            model.forward_with_carry(s)
        st["on"] = False

    n_board, cnt = 0, [0.0, 0.0]
    starts = list(range(0, n_total, batch_size))
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        even = torch.from_numpy(np.array([(n_board + i) % 2 == 0 for i in range(b["B"])]))
        gt = torch.from_numpy(b["cats"] != "off")
        st["mask"] = ((gt & even[:, None]).float().to(device)[..., None], ((~gt) & even[:, None]).float().to(device)[..., None])
        st["mode"] = "dir"; run_t70(b)
        cnt[0] += float(st["mask"][0].sum()); cnt[1] += float(st["mask"][1].sum())
        n_board += b["B"]
    U = torch.stack([sums[c][0] / cnt[0] - sums[c][1] / cnt[1] for c in range(n_step1)])
    U = U / U.norm(dim=1, keepdim=True)
    st["u"] = U
    print(f"path directions for {n_step1} calls; cos(call1, call21) {float(U[0] @ U[-1]):+.2f}", flush=True)

    # phase 2: record
    def attn_hook(l):
        def f(module, args, kwargs, output):
            if not (st["on"] and st["mode"] == "rec" and st["call"] < n_step1):
                return None
            c = st["call"]
            x = kwargs["hidden_states"]; cos_sin = kwargs.get("cos_sin")
            B, S, _ = x.shape
            qkv = module.qkv_proj(x).view(B, S, module.num_heads + 2 * module.num_key_value_heads, module.head_dim)
            q, k = qkv[:, :, :nH], qkv[:, :, nH:2 * nH]
            v = qkv[:, :, 2 * nH:].float()
            if cos_sin is not None:
                q, k = apply_rotary_pos_emb(q, k, *cos_sin)
            a = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / hd ** 0.5, -1)        # (B,H,T,S)
            Wo = module.o_proj.weight.float()                                                                  # (D, H*hd)
            Wo_h = Wo.view(Wo.shape[0], nH, hd)                                                                # (D, H, hd)
            vo = torch.einsum("bshd,ehd->bhse", v, Wo_h)                                                       # (B,H,S,D) per-source write
            u = st["u"][c]
            pv = vo @ u                                                                                        # (B,H,S)
            rec = st["batch_rec"]
            if rec is not None:
                rec["edge"][:, c, l] = (a * pv[:, :, None, :]).half().cpu()
                rec["attn_proj"][:, c, l] = (output.float() @ u).cpu()
            if c in PROBE_CALLS:
                head_out = torch.einsum("bhts,bhse->bhte", a, vo)                                              # (B,H,T,D)
                st["heads"].setdefault((c, l), []).append(head_out.half().cpu())
            return None
        return f

    def mlp_hook(l):
        def f(module, args, output):
            if st["on"] and st["mode"] == "rec" and st["call"] < n_step1 and st["batch_rec"] is not None:
                st["batch_rec"]["mlp_proj"][:, st["call"], l] = (output.float() @ st["u"][st["call"]]).cpu()
            return None
        return f
    for l, layer in enumerate(layers):
        layer.self_attn.register_forward_hook(attn_hook(l), with_kwargs=True)
        layer.mlp.register_forward_hook(mlp_hook(l))

    st["heads"] = {}
    gts, saved, n_board = [], None, 0
    for start in starts:
        b = hs.prepare(ds, model, device, start, n_total, batch_size, seed)
        if b is None:
            continue
        B = b["B"]
        gts.append(torch.from_numpy(b["cats"] != "off"))
        if saved is None:
            st["batch_rec"] = {"edge": torch.zeros(B, n_step1, nL, nH, G * G, G * G, dtype=torch.float16),
                               "attn_proj": torch.zeros(B, n_step1, nL, G * G), "mlp_proj": torch.zeros(B, n_step1, nL, G * G),
                               "state_proj": torch.zeros(B, n_step1, G * G)}
        st["mode"] = "rec"; run_t70(b)
        if saved is None:
            saved = dict(st["batch_rec"], gt=gts[-1], cats=b["cats"], sol=b["sol_u8"], maze=b["maze_u8"], clean=b["clean_u8"], u=U.cpu())
            st["batch_rec"] = None
        n_board += B
        print(f"batch start={start}: {B} boards", flush=True)
    torch.save(saved, f"{ic.out_dir}/msgs_first_batch.pt")

    # what do the heads write? held-out probes on head outputs
    gt = torch.cat(gts); N = len(gt)
    tg = neighbour_labels(gt)
    A = torch.from_numpy(np.repeat(np.arange(N) % 2 == 0, G * G)).to(device)
    print("\nwhat each head writes (held-out R² of a ridge probe on the head's output; odd boards):")
    names = list(tg)
    print("  call layer head | " + " ".join(f"{n:>9s}" for n in names))
    res = {}
    for (c, l), parts in sorted(st["heads"].items()):
        Hc = torch.cat(parts).float()                                     # (N, H, T, D)
        for h in range(nH):
            X = Hc[:, h].reshape(N * G * G, -1).to(device)
            X1 = torch.cat([X, torch.ones(len(X), 1, device=device)], 1)
            XtX = X1[A].T @ X1[A] + 1.0 * torch.eye(X1.shape[1], device=device)
            r2 = []
            for n in names:
                y = tg[n].reshape(-1).to(device)
                w = torch.linalg.solve(XtX, X1[A].T @ y[A])
                pred = X1[~A] @ w; yb = y[~A]
                r2.append(1 - float(((pred - yb) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()))
            res[(c, l, h)] = r2
            print(f"  {c + 1:4d} {l:5d} {h:4d} | " + " ".join(f"{v:9.2f}" for v in r2), flush=True)
    torch.save({"probe_r2": res, "names": names}, f"{ic.out_dir}/head_writes.pt")
    print("saved", ic.out_dir)


if __name__ == "__main__":
    main()
