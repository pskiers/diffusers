"""TEMPORARY (Claude): what is the big shared part of PaTh's thinker state, and does the plan encode HOW to draw?

PaTh thinker (real CFG sampling, conditional pass, features at denoising steps `record_steps`) vs plain TRM (final
state after n_sup), same mazes (square n8 test set, GT path from sol_img).

A) energy decomposition of z_H / z_L / logits, Z (boards B, tokens T, D), exact and orthogonal:
     Z = g (mean over boards and tokens) + P[t] (mean over boards at token t, minus g) + R[b, t] (board-specific)
   shares |g|^2 BT, |P|^2 B, |R|^2 of |Z|^2, plus the participation ratio (effective dimension) of P and of R.
B) what the state encodes (ridge, 2-fold over boards, standardized features), per feature set:
     path      token path / off (GT, ambiguous tokens skipped)                                    balanced acc
     edge N/E/S/W  on path tokens: the GT path crosses that token edge                             balanced acc
     shape     on path tokens: vertical / horizontal / corner / end / other (from the edge bits)  balanced acc
     row, col  token position                                                                      R^2
   feature sets: enc (TRM input), z_H, z_L, logits (what the painter gets: 11 PaTh, 2 plain), and pos (one-hot
   token position, a position-prior baseline).
C) PaTh only: what the painter draws when, in the conditional pass, the logits are replaced by
     normal | template (per-token mean over the batch's boards = g + P) | const (g: mean over boards and tokens)
   path IoU (pixels, GT vs drawn blue), drawn blue pixels / GT blue pixels, and the share of GT path tokens drawn.
Output: <out_dir>/template.pt, printed tables.
"""
import dataclasses
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

import maze_corruption_lib as mcl
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model

CONFIG_DIR = os.environ.get("TRM_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / "configs"))
G, TOK = 12, 12
SHAPES = ("vertical", "horizontal", "corner", "end", "other")


def ints(v):
    if isinstance(v, (int, str)):
        return [int(x) for x in str(v).split(",")]
    return [int(x) for x in v]


def blue(images):
    return torch.from_numpy(np.stack([
        mcl.blue_mask((im.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for im in images
    ])).float()


def token_targets(mask):
    """(B, 144, 144) GT blue mask -> dict of per-token targets (B, 144)."""
    B = mask.shape[0]
    frac = mask.reshape(B, G, TOK, G, TOK).mean((2, 4)).reshape(B, -1)
    path = torch.full_like(frac, -1.0); path[frac >= 0.1] = 1; path[frac == 0] = 0
    m = mask.reshape(B, G, TOK, G, TOK)                                     # (B, r, y, c, x)
    edges = torch.zeros(B, G, G, 4)
    # a path crosses an edge when both pixel rows/cols straddling it hold blue (centre 8 px of the edge)
    edges[:, 1:, :, 0] = ((m[:, 1:, 0, :, 2:10].sum(-1) > 0) & (m[:, :-1, -1, :, 2:10].sum(-1) > 0)).float()   # N
    edges[:, :-1, :, 2] = edges[:, 1:, :, 0]                                                                     # S
    edges[:, :, 1:, 3] = ((m[:, :, 2:10, 1:, 0].sum(2) > 0) & (m[:, :, 2:10, :-1, -1].sum(2) > 0)).float()    # W
    edges[:, :, :-1, 1] = edges[:, :, 1:, 3]                                                                     # E
    edges = edges.reshape(B, G * G, 4)
    n, N_, E_, S_, W_ = edges.sum(-1), edges[..., 0], edges[..., 1], edges[..., 2], edges[..., 3]
    shape = torch.full_like(n, 4)                                                          # other (0 or >= 3 edges)
    shape[(n == 2) & (N_ == 1) & (S_ == 1)] = 0
    shape[(n == 2) & (E_ == 1) & (W_ == 1)] = 1
    shape[(n == 2) & (shape == 4)] = 2
    shape[n == 1] = 3
    rc = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"), -1).reshape(1, -1, 2).expand(B, -1, -1)
    return dict(path=path, edges=edges, shape=shape, row=rc[..., 0].float(), col=rc[..., 1].float())


def ridge_fit_predict(Xtr, Ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-4)
    A, Bt = (Xtr - mu) / sd, (Xte - mu) / sd
    A = torch.cat([A, torch.ones(len(A), 1, device=A.device)], 1)
    Bt = torch.cat([Bt, torch.ones(len(Bt), 1, device=Bt.device)], 1)
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=A.device), A.T @ Ytr)
    return Bt @ W


def bal_acc(pred, y, n_cls):
    accs = [(pred[y == c] == c).float().mean() for c in range(n_cls) if (y == c).any()]
    return float(torch.stack(accs).mean())


def decode(X, tg, fold, device):
    """X (B, T, D). 2-fold over boards. Returns dict of scores."""
    X = X.float().to(device)
    out = {}
    preds = {k: [] for k in ("path", "edges", "shape", "rc")}
    ys = {k: [] for k in preds}
    for fo in (0, 1):
        tr, te = fold == fo, fold != fo
        Xtr, Xte = X[tr].flatten(0, 1), X[te].flatten(0, 1)
        p_tr, p_te = tg["path"][tr].flatten().to(device), tg["path"][te].flatten().to(device)
        k_tr, k_te = p_tr >= 0, p_te >= 0
        s = ridge_fit_predict(Xtr[k_tr], (2 * p_tr[k_tr] - 1)[:, None], Xte[k_te])[:, 0]
        preds["path"].append((s > 0).float()); ys["path"].append(p_te[k_te])
        on_tr, on_te = p_tr == 1, p_te == 1                                   # path tokens only
        e_tr, e_te = tg["edges"][tr].flatten(0, 1).to(device), tg["edges"][te].flatten(0, 1).to(device)
        s = ridge_fit_predict(Xtr[on_tr], 2 * e_tr[on_tr] - 1, Xte[on_te])
        preds["edges"].append((s > 0).float()); ys["edges"].append(e_te[on_te])
        sh_tr, sh_te = tg["shape"][tr].flatten().to(device).long(), tg["shape"][te].flatten().to(device).long()
        s = ridge_fit_predict(Xtr[on_tr], 2 * F.one_hot(sh_tr[on_tr], len(SHAPES)).float() - 1, Xte[on_te])
        preds["shape"].append(s.argmax(1)); ys["shape"].append(sh_te[on_te])
        rc_tr = torch.stack([tg["row"][tr].flatten(), tg["col"][tr].flatten()], 1).to(device)
        rc_te = torch.stack([tg["row"][te].flatten(), tg["col"][te].flatten()], 1).to(device)
        preds["rc"].append(ridge_fit_predict(Xtr, rc_tr, Xte)); ys["rc"].append(rc_te)
    P = {k: torch.cat(v) for k, v in preds.items()}; Y = {k: torch.cat(v) for k, v in ys.items()}
    out["path"] = bal_acc(P["path"], Y["path"], 2)
    for i, e in enumerate("NESW"):
        out[f"edge_{e}"] = bal_acc(P["edges"][:, i], Y["edges"][:, i], 2)
    out["shape"] = bal_acc(P["shape"], Y["shape"], len(SHAPES))
    for i, k in enumerate(("row", "col")):
        y = Y["rc"][:, i]
        out[k] = float(1 - ((P["rc"][:, i] - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return out


def pr(M):
    """Participation ratio of the PCA spectrum of the rows of M (centred)."""
    M = M.reshape(-1, M.shape[-1]).float()
    ev = torch.linalg.svdvals(M - M.mean(0)).pow(2)
    return float(ev.sum() ** 2 / ev.pow(2).sum())


def energy(Z):
    Z = Z.float()
    g = Z.mean((0, 1), keepdim=True)
    P = Z.mean(0, keepdim=True) - g
    R = Z - Z.mean(0, keepdim=True)
    B, T = Z.shape[:2]
    tot = Z.pow(2).sum()
    return dict(const=float(g.pow(2).sum() * B * T / tot), pos=float(P.pow(2).sum() * B / tot), board=float(R.pow(2).sum() / tot),
                pr_pos=pr(P[0]), pr_board=pr(R), norm=float(Z.norm(dim=-1).mean()))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    n_boards, seed = int(ic.get("num_boards", 100)), int(ic.get("seed", 0))
    record_steps = ints(ic.get("record_steps", "0,10"))
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    ds = AmazeDataset(dataset_path=str(ic.data_parquet), split="test", image_size=144, condition_field="m_original_img",
                      target_field="sol_img", num_channels=3)
    nb = min(n_boards, len(ds))
    batch = AmazeDataset.collate_fn([ds[i] for i in range(nb)])
    gt = blue(batch.images.float())
    tg = token_targets(gt)
    fold = torch.arange(nb) % 2
    n_path = int((tg["path"] == 1).sum())
    sh = tg["shape"][tg["path"] == 1]
    print(f"{nb} boards, {n_path} path tokens; shapes: " + ", ".join(f"{s} {int((sh == i).sum())}" for i, s in enumerate(SHAPES)))
    feats = {}

    # ---------------------------------------------------------------- plain TRM
    if str(ic.get("plain_checkpoint", "none")) != "none":
        from train_plain_trm import PlainTRM, T_TRAIN
        ck = torch.load(str(ic.plain_checkpoint), map_location="cpu", weights_only=False)
        plain = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16)), noisy=bool(ck["args"].get("noisy", False)))
        plain.load_state_dict(ck["ema"]); plain = plain.to(device).eval()
        with torch.no_grad():
            maze = batch.spatial_conditions.to(device)
            x_t = t = None
            if plain.noisy:
                x_t = torch.randn(maze.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
                t = torch.full((nb,), T_TRAIN - 1, device=device, dtype=torch.long)
            enc = (plain.encoder(maze, x_t, t) if plain.noisy else plain.encoder(maze)).enc_emb
            z_H, z_L = plain.init_carry(nb, device)
            for _ in range(plain.n_sup):
                lg, z_H, z_L = plain.thinker.reasoning_step(enc, z_H, z_L)
        feats["plain"] = dict(enc=enc.float().cpu(), z_H=z_H.float().cpu(), z_L=z_L.float().cpu(), logits=lg.float().cpu())
        del plain
        logging.info(f"[plain] {ic.plain_checkpoint}: step {ck['step']}")

    # ---------------------------------------------------------------- PaTh: sampling under three plan conditions
    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
    scale = instantiate(cfg.sampling).cfg_scale
    st = {"rec": False, "plan": "normal"}
    model.condition_encoder.register_forward_hook(
        lambda m, a, kw, out: st.__setitem__("enc", out.enc_emb.detach()) if st["rec"] else None, with_kwargs=True)
    orig_rs, orig_rp = model.thinker.reasoning_step, model.run_painter

    def rs(*a, **k):
        lg, zh, zl = orig_rs(*a, **k)
        if st["rec"]:
            st.update(z_H=zh.detach(), z_L=zl.detach())
        return lg, zh, zl

    def rp(sample, logits):
        if st["rec"] and st["plan"] == "template":
            logits = logits.mean(0, keepdim=True).expand_as(logits)
        elif st["rec"] and st["plan"] == "const":
            logits = logits.mean((0, 1), keepdim=True).expand_as(logits)
        if st["rec"]:
            st["logits"] = logits.detach()
        return orig_rp(sample, logits)

    model.thinker.reasoning_step, model.run_painter = rs, rp
    cond = model._batch_to_sample(batch, device)
    T = model.scheduler.config.num_train_timesteps
    draw = {}
    for plan in ("normal", "template", "const"):
        st["plan"] = plan
        model.scheduler.set_timesteps(20, device=device)
        g_ = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((nb, *model.noise_shape), device=device, generator=g_)
        with torch.no_grad():
            for i, t in enumerate(model.scheduler.timesteps):
                s = dataclasses.replace(cond, x_noisy=x, timesteps=t.expand(nb).to(device))
                st["rec"] = True
                pc = model(s).pred
                st["rec"] = False
                if plan == "normal" and i in record_steps:
                    feats[f"path@{i}"] = dict(enc=st["enc"].float().cpu(), z_H=st["z_H"].float().cpu(),
                                              z_L=st["z_L"].float().cpu(), logits=st["logits"].float().cpu())
                pu = model(model.null_condition_sample(s)).pred
                x = model.scheduler.step(pu + scale * (pc - pu), t, x).prev_sample
        dm = blue(model.decode_for_eval(x).float())
        inter, union = (dm * gt).sum((1, 2)), ((dm + gt) > 0).float().sum((1, 2))
        dtok = (dm.reshape(nb, G, TOK, G, TOK).mean((2, 4)).reshape(nb, -1) >= 0.1)
        pth = tg["path"] == 1
        draw[plan] = dict(iou=float((inter / union.clamp_min(1)).mean()), blue_ratio=float(dm.sum() / gt.sum()),
                          path_tok_drawn=float(dtok[pth].float().mean()), off_tok_drawn=float(dtok[tg["path"] == 0].float().mean()))
        logging.info(f"plan={plan}: {draw[plan]}")

    # ---------------------------------------------------------------- report
    pos = F.one_hot(torch.arange(G * G), G * G).float()[None].expand(nb, -1, -1)
    res = dict(energy={}, decode={}, draw=draw)
    print("\nA) energy shares of the state: const (same vector everywhere) | position template | board-specific "
          "|| eff. dim of template | eff. dim of board part | mean norm")
    for name, fs in feats.items():
        for k in ("z_H", "z_L", "logits", "enc"):
            e = energy(fs[k]); res["energy"][(name, k)] = e
            print(f"   {name:8s} {k:6s} {e['const']:6.1%} {e['pos']:6.1%} {e['board']:6.1%} || {e['pr_pos']:6.2f} {e['pr_board']:6.2f} | {e['norm']:7.2f}")
    cols = ["path"] + [f"edge_{e}" for e in "NESW"] + ["shape", "row", "col"]
    print("\nB) linear decoding (2-fold over boards): balanced accuracy (path, edges and shape on path tokens) | R^2 (row, col)")
    print(f"   {'':8s} {'feat':6s} " + " ".join(f"{c:>7s}" for c in cols))
    rows = [("position", "pos", pos)] + [(n, k, fs[k]) for n, fs in feats.items() for k in ("enc", "z_H", "z_L", "logits")]
    for name, k, X in rows:
        d = decode(X, tg, fold, device); res["decode"][(name, k)] = d
        print(f"   {name:8s} {k:6s} " + " ".join(f"{d[c]:7.3f}" for c in cols))
    print("\nC) PaTh painter with the conditional-pass plan replaced: path IoU | drawn/GT blue | GT path tokens drawn | off tokens drawn")
    for plan, d in draw.items():
        print(f"   {plan:9s} {d['iou']:6.3f} {d['blue_ratio']:6.2f} {d['path_tok_drawn']:6.3f} {d['off_tok_drawn']:6.3f}")
    torch.save(res, out_dir / "template.pt")
    print(f"saved {out_dir}/template.pt")


if __name__ == "__main__":
    main()
