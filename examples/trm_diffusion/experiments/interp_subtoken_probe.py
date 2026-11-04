"""TEMPORARY (Claude): is the sub-token path structure in the Thinker's state, and is it lost in the 11-channel readout?

Each 12x12 px token is split into s x s sub-blocks (s in sub_sizes, 12 % s == 0). Sub-block label from the GT solution
image: 1 if blue fraction >= 0.1, 0 if no blue, -1 (ignored) otherwise. A token is "mixed" when its known sub-block
labels disagree: that is where the token has to carry more than a single "path here" bit.

Mazes are sampled with the real pipeline (cfg.sampling, CFG). At the denoising steps in `record_steps`, on the
conditional pass, the features of every token are recorded:
  enc        condition-encoder output (TRM input, 512)
  z_H, z_L   final carry after n_sup reasoning steps (512 each)
  logits     lm_head(z_H), what goes to the ControlNet translator (11)
  logits3x3  logits of the token and its 8 neighbours (99), a generous upper bound on what bilinear upsampling +
             local convs of the ControlNet can read
A linear probe feature -> s*s sub-block labels is trained per (maze size, s, step, feature), 2-fold over boards.
Baselines on the held-out boards:
  token_oracle  every sub-block := the token's GT majority label (best a 1-bit-per-token readout can do)
  output        PaTh: sub-block labels read off the final generated image; plain (model_kind=plain,
                plain_checkpoint=...): the 2-class head's token decision broadcast to the token's sub-blocks
If z_H >> logits3x3 ~ painter on mixed tokens -> readout bottleneck. If z_H ~ token_oracle -> the state itself lacks it.
Output: <out_dir>/subtoken_probe.pt, subtoken_probe.json, printed tables.
"""
import dataclasses
import json
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
from models.sampling import CFGPredictor

CONFIG_DIR = os.environ.get("TRM_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / "configs"))
G, TOK = 12, 12                                                            # token grid, px per token
FEATS = ("enc", "z_H", "z_L", "logits", "logits3x3")


def ints(v):
    """Hydra list ([8,16]), int, or "8,16" string -> list of ints."""
    if isinstance(v, (int, str)):
        return [int(x) for x in str(v).split(",")]
    return [int(x) for x in v]


def blue(images):
    """(B, C, H, W) in [0, 1] -> (B, H, W) float blue mask."""
    return torch.from_numpy(np.stack([
        mcl.blue_mask((im.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for im in images
    ])).float()


def sub_labels(mask, s, thr=0.1, hard=False):
    """(B, 144, 144) blue mask -> (B, 144 tokens, s*s) labels (1 / 0 / -1 ambiguous; hard: frac >= thr)."""
    B = mask.shape[0]
    p = TOK // s
    frac = mask.reshape(B, G, s, p, G, s, p).mean((3, 6))                 # (B, G, s, G, s)
    frac = frac.permute(0, 1, 3, 2, 4).reshape(B, G * G, s * s)
    if hard:
        return (frac >= thr).float()
    lab = torch.full_like(frac, -1.0)
    lab[frac >= thr] = 1.0
    lab[frac == 0] = 0.0
    return lab


def train_probe(X, Y, steps, device, lr=3e-3, wd=1e-2, bs=1024):
    """Linear X (N, D) -> Y (N, K) in {1, 0, -1 = ignore}, standardized inputs, BCE with pos_weight."""
    X, Y = X.to(device).float(), Y.to(device)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-4)
    known = Y >= 0
    pw = ((Y == 0).sum() / (Y == 1).sum().clamp_min(1)).clamp(max=50)
    torch.manual_seed(0)
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(len(X), (bs,), device=device, generator=g)
        out = lin((X[idx] - mu) / sd)
        loss = F.binary_cross_entropy_with_logits(out, Y[idx].clamp(min=0), pos_weight=pw, reduction="none")
        m = known[idx].float()
        loss = (loss * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda Z: (lin((Z.to(device).float() - mu) / sd) > 0).float().cpu()


def scores(pred, lab):
    """pred/lab (N_tok, K). Returns acc over known sub-blocks of all tokens, and acc / path-F1 on mixed tokens."""
    known = lab >= 0
    full = known.all(1)
    mixed = full & (lab.max(1).values != lab.min(1).values)
    ok = (pred == lab) & known
    acc_all = ok.sum().item() / max(1, known.sum().item())
    pm, lm = pred[mixed], lab[mixed]
    acc_mixed = (pm == lm).float().mean().item() if mixed.any() else float("nan")
    tp = ((pm == 1) & (lm == 1)).sum().item()
    fp = ((pm == 1) & (lm == 0)).sum().item()
    fn = ((pm == 0) & (lm == 1)).sum().item()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    tok_exact = (pm == lm).all(1).float().mean().item() if mixed.any() else float("nan")
    return dict(acc_all=acc_all, acc_mixed=acc_mixed, f1_mixed=f1, tok_exact_mixed=tok_exact, n_mixed=int(mixed.sum()))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    sizes = ints(ic.get("maze_sizes", "8,11,13,16"))
    data_tmpl = str(ic.get("data_tmpl", "data/amaze/maze/" + str(ic.get("shape", "square")) + "/n{N}_test.parquet"))
    sub_sizes = ints(ic.get("sub_sizes", "1,2,3,4"))
    record_steps = ints(ic.get("record_steps", "0,10,19"))
    n_boards, bs, seed = int(ic.get("num_boards", 256)), int(ic.get("batch_size", 64)), int(ic.get("seed", 0))
    probe_steps = int(ic.get("probe_steps", 3000))
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    kind = str(ic.get("model_kind", "path"))
    if kind == "plain":                                                     # plain TRM: maze -> 2-class token head
        from train_plain_trm import PlainTRM
        ck = torch.load(str(ic.plain_checkpoint), map_location="cpu", weights_only=False)
        plain = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16)), noisy=bool(ck["args"].get("noisy", False)))
        plain.load_state_dict(ck["ema"]); plain = plain.to(device).eval()
        record_steps = [0]
        logging.info(f"[plain] {ic.plain_checkpoint}: step {ck['step']}, noisy={plain.noisy}")
        results, summary = run_probes(ic, out_dir, sizes, data_tmpl, sub_sizes, record_steps, n_boards, bs, seed,
                                      probe_steps, device, lambda batch, k0: collect_plain(plain, batch, device, seed + k0))
        return

    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
    pipe = instantiate(cfg.sampling)
    scale = pipe.cfg_scale

    # ---------------------------------------------------------------- recording (conditional pass only)
    st = {"on": False}

    def enc_hook(module, args, kwargs, out):
        if st["on"]:
            st["cur"]["enc"] = out.enc_emb.detach()

    model.condition_encoder.register_forward_hook(enc_hook, with_kwargs=True)
    orig_rs = model.thinker.reasoning_step

    def rs(*a, **k):
        logits, z_H, z_L = orig_rs(*a, **k)
        if st["on"]:
            st["cur"].update(logits=logits.detach(), z_H=z_H.detach(), z_L=z_L.detach())
        return logits, z_H, z_L

    model.thinker.reasoning_step = rs

    class RecPredictor(CFGPredictor):
        def predict(self, model_, sample, t, T):
            st.update(on=True, cur={})
            try:
                pred_cond = model_(sample).pred
            finally:
                st["on"] = False
            if st["step"] in record_steps:
                c = st["cur"]
                lg = c["logits"].float()                                     # (B, 144, 11)
                nb = nbhd(lg)
                st["rec"][st["step"]] = dict(enc=c["enc"].half().cpu(), z_H=c["z_H"].half().cpu(), z_L=c["z_L"].half().cpu(),
                                             logits=lg.half().cpu(), logits3x3=nb.half().cpu())
            st["step"] += 1
            if self.scale == 1.0:
                return pred_cond
            pred_uncond = model_(model_.null_condition_sample(sample)).pred
            return pred_uncond + self.scale * (pred_cond - pred_uncond)

    pipe.predictor = RecPredictor(scale)
    logging.info(f"sampling: {pipe.num_inference_steps} steps, cfg scale {scale}, recording steps {record_steps}")

    def collect_path(batch, k0):
        cond = model._batch_to_sample(batch, device)
        st.update(step=0, rec={})
        gen = pipe.sample_one_batch(model, cond, device, generator=torch.Generator(device=device).manual_seed(seed + k0))
        return st["rec"], blue(model.decode_for_eval(gen).float())

    run_probes(ic, out_dir, sizes, data_tmpl, sub_sizes, record_steps, n_boards, bs, seed, probe_steps, device, collect_path)


def nbhd(lg):
    """(B, 144, C) -> (B, 144, 9C): the token's and its 8 neighbours' values."""
    return F.unfold(lg.transpose(1, 2).reshape(lg.shape[0], -1, G, G), 3, padding=1).transpose(1, 2)


@torch.no_grad()
def collect_plain(plain, batch, device, seed):
    """Final features after n_sup steps; 'output' mask = the plain head's token decision broadcast to its 12x12 px."""
    maze = batch.spatial_conditions.to(device)
    x_t = t = None
    if plain.noisy:
        g = torch.Generator(device=device).manual_seed(seed)
        x_t = torch.randn(maze.shape, device=device, generator=g)
        from train_plain_trm import T_TRAIN                             # as PlainTRM.think(): pure noise, last t
        t = torch.full((maze.shape[0],), T_TRAIN - 1, device=device, dtype=torch.long)
    enc = (plain.encoder(maze, x_t, t) if plain.noisy else plain.encoder(maze)).enc_emb
    z_H, z_L = plain.init_carry(maze.shape[0], device)
    for _ in range(plain.n_sup):
        logits, z_H, z_L = plain.thinker.reasoning_step(enc, z_H, z_L)
    lg = logits.float()
    rec = {0: dict(enc=enc.half().cpu(), z_H=z_H.half().cpu(), z_L=z_L.half().cpu(), logits=lg.half().cpu(),
                   logits3x3=nbhd(lg).half().cpu())}
    tok = (lg.argmax(-1) == 1).float().reshape(-1, 1, G, G)
    return rec, F.interpolate(tok, scale_factor=TOK, mode="nearest")[:, 0].cpu()


def run_probes(ic, out_dir, sizes, data_tmpl, sub_sizes, record_steps, n_boards, bs, seed, probe_steps, device, collect):
    results, summary = {}, []
    for N in sizes:
        ds = AmazeDataset(dataset_path=data_tmpl.format(N=N), split="test", image_size=144, condition_field="m_original_img",
                          target_field="sol_img", num_channels=3)
        nb_ = min(n_boards, len(ds))
        feats = {k: {f: [] for f in FEATS} for k in record_steps}
        gt_mask, gen_mask = [], []
        for k0 in range(0, nb_, bs):
            smps = [ds[i] for i in range(k0, min(nb_, k0 + bs))]
            batch = AmazeDataset.collate_fn(smps)
            rec, out_mask = collect(batch, k0)
            gen_mask.append(out_mask)
            gt_mask.append(blue(batch.images.float()))
            for k in record_steps:
                for f in FEATS:
                    feats[k][f].append(rec[k][f])
            logging.info(f"n{N}: {k0 + len(smps)}/{nb_} boards")
        gt_mask, gen_mask = torch.cat(gt_mask), torch.cat(gen_mask)
        feats = {k: {f: torch.cat(v) for f, v in d.items()} for k, d in feats.items()}
        fold = torch.arange(nb_) % 2

        for s in sub_sizes:
            lab = sub_labels(gt_mask, s)                                     # (B, 144, s*s)
            gen_lab = sub_labels(gen_mask, s, hard=True)
            known = lab >= 0
            maj = ((lab == 1).sum(-1, keepdim=True) > (lab == 0).sum(-1, keepdim=True)).float().expand_as(lab)
            row = {"N": N, "s": s, "cells_per_token": (N / G) ** 2, "sub_px": TOK // s}
            heldout = {}
            for name, pred in (("token_oracle", maj), ("output", gen_lab)):
                heldout[name] = scores(pred.flatten(0, 1), lab.flatten(0, 1))
            for k in record_steps:
                for f in FEATS:
                    preds = torch.zeros_like(lab)
                    for fo in (0, 1):
                        tr, te = fold == fo, fold != fo
                        probe = train_probe(feats[k][f][tr].flatten(0, 1), lab[tr].flatten(0, 1), probe_steps, device)
                        preds[te] = probe(feats[k][f][te].flatten(0, 1)).reshape(int(te.sum()), G * G, s * s)
                    heldout[f"{f}@{k}"] = scores(preds.flatten(0, 1), lab.flatten(0, 1))
            results[(N, s)] = dict(row, scores=heldout, frac_known=float(known.float().mean()))
            summary.append(dict(row, **{f"{m}:{key}": v[key] for m, v in heldout.items() for key in ("acc_mixed", "f1_mixed", "acc_all")},
                                n_mixed=heldout["token_oracle"]["n_mixed"]))
            print(f"\n=== n{N} ({(N / G) ** 2:.2f} cells/token), s={s} ({TOK // s}px sub-blocks): "
                  f"{heldout['token_oracle']['n_mixed']} mixed tokens over {nb_} boards")
            print(f"   {'readout':16s} {'acc_all':>7s} {'acc_mix':>7s} {'F1_mix':>7s} {'tokEx':>6s}")
            for m, v in heldout.items():
                print(f"   {m:16s} {v['acc_all']:7.3f} {v['acc_mixed']:7.3f} {v['f1_mixed']:7.3f} {v['tok_exact_mixed']:6.3f}")
        del feats
        torch.save(results, out_dir / "subtoken_probe.pt")
        (out_dir / "subtoken_probe.json").write_text(json.dumps(summary, indent=1))
    print(f"saved {out_dir}/subtoken_probe.pt / .json")


    return results, summary


if __name__ == "__main__":
    main()
