# ABOUTME: CLEVR saturation screen and per-sample gradient noise (B) of a frozen painter under oracle layout steering.
# ABOUTME: The CLEVR port of saturation_screen.py: the oracle is the scene's centroid mask on the thinker's token grid.
"""Does a frozen CLEVR painter reward layout information from the thinker, and how cleanly does it ask for it?

Oracle input (what a perfect thinker could output), per image token of the translator's grid (16x16), 16 channels
in make_mask_from_scene's layout (0-7 color, 8-10 shape, 11-12 material, 13-14 size, 15 presence; the 32x32
centroid mask average-pooled to the grid). Three information levels per scene:
  uni    the scene's mean token at every position: which attributes the scene has, nowhere in particular
  anon   presence channel kept, attribute channels = presence x the scene's attribute histogram: where objects
         are, not which is where (the UNet thinker's presence-only centroid mask)
  bound  the full mask: which object is where
and three blend paths x(alpha) = start + alpha (end - start): where (uni -> anon), binding (anon -> bound),
full (uni -> bound). The oracle translator (the config's translator, input = these 16 channels) is trained on
all three, a random path and alpha ~ U(0, 1) per sample, with the config's painter blinding.

Printed per path and alpha, on fixed val samples (t, noise drawn once per batch as in training):
  red     loss reduction vs the unsteered painter (blinded if the config blinds it), per noise band
  cos     cosine between the request r = -dL/dx and the path direction d = end - x, per sample
  point   share of informative tokens (|d_token| > 0.05) whose request has a positive component along d
  SNR     mean / std over informative tokens of the margin m = <r_token, d_token / |d_token|>
  pooled  per sample, the request summed over a grid of 14 noise levels with one noise image: point, cos
SUMMARY: sat = red(full, 0.1) / red(full, 1) over all t, pooled point at alpha 0 / 0.1 / 0.25 per path.
Noise bands are on (1 - t) x 100 (0 clean, 100 pure noise) for flow painters, DDPM indices otherwise.
Reference lines: the conditional painter's own loss (unblinded, unsteered), so "red" can be read against it.

Levers (training and path evaluation, off for the "deploy" lines):
  BNOISE=s   Gaussian noise of std s on the tokens after the normalizing bridge (translator.input_noise_std)
With GNS=1 (GNS_K=16): only the per-sample gradient noise B_margin per path and alpha: draws of (t, noise) a sample
needs until its averaged margin's mean equals its noise (sum of var / sum of mean^2 over informative tokens),
t from the painter's training distribution, and the share of the margin signal per band.

    [BATCH=64] [N_EVAL=8] [ENV...] python experiments/diagnostics/clevr_saturation_screen.py <overrides.txt> \
        <painter_ckpt> <name> <translator.pt> [steps]

<overrides.txt>: thinker-stage config, one Hydra override per line (RUN=, painter.checkpoint= and eval* lines are
skipped); thinker.vocab_size must be 16. <translator.pt> is trained (default 600 steps) if missing, else loaded.
Smoke: BATCH=8 N_EVAL=1 GNS_K=2 with steps 5.
"""
import dataclasses
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from datasets.data_sample import collate_data_samples
from factory import build_model

ov_file, painter_ckpt, name, tr_path = sys.argv[1:5]
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600
dev = torch.device("cuda")
BATCH, LR, N_VAL = int(os.environ.get("BATCH", 64)), 1e-3, int(os.environ.get("N_EVAL", 8))
BANDS = [(0, 30), (30, 60), (60, 100)]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]
PATHS = ["where", "binding", "full"]
BNOISE = float(os.environ.get("BNOISE", 0))
T_GRID = list(range(2, 100, 7))
N_POOL = 4
INFO_EPS = 0.05
C = 16

ov = [line.strip() for line in open(ov_file)
      if line.strip() and not line.startswith(("RUN=", "painter.checkpoint=", "eval"))]
ov += [f"painter.checkpoint={painter_ckpt}", "run.wandb_project=null", "data.include_centroid_mask=true",
       "data.centroid_mask_presence_only=false"]
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    cfg = compose(config_name="config", overrides=ov)
cfg.eval_callbacks = None
model = build_model(cfg, instantiate(cfg.diffusion)).to(dev)
assert model.thinker.vocab_size == C, f"thinker.vocab_size must be {C} (the oracle's channels)"
tr = model.thinker_painter_translator
G = tr._grid
FLOW = getattr(model.painter, "objective", "ddpm") == "flow"
for p in model.parameters():
    p.requires_grad_(False)
print(f"== {name}  (bridge noise {BNOISE}; steer blocks {cfg.painter.steer_from_block}.."
      f"{cfg.painter.steer_to_block or cfg.painter.net.depth} - 1 ({len(tr.projs)} projections), "
      f"translator {type(tr).__name__} grid {G}, "
      f"blinded {bool(model.train_cfg.force_unconditional_painter)}, painter dtype "
      f"{getattr(model.painter, 'painter_dtype', None)})", flush=True)
REG = {"on": True}
bridge = tr._bridge_tokens


def bridge_reg(logits):
    tok = bridge(logits)
    if REG["on"] and BNOISE > 0:
        tok = tok + BNOISE * torch.randn_like(tok)
    return tok


tr._bridge_tokens = bridge_reg


def level(t):
    """Noise level on the DDPM index scale (0 clean .. 100 pure noise); flow t runs 0 = noise -> 1 = clean."""
    return (1 - t.float()) * 100 if FLOW else t.float()


def renoise(s, lv, noise):
    """s with x_t (and target) redrawn at noise level lv (B,) from standard-normal noise, as the painter's training."""
    if FLOW:
        p = model.painter
        t = (1 - lv.float() / 100).clamp(1e-5, 1 - 1e-5)
        t4 = t[:, None, None, None]
        x_t = s.images * t4 + noise * p.noise_scale * (1 - t4)
        target = (s.images - x_t) / (1 - t4).clamp_min(p.velocity_clamp)
        st = dataclasses.replace(s, x_noisy=x_t, timesteps=t, target=target, painter_hidden=None)
    else:
        tb = lv.long()
        st = dataclasses.replace(s, x_noisy=model.painter.scheduler.add_noise(s.images, noise, tb), timesteps=tb,
                                 painter_hidden=None)
    return model.painter.fill_runtime_fields(st)


def draw_levels(b):
    """(b,) training noise levels: the flow painter's logit-normal t, else uniform DDPM indices."""
    if FLOW:
        p = model.painter
        t = torch.sigmoid(torch.randn(b, device=dev) * p.t_logit_std + p.t_logit_mean).clamp(1e-5, 1 - 1e-5)
        return (1 - t) * 100
    return torch.randint(0, 100, (b,), device=dev)


def painter_loss(s, pred):
    return (pred.float() - s.target.float()).pow(2).flatten(1).mean(1)


def unsteered(s, blinded):
    """The frozen painter's prediction without steering, on the same input and dtype as run_painter."""
    ps = model.painter.null_condition_sample(s) if blinded else dataclasses.replace(s, painter_hidden=None)
    dt = getattr(model.painter, "painter_dtype", None)
    ctx = torch.autocast("cuda", dtype=dt) if dt is not None else torch.autocast("cuda", enabled=False)
    with ctx:
        return model.painter(ps).pred.float()


def levels(s):
    """(uni, anon, bound), each (B, G*G, 16), from the sample's centroid mask and object attributes."""
    m = s.spatial_conditions.float()
    assert m.shape[1:] == (C, 32, 32), m.shape
    bound = F.adaptive_avg_pool2d(m, G).flatten(2).transpose(1, 2)  # (B, G*G, 16)
    uni = bound.mean(1, keepdim=True).expand_as(bound)
    e, om = s.embedding_conditions.float(), s.embedding_mask.float()  # (B, 10, 55), (B, 10)
    size, mat = e[..., 2].long(), e[..., 3].long()
    attr = torch.cat([e[..., 7:15], e[..., 4:7], F.one_hot(mat, 2).float(), F.one_hot(size, 2).float()], -1)
    hist = (attr * om[..., None]).sum(1) / om.sum(1, keepdim=True).clamp(min=1)  # (B, 15)
    anon = torch.cat([bound[..., 15:] * hist[:, None], bound[..., 15:]], -1)
    return uni, anon, bound


def path_ends(lv, path):
    uni, anon, bound = lv
    return {"where": (uni, anon), "binding": (anon, bound), "full": (uni, bound)}[path]


def blend(lv, path, alpha):
    start, end = path_ends(lv, path)
    a = torch.as_tensor(alpha, device=dev, dtype=torch.float32).view(-1, 1, 1)
    return start + a * (end - start), end


def request_stats(r, d):
    """Per-sample cosine, informative-token pointing and margins of request r along direction d, (B, T, C)."""
    cos = F.cosine_similarity(r.flatten(1), d.flatten(1), dim=1)
    dn = d.norm(dim=-1)
    info = dn > INFO_EPS
    m = (r * d).sum(-1) / dn.clamp(min=1e-8)
    return cos, m[info], info


if os.path.exists(tr_path):
    tr.load_state_dict(torch.load(tr_path, map_location="cpu", weights_only=True))
    print(f"loaded translator {tr_path}")
else:
    for p in tr.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(tr.parameters(), lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(instantiate(cfg.data.train_dataset), batch_size=BATCH, shuffle=True, generator=g,
                        collate_fn=collate_data_samples, num_workers=4, drop_last=True)
    it = iter(loader)
    model.train()
    for step in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = LR * min(1.0, step / 100)
        [d] = model.prep_mb_data([next(it)], dev)
        s = d["sample"]
        lv = levels(s)
        which = torch.randint(0, len(PATHS), (BATCH,), device=dev)
        a = torch.rand(BATCH, device=dev)
        x = torch.stack([blend(lv, p, a)[0] for p in PATHS])[which, torch.arange(BATCH, device=dev)]
        if step == 1:
            print(f"first batch: images {tuple(s.images.shape)} {s.images.dtype}, mask "
                  f"{tuple(s.spatial_conditions.shape)}, oracle {tuple(x.shape)}; token norms uni/anon/bound "
                  + "/".join(f"{v.norm(dim=-1).mean():.3f}" for v in lv), flush=True)
        loss = painter_loss(s, model.run_painter(s, x)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"  translator step {step}: loss {loss.item():.5f}", flush=True)
    torch.save(tr.state_dict(), tr_path)
    for p in tr.parameters():
        p.requires_grad_(False)
    print(f"trained translator ({steps} steps) -> {tr_path}")

model.eval()  # translator BatchNorm on its running statistics: a fixed map per sample
val_ds = instantiate(cfg.data.val_dataset)
random.seed(1234)  # reduced-relation variant and edge-direction coins per val scene
np.random.seed(1234)
val = []
for i in range(N_VAL):
    b = collate_data_samples([val_ds[j] for j in range(i * BATCH, (i + 1) * BATCH)])
    with torch.random.fork_rng(devices=[dev]):
        torch.manual_seed(1000 + i)
        [d] = model.prep_mb_data([b], dev)
    val.append(d["sample"])
LV = [levels(s) for s in val]

if os.environ.get("GNS"):
    K = int(os.environ.get("GNS_K", 16))
    print(f"GNS: K {K} draws of (t, noise) per sample, t from the painter's training distribution")
    for path in PATHS:
        for alpha in [0.0, 0.1, 0.25]:
            num = den = 0.0
            band_sig = torch.zeros(4)
            for i, s in enumerate(val[:4]):
                torch.manual_seed(7000 + i)
                x, end = blend(LV[i], path, alpha)
                d = end - x
                ms, ts = [], []
                for _ in range(K):
                    lvl = draw_levels(s.images.shape[0])
                    xr = x.clone().requires_grad_(True)
                    with torch.no_grad():
                        st = renoise(s, lvl, torch.randn_like(s.images))
                    loss = painter_loss(st, model.run_painter(st, xr))
                    r = -torch.autograd.grad(loss.sum(), xr)[0]
                    _, _, info = request_stats(r, d)
                    ms.append((r * d).sum(-1) / d.norm(dim=-1).clamp(min=1e-8))
                    ts.append(lvl)
                m = torch.stack(ms)[:, info]  # (K, informative tokens)
                num += m.var(0).sum().item()
                den += (m.mean(0).pow(2) - m.var(0) / K).sum().item()
                tc = torch.stack(ts)[:, :, None].expand(-1, -1, info.shape[1])[:, info]
                for bi, (lo, hi) in enumerate([(0, 10), (10, 30), (30, 60), (60, 100)]):
                    band_sig[bi] += (m * ((tc >= lo) & (tc < hi))).sum().item()
            print(f"  {path:8s} alpha {alpha:.2f} | B_margin {num / max(den, 1e-30):8.1f} | signal share "
                  "t0-10/10-30/30-60/60-100 " + "/".join(f"{v:.2f}" for v in (band_sig / band_sig.sum()).tolist()),
                  flush=True)
    sys.exit(0)

blinded = bool(model.train_cfg.force_unconditional_painter)
with torch.no_grad():
    pe = torch.cat([painter_loss(s, unsteered(s, blinded)) for s in val])
    pc = torch.cat([painter_loss(s, unsteered(s, False)) for s in val])
tt = torch.cat([level(s.timesteps) for s in val])


def band_masks():
    return [("all", torch.ones_like(tt, dtype=torch.bool))] + [
        (f"t{lo}-{hi}", (tt >= lo) & (tt < hi)) for lo, hi in BANDS]


print(f"unsteered painter loss ({'blinded' if blinded else 'conditional'}): "
      + "  ".join(f"{label} {pe[m].mean():.5f}" for label, m in band_masks()))
print("conditional painter (unblinded, unsteered): loss "
      + "  ".join(f"{label} {pc[m].mean():.5f}" for label, m in band_masks())
      + " | as a reduction: " + "  ".join(f"{label} {1 - pc[m].sum() / pe[m].sum():+.3f}" for label, m in band_masks()))
RED, POOL = {}, {}
for path in PATHS:
    print(f"path {path}")
    for alpha in ALPHAS:
        se, cos, mg, pts, cell_t = [], [], [], [], []
        for i, s in enumerate(val):
            torch.manual_seed(2000 + i)
            x, end = blend(LV[i], path, alpha)
            x = x.clone().requires_grad_(True)
            loss = painter_loss(s, model.run_painter(s, x))
            r = -torch.autograd.grad(loss.sum(), x)[0]
            c, m, info = request_stats(r, end - x.detach())
            se.append(loss.detach())
            cos.append(c)
            mg.append(m)
            pts.append((m > 0).float())
            cell_t.append(level(s.timesteps)[:, None].expand_as(info)[info])
        se, cos, mg, pts, cell_t = (torch.cat(v) for v in (se, cos, mg, pts, cell_t))
        rows = []
        for label, msk in band_masks():
            lo, hi = (0, 101) if label == "all" else BANDS[[f"t{a}-{b}" for a, b in BANDS].index(label)]
            mc = (cell_t >= lo) & (cell_t < hi)
            rows.append(f"{label}: red {1 - se[msk].sum() / pe[msk].sum():+.3f} cos {cos[msk].mean():+.3f}"
                        f" point {pts[mc].mean():.2f} SNR {mg[mc].mean() / mg[mc].std():.2f}")
        RED[path, alpha] = (1 - se.sum() / pe.sum()).item()
        print(f"  alpha {alpha:.2f} | " + " | ".join(rows), flush=True)
    for alpha in [0.0, 0.1, 0.25]:
        cos, pts = [], []
        for i, s in enumerate(val[:N_POOL]):
            torch.manual_seed(5000 + i)
            x, end = blend(LV[i], path, alpha)
            x = x.clone().requires_grad_(True)
            noise = torch.randn_like(s.images)
            gx = 0
            for lvl in T_GRID:
                with torch.no_grad():
                    st = renoise(s, torch.full((s.images.shape[0],), float(lvl), device=dev), noise)
                gx = gx + torch.autograd.grad(painter_loss(st, model.run_painter(st, x)).sum(), x)[0]
            c, m, _ = request_stats(-gx, end - x.detach())
            cos.append(c)
            pts.append((m > 0).float())
        POOL[path, alpha] = torch.cat(pts).mean().item()
        print(f"  pooled alpha {alpha:.2f} | point {POOL[path, alpha]:.2f} cos {torch.cat(cos).mean():+.3f}",
              flush=True)
print(f"SUMMARY {name}: sat {RED['full', 0.1] / RED['full', 1.0]:.2f} (red full a0 {RED['full', 0.0]:+.3f} "
      f"a0.1 {RED['full', 0.1]:+.3f} a1 {RED['full', 1.0]:+.3f}; anon {RED['where', 1.0]:+.3f}, "
      f"conditional painter {1 - pc.sum() / pe.sum():+.3f}) | pooled point a0/a0.1/a0.25 "
      + " ".join(f"{p} " + "/".join(f"{POOL[p, a]:.2f}" for a in [0.0, 0.1, 0.25]) for p in PATHS), flush=True)

if BNOISE > 0:
    REG["on"] = False
    rows = []
    for alpha in [0.0, 0.1, 0.25, 1.0]:
        torch.manual_seed(4000)
        with torch.no_grad():
            se = torch.cat([painter_loss(s, model.run_painter(s, blend(LV[i], "full", alpha)[0]))
                            for i, s in enumerate(val)])
        rows.append(f"a{alpha:.2f}: " + " ".join(f"{label} {1 - se[m].sum() / pe[m].sum():+.3f}"
                                                  for label, m in band_masks()))
    print("deploy full path (bridge noise off) | " + " | ".join(rows), flush=True)
