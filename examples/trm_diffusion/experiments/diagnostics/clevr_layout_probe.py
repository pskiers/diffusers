# ABOUTME: Layout information in a CLEVR thinker's output tokens (held-out LDA), per noise level, against an x_t baseline.
# ABOUTME: The CLEVR analogue of digit_lda.py: an early learn/fail readout of thinker-stage checkpoints.
"""How much of the scene's layout does the thinker's output carry where the translator reads it?

Per val scene and fixed noise level (DDPM index scale: 0 clean .. 100 pure noise; flow t = 1 - level / 100), the
thinker runs its n_sup reasoning steps on x_t (one noise image per scene, shared by all levels), and its output at
the translator's image tokens (logits[:, slice_start : slice_start + grid²]) is probed against labels from the
scene's centroid mask average-pooled to the same grid (make_mask_from_scene's channels):
  occ     the token is occupied (presence > 0.5); balanced accuracy (chance 0.5)
  color   at occupied tokens, the argmax of the 8 color channels; accuracy (chance: majority class, printed)
  shape   at occupied tokens, the argmax of the 3 shape channels; the same
Probe: equal-variance LDA (digit_lda.py's), standardized features, ridge 1e-3 x mean variance, fit on the first
half of the scenes, scored on the second half. Baseline row "x_t": the same probe on the token's own x_t patch
(patch² x channels pixels): what is linearly readable from the noisy image without any thinking. The thinker adds
layout information where its accuracy exceeds that baseline, most clearly at levels 95 / 80.

    [N_SCENES=512] [LEVELS=95,80,60,30] python experiments/diagnostics/clevr_layout_probe.py <overrides.txt> \
        <painter_ckpt> <name> <ckpt>...

<overrides.txt>: the run's thinker-stage overrides (as for saturation_screen.py). A <ckpt> of "init" probes the
freshly initialized thinker. Smoke: N_SCENES=32 LEVELS=95.
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

from datasets.data_sample import collate_data_samples
from factory import build_model
from models.utility_models import strip_compiled_prefix

ov_file, painter_ckpt, name = sys.argv[1:4]
ckpts = sys.argv[4:]
dev = torch.device("cuda")
N_SCENES = int(os.environ.get("N_SCENES", 512))
LEVELS = [float(v) for v in os.environ.get("LEVELS", "95,80,60,30").split(",")]
B = 32

ov = [line.strip() for line in open(ov_file)
      if line.strip() and not line.startswith(("RUN=", "painter.checkpoint=", "eval"))]
ov += [f"painter.checkpoint={painter_ckpt}", "run.wandb_project=null", "data.include_centroid_mask=true",
       "data.centroid_mask_presence_only=false"]
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    cfg = compose(config_name="config", overrides=ov)
cfg.eval_callbacks = None
model = build_model(cfg, instantiate(cfg.diffusion)).to(dev)
tr = model.thinker_painter_translator
G, S0 = tr._grid, getattr(tr, "slice_start", 0)
FLOW = getattr(model.painter, "objective", "ddpm") == "flow"
assert FLOW, "fixed noise levels are implemented for flow painters"
n_sup = model.thinker.n_sup

val_ds = instantiate(cfg.data.val_dataset)
random.seed(1234)  # reduced-relation variant and edge-direction coins per val scene
np.random.seed(1234)
batches = [collate_data_samples([val_ds[i] for i in range(k * B, (k + 1) * B)]) for k in range(N_SCENES // B)]


def at_level(s, lv, noise):
    p = model.painter
    t = torch.full((s.images.shape[0],), 1 - lv / 100, device=dev)
    t4 = t[:, None, None, None]
    x_t = s.images * t4 + noise * p.noise_scale * (1 - t4)
    st = dataclasses.replace(s, x_noisy=x_t, timesteps=t, painter_hidden=None)
    return p.fill_runtime_fields(st) if hasattr(p, "fill_runtime_fields") else st


def labels(s):
    m = F.adaptive_avg_pool2d(s.spatial_conditions.float(), G).flatten(2).transpose(1, 2)  # (B, G², 16)
    return m[..., 15] > 0.5, m[..., :8].argmax(-1), m[..., 8:11].argmax(-1)


def patches(x_t):
    p = x_t.shape[-1] // G
    return F.unfold(x_t.float(), kernel_size=p, stride=p).transpose(1, 2)  # (B, G², C p²)


@torch.no_grad()
def collect():
    out = {lv: {"thinker": [], "x_t": []} for lv in LEVELS}
    lab = {"occ": [], "color": [], "shape": []}
    for i, b in enumerate(batches):
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(500 + i)
            [d] = model.prep_mb_data([b], dev)
            noise = torch.randn_like(d["sample"].images)
        s = d["sample"]
        occ, col, sh = labels(s)
        lab["occ"].append(occ)
        lab["color"].append(col)
        lab["shape"].append(sh)
        for lv in LEVELS:
            st = at_level(s, lv, noise)
            z_H, z_L = d["z_H"], d["z_L"]
            enc = model._encode_condition(st)
            for _ in range(n_sup):
                logits, z_H, z_L = model.thinker.reasoning_step(enc, z_H, z_L, st.puzzle_id, timesteps=st.timesteps)
            tok = logits.float()[:, S0:S0 + G * G]
            assert tok.shape[1] == G * G, (logits.shape, S0, G)
            out[lv]["thinker"].append(tok)
            out[lv]["x_t"].append(patches(st.x_noisy))
            if i == 0 and lv == LEVELS[0]:
                print(f"  first batch: logits {tuple(logits.shape)} {logits.dtype}, probed tokens {tuple(tok.shape)}, "
                      f"x_t patches {tuple(out[lv]['x_t'][0].shape)}, occupied share {occ.float().mean():.3f}",
                      flush=True)
    lab = {k: torch.cat(v) for k, v in lab.items()}  # (N, G²)
    return {lv: {k: torch.cat(v) for k, v in o.items()} for lv, o in out.items()}, lab


def lda_acc(x, y, train, balanced=False):
    """Held-out accuracy of equal-variance LDA (fit on rows `train`, scored on the rest)."""
    x = x.double()
    x = (x - x[train].mean(0)) / (x[train].std(0) + 1e-8)
    classes = y[train].unique()
    xa, ya = x[train], y[train]
    mu = torch.stack([xa[ya == c].mean(0) for c in classes])
    xc = xa - mu[torch.searchsorted(classes, ya)]
    sw = xc.T @ xc / len(xa)
    prec = torch.linalg.inv(sw + 1e-3 * torch.trace(sw) / sw.shape[0] * torch.eye(x.shape[1], dtype=x.dtype, device=x.device))
    logp = torch.log(torch.stack([(ya == c).double().mean() for c in classes]))
    xt, yt = x[~train], y[~train]
    pred = classes[(xt @ prec @ mu.T - 0.5 * ((mu @ prec) * mu).sum(1) + logp).argmax(1)]
    if balanced:
        return torch.stack([(pred[yt == c] == c).double().mean() for c in classes]).mean().item()
    return (pred == yt).double().mean().item()


def probe(feats, lab):
    n = lab["occ"].shape[0]
    train_scene = torch.arange(n, device=dev) < n // 2
    tr_tok = train_scene[:, None].expand_as(lab["occ"]).flatten()
    occ = lab["occ"].flatten()
    f = feats.flatten(0, 1)
    res = {"occ": lda_acc(f, occ.long(), tr_tok, balanced=True)}
    for k in ("color", "shape"):
        y = lab[k].flatten()[occ]
        res[k] = lda_acc(f[occ], y, tr_tok[occ])
    return res


print(f"== {name} (grid {G}, slice_start {S0}, n_sup {n_sup}, {len(batches) * B} val scenes, levels {LEVELS})")
baseline_done = False
for ck in ckpts:
    if ck != "init":
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)  # own checkpoints, as digit_lda.py
        sd = strip_compiled_prefix(ckpt["model_state"])
        own = model.state_dict()
        sd = {k: v for k, v in sd.items() if not (k in own and own[k].shape != v.shape)}
        res = model.load_state_dict(sd, strict=False)
        print(f"load {ck}: missing {len(res.missing_keys)} unexpected {len(res.unexpected_keys)}", flush=True)
        step = ckpt.get("step")
    else:
        step = 0
    model.to(dev).eval()
    feats, lab = collect()
    if not baseline_done:
        occ = lab["occ"].flatten()
        maj = {k: torch.bincount(lab[k].flatten()[occ]).max().item() / occ.sum().item() for k in ("color", "shape")}
        print(f"  chance: occ 0.500, color {maj['color']:.3f}, shape {maj['shape']:.3f} (majority class)")
        for lv in LEVELS:
            r = probe(feats[lv]["x_t"], lab)
            print(f"  x_t     level {lv:4.0f} | occ {r['occ']:.3f} color {r['color']:.3f} shape {r['shape']:.3f}",
                  flush=True)
        baseline_done = True
    for lv in LEVELS:
        f = feats[lv]["thinker"]
        r = probe(f, lab)
        # control: each scene's mean token at every position carries scene composition (which colors / shapes /
        # how many objects) but no location; layout is what the per-token probe adds beyond it
        rm = probe(f.mean(1, keepdim=True).expand_as(f), lab)
        print(f"  step {step} level {lv:4.0f} | occ {r['occ']:.3f} color {r['color']:.3f} shape {r['shape']:.3f}"
              f" | scene-mean occ {rm['occ']:.3f} color {rm['color']:.3f} shape {rm['shape']:.3f}", flush=True)
