"""Stage 1 (qualitative): show what each detector sees on real Bagel generations.

Per board, 4 panels:
  1. generated attempt (native resolution)
  2. ground-truth solution
  3. detected pixels: blue = original detector, orange = extra pixels found by the
     faint detector (tint >= TINT), yellow = only found at tint >= LOW_TINT
  4. cell verdicts (faint detector at TINT) over the generated image:
     green = correct path cell, red = extra cell the ORIGINAL detector also flags,
     orange = extra cell only the faint detector flags, yellow = extra cell only at
     LOW_TINT, magenta = missing GT cell, grey = endpoint-tolerance zone (ignored)
Title: pass / exact of (a) original scorer as run (144px), (b) original detector at
native res, (c) faint detector at native res.

Usage: stage1_visual.py [boards_scanned_per_shape] [shape ...]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from faint_metrics import (BAGEL_RUNS, FaintAmazeMetrics, attempt_id, faint_blue_mask,  # noqa: E402
                           load_native, load_shape_index, solid_blue_mask)
from eval.amaze_eval import AmazeMetrics  # noqa: E402

TINT, LOW_TINT = 10, 5
N_SCAN = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SHAPES = sys.argv[2:] or list(BAGEL_RUNS)
OUT = Path(__file__).parent / "stage1"
OUT.mkdir(exist_ok=True)

orig = AmazeMetrics(task="maze")
solid = FaintAmazeMetrics(detector="solid")
faint = FaintAmazeMetrics(detector="faint", tint_min=TINT)
faint_lo = FaintAmazeMetrics(detector="faint", tint_min=LOW_TINT)
to144 = transforms.Compose([transforms.Resize((144, 144), interpolation=transforms.InterpolationMode.BICUBIC,
                                              antialias=True), transforms.ToTensor()])


def paint_cells(cell_ids, groups):
    """RGBA overlay colouring each cell-id group."""
    over = np.zeros(cell_ids.shape + (4,), dtype=np.float32)
    for ids, rgba in groups:
        if ids:
            over[np.isin(cell_ids, list(ids))] = rgba
    return over


def fmt(s):
    return f"pass {s['pass']:.2f} exact {s['exact']:.0f}"


def render(shape, pid, f, md, tag):
    gen = load_native(f)
    s144 = orig._compute_maze_metrics(to144(Image.fromarray(gen)), md)
    a_sol = solid.analyse(gen, md)
    a_f = faint.analyse(gen, md)
    a_lo = faint_lo.analyse(gen, md)
    arr = a_f["gen_arr"]

    # panel 3: pixel masks
    m_solid = solid.path_mask(arr).astype(bool)
    m_f = a_f["mask"].astype(bool)
    m_lo = a_lo["mask"].astype(bool)
    m_raw = faint._morph_open(faint_blue_mask(arr, LOW_TINT)).astype(bool)  # before stroke-length filter
    pix = np.full(arr.shape, 255, np.uint8)
    pix[(arr.min(2) < 80) & ~m_lo] = (170, 170, 170)          # walls, for orientation
    pix[m_raw & ~m_lo] = (200, 0, 200)                          # dropped as too short
    pix[m_lo & ~m_f] = (240, 200, 0)
    pix[m_f & ~m_solid] = (255, 120, 0)
    pix[m_solid] = (0, 70, 200)

    # panel 4: cell verdicts
    extra_solid = a_sol["extra"]
    extra_f = a_f["extra"]
    extra_lo_only = a_lo["extra"] - extra_f
    over = paint_cells(a_f["cell_ids"], [
        (a_f["ignored"], (0.5, 0.5, 0.5, 0.45)),
        (a_f["predicted"] & a_f["gt"], (0.1, 0.8, 0.2, 0.40)),
        (extra_lo_only, (1.0, 0.85, 0.0, 0.60)),
        (extra_f - extra_solid, (1.0, 0.5, 0.0, 0.60)),
        (extra_f & extra_solid, (0.9, 0.0, 0.0, 0.60)),
        (a_f["missing"], (0.9, 0.0, 0.9, 0.60)),
    ])

    fig, ax = plt.subplots(1, 4, figsize=(22, 6.2))
    ax[0].imshow(gen); ax[0].set_title("generated (attempt001)")
    ax[1].imshow(md["sol_img"]); ax[1].set_title("ground truth")
    ax[2].imshow(pix); ax[2].set_title(f"pixels: blue=original, orange=faint tint>={TINT}, yellow=tint>={LOW_TINT} only")
    ax[3].imshow(arr); ax[3].imshow(over)
    ax[3].set_title(f"cells (tint>={TINT}): extra solid={len(extra_f & extra_solid)} "
                    f"faint-only={len(extra_f - extra_solid)} (+{len(extra_lo_only)} @{LOW_TINT})  "
                    f"missing={len(a_f['missing'])}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{shape} n{md['_scale']}  {pid[:8]}   |   original@144px: {fmt(s144)}   |   "
                 f"original@native: {fmt(a_sol['scores'])}   |   faint@native: {fmt(a_f['scores'])}",
                 fontsize=13)
    fig.tight_layout()
    out = OUT / f"{shape}_{tag}_{pid[:8]}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(f"  {out.name}: 144px {fmt(s144)} | native-orig {fmt(a_sol['scores'])} | faint {fmt(a_f['scores'])}"
          f" | extra cells solid={len(extra_solid)} faint={len(extra_f)} low={len(a_lo['extra'])}")


for shape in SHAPES:
    files = sorted(BAGEL_RUNS[shape].glob("*_attempt001.jpg"))
    files = [files[i] for i in np.random.default_rng(1).permutation(len(files))[:N_SCAN]]
    index = load_shape_index(shape, wanted={attempt_id(f) for f in files})
    # classify the scanned boards, then render 2 with faint-only extras + 1 clean
    faint_cases, clean = [], []
    for f in files:
        pid = attempt_id(f)
        s, md = index[pid]
        md["_scale"] = s
        gen = load_native(f)
        e_sol, e_f = solid.analyse(gen, md)["extra"], faint.analyse(gen, md)["extra"]
        (faint_cases if len(e_f - e_sol) > 0 else clean).append((len(e_f - e_sol), pid, f, md))
    print(f"{shape}: scanned {len(files)}, {len(faint_cases)} with faint-only extra cells, {len(clean)} without")
    faint_cases.sort(key=lambda t: t[0])
    picks = []
    if faint_cases:
        picks.append(("faint", faint_cases[len(faint_cases) // 2]))   # median-size ghost
        picks.append(("faint", faint_cases[-1]))                      # largest ghost
    if clean:
        picks.append(("clean", clean[0]))
    for tag, (_, pid, f, md) in picks:
        render(shape, pid, f, md, tag)
