"""
eval/topodiff_eval.py — Evaluation for the TopoDiff dataset
(datasets/topodiff_dataset.py): topology-optimization structures scored by
the paper's own four metrics — CE (compliance error), VFE (volume fraction
error), LV (load violation), FM (floating material).

This is a thin, *direct-port* wrapper around eval/topodiff_analysis.py — a
verbatim vendored copy of the paper's own topodiff/topodiff_analysis.py
(see that module's header for exact provenance/commit/license notes). It
does not reimplement any metric math; it only adapts calling convention:

  - Per-sample scratch directories (tempfile.mkdtemp, cleaned up after each
    FEA solve) instead of the vendored fem_compliance_i's hardcoded
    "./fem_files/" — safe to call from a training loop / across workers.
    Achieved by calling create_files/mysolidspy directly (both already take
    a directory argument) rather than fem_compliance_i, which is never
    called from here (see topodiff_analysis.py's header for why that's
    fine — its recompute-loop bug never fires with data we control).
  - Tensor <-> the vendored code's (H, W) uint8, material<127/void>=127
    array convention (see _to_topo_array / topo_to_tab in the vendored
    module).
  - Metrics are computed per-instance from a caller-supplied `summaries`
    list (BC_conf/load_nodes/x_loads/y_loads/VF/load_coord — exactly the
    per-sample dict shape already stored in TopoDiff's own
    training_data_summary.npy / test_data_level_*_summary.npy, unchanged)
    rather than from the paper's own topodiff_analysis()/order_list()
    bookkeeping, which existed only to reconstruct their sampler's on-disk
    output order — irrelevant here since the caller (models/eval_callbacks.py's
    TopodiffEvalCallback) already knows exactly which puzzle_id/summary
    goes with which generated image.

solidspy version pin: the vendored mysolidspy() calls ass.DME(nodes, elements)
and ass.assembler(elements, mats, nodes, neq, DME) with the *unsliced* 5-column
nodes array and expects a single-value return from assembler — the calling
convention of solidspy<=1.0.16 (the latest release that existed when
topodiff's code was written, Dec 2022). The current PyPI "latest"
(1.1.0.post1, released Nov 2023) changed both DME's expected `cons` shape and
assembler's return arity, breaking this exact call. This project therefore
pins solidspy==1.0.16 (see examples/trm_diffusion/requirements or the
environment notes) — a compatibility requirement of the vendored paper code
itself, not something introduced here. That old release also predates
numpy's removal of the `np.int`/`np.float` aliases (deprecated in numpy 1.20,
removed later) and calls them directly in solidspy.preprocesor.readin(); the
small shim below restores those aliases process-wide before any FEA call
(mirroring the deprecation notice's own "use `int`/`float` instead" fix,
applied as a shim rather than editing the installed third-party package).
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Optional

import numpy as np
import torch

# solidspy==1.0.16 (see module docstring) calls the numpy 1.20-deprecated,
# now-removed np.int/np.float aliases inside its own readin(). Shim once,
# before importing anything that transitively imports solidspy.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import eval.topodiff_analysis as ta

# ── Tensor <-> vendored-code array conventions ────────────────────────────────


def _to_topo_array(image: torch.Tensor) -> np.ndarray:
    """(1, H, W) float in [-1, 1] -> (H, W) uint8 in {0..255}, material dark
    (near 0), void bright (near 255) — matches gt_topo_*.png's own convention
    (see datasets/topodiff_dataset.py) and topo_to_tab's `< 127` threshold."""
    arr = image.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
    return np.round((arr + 1.0) * 127.5).astype(np.uint8)


@torch.no_grad()
def evaluate_topodiff(
    images: torch.Tensor,  # (B, 1, H, W) float in [-1, 1], generated
    summaries: list[dict],  # length B; each: BC_conf, load_nodes, x_loads, y_loads, VF, load_coord
    compliance_opt: Optional[np.ndarray] = None,  # (B,) reference SIMP compliance; None => CE not computed
) -> dict:
    """Score a batch of generated topologies against the paper's 4 metrics.

    Returns dict with keys:
      per_sample_compliance — (B,) float64; raw FEA compliance (sum(E_nodes*S_nodes))
                               of each generated topology, via the vendored solver.
      per_sample_ce         — (B,) float64; compliance_opt is not None:
                               per_sample_compliance / compliance_opt - 1 (the
                               paper's own CE definition, arXiv 2208.09591).
                               NaN everywhere if compliance_opt is None.
      per_sample_vfe        — (B,) float64; |VF(generated) - VF(target)| / VF(target).
      per_sample_lv         — (B,) bool; True = no material at the load point (violation).
      per_sample_fm         — (B,) bool; True = floating (disconnected) material present.
      CE, VFE               — mean of the finite entries of per_sample_ce / per_sample_vfe
                               (NaN-safe: CE is NaN if compliance_opt was None).
      LV, FM                — mean (proportion) of per_sample_lv / per_sample_fm.
    Matches the paper's own aggregate names/definitions (print_results).
    """
    B = images.shape[0]
    assert len(summaries) == B, f"summaries length {len(summaries)} != batch size {B}"

    topo_arrays = [_to_topo_array(images[i]) for i in range(B)]

    per_sample_compliance = np.empty(B, dtype=np.float64)
    per_sample_vfe = np.empty(B, dtype=np.float64)
    per_sample_lv = np.empty(B, dtype=bool)
    per_sample_fm = np.empty(B, dtype=bool)

    for i in range(B):
        topo = topo_arrays[i]
        s = summaries[i]

        tmp = tempfile.mkdtemp(prefix="topodiff_fem_")
        try:
            folder = tmp + "/"
            ta.create_files(topo, s["BC_conf"], s["load_nodes"], s["x_loads"], s["y_loads"], folder)
            _, E_nodes, S_nodes = ta.mysolidspy(folder)
            per_sample_compliance[i] = np.sum(np.multiply(E_nodes, S_nodes))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        per_sample_vfe[i] = abs(ta.compute_vf(topo) - s["VF"]) / s["VF"]
        per_sample_fm[i] = ta.check_floating_material(topo)
        per_sample_lv[i] = ta.check_load(s["load_coord"][0], topo)

    if compliance_opt is not None:
        compliance_opt = np.asarray(compliance_opt, dtype=np.float64)
        per_sample_ce = per_sample_compliance / compliance_opt - 1.0
    else:
        per_sample_ce = np.full(B, np.nan, dtype=np.float64)

    return {
        "per_sample_compliance": per_sample_compliance,
        "per_sample_ce": per_sample_ce,
        "per_sample_vfe": per_sample_vfe,
        "per_sample_lv": per_sample_lv,
        "per_sample_fm": per_sample_fm,
        "CE": float(np.nanmean(per_sample_ce)),
        "VFE": float(np.mean(per_sample_vfe)),
        "LV": float(np.mean(per_sample_lv)),
        "FM": float(np.mean(per_sample_fm)),
    }


def make_topodiff_panel_image(
    condition: torch.Tensor,  # (5, H, W) float — VF, von Mises, SED, load_x, load_y
    generated: torch.Tensor,  # (1, H, W) float in [-1, 1]
    reference: Optional[torch.Tensor] = None,  # (1, H, W) float in [-1, 1], or None (no GT image available)
) -> np.ndarray:
    """load-magnitude condition | generated | reference (if available), each
    mapped to an [0,255] grayscale panel and concatenated horizontally. The
    load-field channels (indices 3, 4 — see datasets/topodiff_dataset.py) are
    the most visually informative single condition channel to display; VF is
    a flat constant and the physical fields are dense heatmaps that don't
    reduce to a clean binary-looking panel the way load magnitude does."""

    def topo_to_uint8(t: torch.Tensor) -> np.ndarray:
        arr = _to_topo_array(t)
        return np.stack([arr] * 3, axis=-1)

    load_mag = torch.sqrt(condition[3] ** 2 + condition[4] ** 2).cpu().numpy()
    load_mag = load_mag / (load_mag.max() + 1e-8)
    cond_img = np.stack([(load_mag * 255).astype(np.uint8)] * 3, axis=-1)
    cond_img = 255 - cond_img  # white background, dark = load location, matching topology convention

    sep = np.full((cond_img.shape[0], 4, 3), 128, dtype=np.uint8)
    panels = [cond_img, sep, topo_to_uint8(generated)]
    if reference is not None:
        panels += [sep, topo_to_uint8(reference)]
    return np.concatenate(panels, axis=1)
