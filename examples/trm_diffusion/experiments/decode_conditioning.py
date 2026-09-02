"""
experiments/decode_conditioning.py — what maze facts are LINEARLY encoded in the
TRM's conditioning on the 12x12 grid.

For each of the 144 grid tokens we take its post-bridge conditioning vector (11-d,
Tap2) [optionally the 512-d z_H] and ask whether a LINEAR probe recovers a known
maze fact for that token: on_path, is_wall, is_marker (binary), path_pos and
dist_to_goal/dist_to_start (continuous). Features are captured TEACHER-FORCED at a
high noise level t so the answer isn't leaking through x_noisy — decodability then
means the TRM reasoned it.

Guardrails (else the number lies): probes are strictly linear, evaluated on
held-out mazes, and reported against two floors — majority-class and an
UNTRAINED (random-init) pipeline's features. Signal = probe - floor.

Failure attribution (optional, +decode_failure=true): on solved vs failed puzzles,
decode against GT path AND the model's OWN painted path -> localises whether a
failure is the thinker planning wrong or the painter rendering wrong.

Usage:
    python experiments/decode_conditioning.py \
      experiment=amaze_thinker_v2_controlnet \
      +checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt \
      +task=maze +trajectory_combo=square_n7 \
      [+decode_n_mazes=200] [+decode_frac=0.9] [+decode_use_zH=false] \
      [+decode_failure=true] [+wandb_run_id=<id>]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import instantiate
from omegaconf import DictConfig

from experiments.conditioning_lib import (
    TRM_ROOT,
    capture_teacher_forced,
    decode_cell_ids,
    load_model,
    maze_token_labels,
    select_good_bad,
    to_hwc_uint8,
    token_cell_grid,
    wandb_attach,
)
from experiments.sample_amaze_metrics import _build_amaze_dataset, _require_test_parquet
from experiments.sample_amaze_trajectory import _resolve_combo

logger = get_logger(__name__, log_level="INFO")

BINARY = ["on_path", "is_wall", "is_marker"]
CONTINUOUS = ["path_pos", "dist_to_goal", "dist_to_start"]


# ── linear probes (torch, no sklearn dependency) ──────────────────────────────


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allp = np.concatenate([pos, neg])
    order = np.argsort(allp, kind="mergesort")
    ranks = np.empty(allp.size, dtype=float)
    ranks[order] = np.arange(1, allp.size + 1)
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def fit_logistic(Xtr, ytr, Xte, yte, steps: int = 400, lr: float = 0.05, wd: float = 1e-3) -> dict:
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr_s = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xte_s = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(Xtr_s.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xtr_s @ w + b, ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(Xte_s @ w + b).numpy()
    acc = float(((p >= 0.5).astype(float) == yte).mean())
    return {"acc": acc, "auc": _auc(yte, p), "w": w.detach().numpy(), "b": float(b.item()),
            "mu": mu, "sd": sd}


def proba(probe: dict, X: np.ndarray) -> np.ndarray:
    Xs = (X - probe["mu"]) / probe["sd"]
    return 1.0 / (1.0 + np.exp(-(Xs @ probe["w"] + probe["b"])))


def fit_ridge(Xtr, ytr, Xte, yte, lam: float = 1.0) -> dict:
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    A = np.concatenate([(Xtr - mu) / sd, np.ones((Xtr.shape[0], 1))], axis=1)
    B = np.concatenate([(Xte - mu) / sd, np.ones((Xte.shape[0], 1))], axis=1)
    D = A.shape[1]
    reg = lam * np.eye(D)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    pred = B @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-9
    return {"r2": 1.0 - ss_res / ss_tot}


def majority_floor(ytr, yte) -> dict:
    pred = 1.0 if ytr.mean() >= 0.5 else 0.0
    return {"acc": float((yte == pred).mean()), "auc": 0.5}


# ── feature / label extraction ────────────────────────────────────────────────


@torch.no_grad()
def build_dataset(model, ds, device, idxs, decode_t, use_zH, batch_size=24, seed=0):
    """Per-token features + labels over the given maze indices."""
    feats_s, feats_z = [], []
    labels: dict[str, list] = {}
    for start in range(0, len(idxs), batch_size):
        chunk = idxs[start:start + batch_size]
        puzzles = [ds[i] for i in chunk]
        conditions = model._batch_to_sample(ds.collate_fn(puzzles), device)
        cap = capture_teacher_forced(model, conditions, decode_t, device, seed=seed)
        spatial = cap["spatial"]                                   # (B,11,12,12)
        zH = cap["z_H"] if use_zH else None
        for pi, p in enumerate(puzzles):
            feats_s.append(spatial[pi].reshape(11, 144).transpose(0, 1).numpy())   # (144,11)
            if use_zH and zH is not None:
                feats_z.append(zH[pi].numpy())                                     # (144,512)
            lab = maze_token_labels(p.metadata, grid=12)
            for name, arr in lab.items():
                if name == "token_cells":
                    continue
                labels.setdefault(name, []).append(np.asarray(arr).reshape(-1))
    X_s = np.concatenate(feats_s, axis=0) if feats_s else np.zeros((0, 11))
    X_z = np.concatenate(feats_z, axis=0) if feats_z else None
    y = {k: np.concatenate(v, axis=0) for k, v in labels.items()}
    return X_s, X_z, y


def _eval_targets(Xtr, ytr, Xte, yte, Xtr_u, Xte_u):
    """Run every available target; return rows + the fitted trained on_path probe."""
    rows, onpath_probe = [], None
    for tgt in BINARY:
        if tgt not in ytr:
            continue
        m = ~np.isnan(ytr[tgt])
        mt = ~np.isnan(yte[tgt])
        if m.sum() < 10 or mt.sum() < 10 or len(np.unique(ytr[tgt][m])) < 2:
            continue
        tr = fit_logistic(Xtr[m], ytr[tgt][m], Xte[mt], yte[tgt][mt])
        un = fit_logistic(Xtr_u[m], ytr[tgt][m], Xte_u[mt], yte[tgt][mt])
        mf = majority_floor(ytr[tgt][m], yte[tgt][mt])
        rows.append({"target": tgt, "kind": "binary", "acc": tr["acc"], "auc": tr["auc"],
                     "acc_untrained": un["acc"], "auc_untrained": un["auc"], "acc_majority": mf["acc"]})
        if tgt == "on_path":
            onpath_probe = tr
    for tgt in CONTINUOUS:
        if tgt not in ytr:
            continue
        m = ~np.isnan(ytr[tgt]) & (ytr[tgt] >= 0)
        mt = ~np.isnan(yte[tgt]) & (yte[tgt] >= 0)
        if m.sum() < 10 or mt.sum() < 10:
            continue
        tr = fit_ridge(Xtr[m], ytr[tgt][m], Xte[mt], yte[tgt][mt])
        un = fit_ridge(Xtr_u[m], ytr[tgt][m], Xte_u[mt], yte[tgt][mt])
        rows.append({"target": tgt, "kind": "continuous", "r2": tr["r2"], "r2_untrained": un["r2"]})
    return rows, onpath_probe


# ── failure attribution ───────────────────────────────────────────────────────


def _blue_mask(img_uint8: np.ndarray) -> np.ndarray:
    r, g, b = img_uint8[..., 0].astype(int), img_uint8[..., 1].astype(int), img_uint8[..., 2].astype(int)
    return (b > 120) & (r < 110) & (g < 130)


@torch.no_grad()
def failure_attribution(model, ds, device, good, bad, decode_t, probe, seed=0) -> list[dict]:
    """For solved/failed puzzles, decode on_path with the trained probe and compare
    to GT vs the model's own painted path (per-token)."""
    pipeline = model.sampling_pipeline
    out = []
    for group, idxs in (("solved", good), ("failed", bad)):
        if not idxs:
            continue
        puzzles = [ds[i] for i in idxs]
        conditions = model._batch_to_sample(ds.collate_fn(puzzles), device)
        gen = pipeline.sample_one_batch(model, conditions, device,
                                        generator=torch.Generator(device=device).manual_seed(seed))
        decoded = model.decode_for_eval(gen).cpu()
        cap = capture_teacher_forced(model, conditions, decode_t, device, seed=seed)
        spatial = cap["spatial"]
        acc_gt, acc_model = [], []
        for pi, p in enumerate(puzzles):
            lab = maze_token_labels(p.metadata, grid=12, with_dist=False)
            gt = lab["on_path"].reshape(-1)
            cell_ids = decode_cell_ids(p.metadata.get("cell_map"))
            gen_arr = to_hwc_uint8(decoded[pi])
            if gen_arr.shape[:2] != cell_ids.shape:  # blue mask must match cell_ids resolution
                gen_arr = np.asarray(Image.fromarray(gen_arr).resize(
                    (cell_ids.shape[1], cell_ids.shape[0]), Image.BILINEAR))
            blue = _blue_mask(gen_arr)
            pred_cells = set(int(c) for c in cell_ids[blue].tolist()) - {0}
            tok = token_cell_grid(cell_ids, grid=12).reshape(-1)
            model_on = np.array([1.0 if int(c) in pred_cells else 0.0 for c in tok])
            X = spatial[pi].reshape(11, 144).transpose(0, 1).numpy()
            pred = (proba(probe, X) >= 0.5).astype(float)
            acc_gt.append(float((pred == gt).mean()))
            acc_model.append(float((pred == model_on).mean()))
        out.append({"group": group, "n": len(puzzles),
                    "probe_vs_gt": float(np.mean(acc_gt)),
                    "probe_vs_model_path": float(np.mean(acc_model))})
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    task = cfg.get("task", "maze")
    if checkpoint is None or task != "maze":
        print("ERROR: needs +checkpoint=<thinker.pt> +task=maze", file=sys.stderr)
        sys.exit(1)

    n_mazes = int(cfg.get("decode_n_mazes", 200))
    decode_frac = float(cfg.get("decode_frac", 0.9))
    use_zH = str(cfg.get("decode_use_zH", False)).lower() in ("1", "true", "yes")
    do_failure = str(cfg.get("decode_failure", True)).lower() in ("1", "true", "yes")
    seed = int(cfg.get("seed", 0))

    data_root = Path(cfg.get("data_root", str(TRM_ROOT / "data" / "amaze")))
    parquet, combo_label = _resolve_combo(task, data_root, cfg.get("trajectory_combo", None))
    _require_test_parquet(parquet, task)

    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    device = accelerator.device

    model = load_model(cfg, checkpoint)
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()
    if getattr(model, "thinker_painter_translator", None) is None:
        print("ERROR: decoding needs a TRM painter-thinker (no translator).", file=sys.stderr)
        sys.exit(1)

    ds = _build_amaze_dataset(cfg, str(parquet))
    T = model.scheduler.config.num_train_timesteps
    decode_t = int(decode_frac * (T - 1))
    n = min(n_mazes, len(ds))
    idxs = list(range(n))
    split = max(1, int(0.8 * n))
    tr_idx, te_idx = idxs[:split], idxs[split:] or idxs[-max(1, n // 5):]
    logger.info(f"[{combo_label}] decoding at t={decode_t} over {n} mazes "
                f"(train {len(tr_idx)} / test {len(te_idx)} mazes, use_zH={use_zH})...")

    Xtr_s, Xtr_z, ytr = build_dataset(model, ds, device, tr_idx, decode_t, use_zH, seed=seed)
    Xte_s, Xte_z, yte = build_dataset(model, ds, device, te_idx, decode_t, use_zH, seed=seed)

    # Untrained (random-init) pipeline floor — same features, no checkpoint loaded.
    scheduler = instantiate(cfg.diffusion)
    from factory import build_model
    untrained = accelerator.unwrap_model(accelerator.prepare(build_model(cfg, scheduler)))
    untrained.eval()
    Xtr_u_s, _z, _y = build_dataset(untrained, ds, device, tr_idx, decode_t, False, seed=seed)
    Xte_u_s, _z2, _y2 = build_dataset(untrained, ds, device, te_idx, decode_t, False, seed=seed)
    del untrained

    feature_sets = [("spatial11", Xtr_s, Xte_s, Xtr_u_s, Xte_u_s)]
    if use_zH and Xtr_z is not None:
        # untrained z_H floor
        untrained2 = accelerator.unwrap_model(accelerator.prepare(build_model(cfg, scheduler)))
        untrained2.eval()
        _s, Xtr_u_z, _yy = build_dataset(untrained2, ds, device, tr_idx, decode_t, True, seed=seed)
        _s2, Xte_u_z, _yy2 = build_dataset(untrained2, ds, device, te_idx, decode_t, True, seed=seed)
        del untrained2
        feature_sets.append(("z_H", Xtr_z, Xte_z, Xtr_u_z, Xte_u_z))

    all_rows, onpath_probe = [], None
    for fname, Xtr, Xte, Xtr_u, Xte_u in feature_sets:
        rows, probe = _eval_targets(Xtr, ytr, Xte, yte, Xtr_u, Xte_u)
        for r in rows:
            r["features"] = fname
        all_rows.extend(rows)
        if fname == "spatial11" and probe is not None:
            onpath_probe = probe

    _print_table(combo_label, decode_t, all_rows)

    fail_rows = []
    if do_failure and onpath_probe is not None:
        try:
            good, bad = select_good_bad(model, ds, device, int(cfg.get("n_each", 10)), seed=seed)
            fail_rows = failure_attribution(model, ds, device, good, bad, decode_t, onpath_probe, seed=seed)
            _print_failure(fail_rows)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failure attribution skipped ({e!r}).")

    if accelerator.is_main_process:
        _log_wandb(cfg, str(checkpoint), combo_label, all_rows, fail_rows)


def _print_table(label, decode_t, rows) -> None:
    print("\n" + "=" * 92)
    print(f"Conditioning decoding — {label}  (teacher-forced t={decode_t})")
    print("=" * 92)
    print(f"{'features':>10} {'target':>14} {'kind':>10} {'score':>8} {'untrained':>10} {'majority':>9}")
    for r in rows:
        if r["kind"] == "binary":
            print(f"{r['features']:>10} {r['target']:>14} {'AUC':>10} {r['auc']:>8.3f} "
                  f"{r['auc_untrained']:>10.3f} {'-':>9}   (acc {r['acc']:.3f} / maj {r['acc_majority']:.3f})")
        else:
            print(f"{r['features']:>10} {r['target']:>14} {'R2':>10} {r['r2']:>8.3f} "
                  f"{r['r2_untrained']:>10.3f} {'-':>9}")
    print("=" * 92)


def _print_failure(rows) -> None:
    print("\n" + "-" * 70)
    print("Failure attribution — on_path probe accuracy (trained on GT):")
    print(f"{'group':>8} {'n':>4} {'vs GT path':>12} {'vs model path':>14}")
    for r in rows:
        print(f"{r['group']:>8} {r['n']:>4} {r['probe_vs_gt']:>12.3f} {r['probe_vs_model_path']:>14.3f}")
    print("-" * 70)


def _log_wandb(cfg, checkpoint, combo_label, rows, fail_rows) -> None:
    run = wandb_attach(cfg, checkpoint, logger)
    if run is None:
        return
    import wandb

    try:
        cols = ["features", "target", "kind", "score", "untrained", "acc", "majority"]
        table = wandb.Table(columns=cols)
        for r in rows:
            if r["kind"] == "binary":
                table.add_data(r["features"], r["target"], "AUC", r["auc"], r["auc_untrained"],
                               r["acc"], r["acc_majority"])
            else:
                table.add_data(r["features"], r["target"], "R2", r["r2"], r["r2_untrained"], None, None)
        payload = {f"conditioning/decode/{combo_label}/table": table}
        if fail_rows:
            ft = wandb.Table(columns=["group", "n", "probe_vs_gt", "probe_vs_model_path"])
            for r in fail_rows:
                ft.add_data(r["group"], r["n"], r["probe_vs_gt"], r["probe_vs_model_path"])
            payload[f"conditioning/decode/{combo_label}/failure_attribution"] = ft
        run.log(payload)
        logger.info(f"wandb: logged decoding results ({combo_label}).")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"wandb: logging failed ({e!r}).")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
