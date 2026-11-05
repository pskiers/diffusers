# ABOUTME: Samples CLEVR scenes from a checkpoint and rescores spatial accuracy split by relation type.
# ABOUTME: Given = edges in the reduced condition (hop 1); implied = GT relations recoverable only by chaining.
"""Which relations does a CLEVR model get wrong: the edges it is given, or the ones implied by chaining them?

The "reduced" condition keeps, per axis (left/right, front/behind), only a transitive reduction of the scene's
relations; the evaluator (eval/evaluate_clevr.py) scores all of the scene's GT relations. Per GT relation
(rel_type, s, t) this script records its hop count: the shortest chain of given edges of that axis that implies it
(1 = given directly, inf = not implied by the given edges). Since geometry is transitive, an implied relation can
only fail if some edge on its chains fails; implied pairs are most of the count, so the headline spatial accuracy
is dominated by them.

Sampling and scoring are the W&B callback's (ClevrMetricsCallback + _score_image) on a seeded subset of val scenes
(3-10 objects), so the totals reproduce its metrics up to the subset. Reported:
  spatial          c_rel / t_rel, the W&B `val/clevr_spatial_acc` (checked: equals the split's sum)
  given / implied  the same over hop-1 / hop >= 2 relations; by hop 2, 3, 4, 5+
  scene_all_rel    share of scenes with every object matched and every GT relation satisfied
  scene_all_given  the same over the given edges only
  by n_objects     spatial / given / implied per object count

    [N_SCENES=1024] [SEED=0] [BATCH=128] [USE_EMA=1] python experiments/diagnostics/clevr_relation_split.py \
        <overrides.txt> <checkpoint.pt> <out_dir>

<overrides.txt>: one Hydra override per line (RUN= and eval* lines are skipped). <checkpoint.pt> is loaded on top:
its EMA weights (USE_EMA=1, default: what train_trm.py's evals use) or its raw model_state (USE_EMA=0: what a
thinker stage's frozen painter loads, painter_base._load_frozen_checkpoint). Writes relsplit.json (summary + per-scene records), results.md, samples.png
(first 16) and run_meta.json. Smoke: N_SCENES=16.
"""
import datetime
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image
from torchvision.utils import save_image

from datasets.clevr_dataset import MAX_OBJECTS, ORIG_H, ORIG_W
from eval.clevr_eval_callbacks import ClevrMetricsCallback, _generate_images, _scene_to_data_sample
from eval.evaluate_clevr import _score_image, check_spatial_relationship_3d
from factory import build_model
from models.utility_models import strip_compiled_prefix

N_SCENES = int(os.environ.get("N_SCENES", 1024))
SEED = int(os.environ.get("SEED", 0))
BATCH = int(os.environ.get("BATCH", 128))
USE_EMA = os.environ.get("USE_EMA", "1") == "1"
HOP_BINS = ["1", "2", "3", "4", "5+", "inf"]
dev = torch.device("cuda")


def hop_matrix(adj: np.ndarray) -> np.ndarray:
    """All-pairs shortest directed path length (np.inf if unreachable) of a boolean (n, n) adjacency."""
    n = adj.shape[0]
    hop = np.full((n, n), np.inf)
    reach = np.eye(n, dtype=bool)
    frontier = np.eye(n, dtype=bool)
    for k in range(1, n):
        frontier = (frontier.astype(int) @ adj.astype(int)) > 0
        new = frontier & ~reach
        hop[new] = k
        reach |= new
    return hop


def given_hops(cond: torch.Tensor, n: int) -> dict:
    """Hop count of every ordered pair per relation type, from a reduced-mode condition (MAX_OBJECTS, 55).

    rel[i] = (left, right, front, behind) rows of object i: left[a, b] = "b is left of a", right[b, a] = "a is
    right of b" (the same fact), likewise front / behind. L[a, b] = b is left of a; F[a, b] = b is in front of a.
    """
    rel = cond[:, 15:].reshape(MAX_OBJECTS, 4, MAX_OBJECTS).numpy() > 0.5
    left, right, front, behind = (rel[:n, k, :n] for k in range(4))
    L = left | right.T
    F = front | behind.T
    hL, hF = hop_matrix(L), hop_matrix(F)
    # GT "left"[s] holds t when t is left of s: path s -> t in L; "right"[s] holds t when s is left of t: t -> s
    return {"left": hL, "right": hL.T, "front": hF, "behind": hF.T,
            "n_given": int(L.sum() + F.sum())}


def hop_bin(h: float) -> str:
    if np.isinf(h):
        return "inf"
    return str(int(h)) if h < 5 else "5+"


def score(cond_samples, scenes, imgs, cb):
    H, l_vec, f_vec, sz_thresh = cb._calibration
    dino_proc, dino_mod, sig_proc, sig_mod, text_embeds = cb._eval_models
    records = []
    for i, (img, scene, ds) in enumerate(zip(imgs, scenes, cond_samples)):
        pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).resize(
            (ORIG_W, ORIG_H), Image.BILINEAR)
        m, matched = _score_image(pil, scene["objects"], scene["relationships"], H, l_vec, f_vec, sz_thresh,
                                  dino_proc, dino_mod, sig_proc, sig_mod, text_embeds, return_matches=True)
        n = len(scene["objects"])
        hops = given_hops(ds.embedding_conditions, n)
        rels = []
        for rel_type, lists in scene["relationships"].items():
            for s, targets in enumerate(lists):
                for t in targets:
                    if s in matched and t in matched:
                        ok = bool(check_spatial_relationship_3d(rel_type, matched[s], matched[t], l_vec, f_vec))
                        rels.append([rel_type, s, t, float(hops[rel_type][s, t]), ok])
        assert sum(r[4] for r in rels) == m["c_rel"] and len(rels) == m["t_rel"], (i, m, len(rels))
        n_gt = sum(len(tg) for lists in scene["relationships"].values() for tg in lists)
        records.append({"image_index": scene["image_index"], "n_objects": n, "n_given": hops["n_given"],
                        "n_gt_rel": n_gt, "matched": len(matched), "metrics": m, "rels": rels})
    return records


def summarize(records):
    def acc(rows):
        return {"acc": sum(r[4] for r in rows) / len(rows) if rows else None, "n": len(rows)}

    allr = [r for rec in records for r in rec["rels"]]
    tot = {k: sum(rec["metrics"][k] for rec in records) for k in records[0]["metrics"]}
    v = max(1, tot["v_matches"])
    out = {
        "n_scenes": len(records),
        "precision": tot["v_matches"] / max(1, tot["t_pred"]), "recall": tot["v_matches"] / max(1, tot["t_req"]),
        "perfect_gen": tot["perf"] / v,
        "spatial": tot["c_rel"] / max(1, tot["t_rel"]),
        "given": acc([r for r in allr if r[3] == 1]),
        "implied": acc([r for r in allr if 2 <= r[3] < np.inf]),
        "by_hop": {b: acc([r for r in allr if hop_bin(r[3]) == b]) for b in HOP_BINS},
        "unscored_gt_rel_share": 1 - len(allr) / max(1, sum(rec["n_gt_rel"] for rec in records)),
    }
    full = [rec for rec in records if rec["matched"] == rec["n_objects"]]
    out["scenes_all_matched"] = len(full) / len(records)
    out["scene_all_rel"] = sum(all(r[4] for r in rec["rels"]) for rec in full) / len(records)
    out["scene_all_given"] = sum(all(r[4] for r in rec["rels"] if r[3] == 1) for rec in full) / len(records)
    out["by_n_objects"] = {}
    for n in sorted({rec["n_objects"] for rec in records}):
        rows = [r for rec in records if rec["n_objects"] == n for r in rec["rels"]]
        out["by_n_objects"][n] = {"scenes": sum(rec["n_objects"] == n for rec in records), "spatial": acc(rows),
                                  "given": acc([r for r in rows if r[3] == 1]),
                                  "implied": acc([r for r in rows if 2 <= r[3] < np.inf])}
    return out


def fmt(a):
    return "–" if a["acc"] is None else f"{a['acc']:.3f} ({a['n']})"


def write_md(summ, path, meta):
    L = ["# CLEVR spatial accuracy: given vs chain-implied relations", "",
         f"Checkpoint `{meta['checkpoint']}` ({'EMA' if meta['use_ema'] else 'raw'} weights), {summ['n_scenes']} val scenes (seed {meta['seed']}).  ",
         f"Command: `{meta['command']}`  ",
         f"Git {meta['git_sha']} (dirty: {meta['git_dirty']}), {meta['hardware']}, {meta['wall_clock_s']:.0f} s. "
         "Definitions: the script's docstring.", "",
         "| metric | value |", "|---|---|",
         f"| spatial (W&B `val/clevr_spatial_acc`) | {summ['spatial']:.3f} |",
         f"| given (hop 1) | {fmt(summ['given'])} |", f"| implied (hop ≥ 2) | {fmt(summ['implied'])} |",
         f"| scenes with all objects matched | {summ['scenes_all_matched']:.3f} |",
         f"| scene: all relations right | {summ['scene_all_rel']:.3f} |",
         f"| scene: all given edges right | {summ['scene_all_given']:.3f} |",
         f"| precision / recall / perfect_gen | {summ['precision']:.3f} / {summ['recall']:.3f} / "
         f"{summ['perfect_gen']:.3f} |",
         f"| GT relations not scored (an object unmatched) | {summ['unscored_gt_rel_share']:.3f} |", "",
         "## By hop count (accuracy (n relations))", "", "| hop | " + " | ".join(HOP_BINS) + " |",
         "|---" * (len(HOP_BINS) + 1) + "|", "| acc | " + " | ".join(fmt(summ["by_hop"][b]) for b in HOP_BINS) + " |",
         "", "## By object count", "", "| n | scenes | spatial | given | implied |", "|---|---|---|---|---|"]
    for n, r in summ["by_n_objects"].items():
        L.append(f"| {n} | {r['scenes']} | {fmt(r['spatial'])} | {fmt(r['given'])} | {fmt(r['implied'])} |")
    path.write_text("\n".join(L) + "\n")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    ov_file, ckpt, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, start = time.time(), datetime.datetime.now().isoformat(timespec="seconds")
    ov = [line.strip() for line in open(ov_file) if line.strip() and not line.startswith(("RUN=", "eval"))]
    ov += ["run.wandb_project=null", f"eval.batch_size={BATCH}"]
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=ov)
    root = cfg.data.clevr_root
    cfg.eval_callbacks = None
    model = build_model(cfg, instantiate(cfg.diffusion))
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    step = sd.get("step")
    sd = sd["ema_state"] if USE_EMA else sd["model_state"]
    res = model.load_state_dict(strip_compiled_prefix(sd), strict=False)
    print(f"{'EMA' if USE_EMA else 'raw'} weights: {len(sd)} tensors, missing {len(res.missing_keys)}, "
          f"unexpected {len(res.unexpected_keys)} {res.unexpected_keys[:5]}", flush=True)
    assert not res.unexpected_keys
    model = model.to(dev).eval()
    print(f"loaded {ckpt} (step {step}); sampling {model.sampling_pipeline.num_inference_steps} steps, "
          f"batch {model.sampling_pipeline.batch_size}", flush=True)

    cb = ClevrMetricsCallback(root_dir=root, mode=cfg.data.mode)
    cb._load(dev)
    scenes = random.Random(SEED).sample(cb._scenes, min(N_SCENES, len(cb._scenes)))
    random.seed(SEED)  # reduction order and edge-direction coins in make_tensor_from_scene
    np.random.seed(SEED)
    cond_samples = [_scene_to_data_sample(s, cfg.data.mode) for s in scenes]
    c0 = cond_samples[0].embedding_conditions
    print(f"first scene: {len(scenes[0]['objects'])} objects, condition {tuple(c0.shape)} {c0.dtype}, "
          f"given edges {given_hops(c0, len(scenes[0]['objects']))['n_given']}", flush=True)
    torch.manual_seed(SEED)
    imgs = _generate_images(model, cond_samples, dev)
    print(f"images {tuple(imgs.shape)} {imgs.dtype} range [{imgs.min():.3f}, {imgs.max():.3f}]", flush=True)
    save_image(imgs[:16].float().cpu(), out_dir / "samples.png", nrow=4)

    records = score(cond_samples, scenes, imgs, cb)
    summ = summarize(records)
    meta = {"command": " ".join([f"N_SCENES={N_SCENES} SEED={SEED} BATCH={BATCH} USE_EMA={int(USE_EMA)}", "python", *sys.argv]),
            "checkpoint": ckpt, "checkpoint_step": step, "use_ema": USE_EMA, "overrides": ov, "seed": SEED,
            "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", ".")),
            "hardware": f"{torch.cuda.get_device_name(dev)} ({platform.node()})", "torch": torch.__version__,
            "start": start, "end": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_clock_s": time.time() - t0}
    (out_dir / "relsplit.json").write_text(json.dumps({"summary": summ, "records": records}))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    write_md(summ, out_dir / "results.md", meta)
    print((out_dir / "results.md").read_text())
