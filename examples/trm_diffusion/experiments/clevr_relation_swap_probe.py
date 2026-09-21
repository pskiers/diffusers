"""
experiments/clevr_relation_swap_probe.py — CLEVR analogue of
corruption_recovery_grid_probe.py / violation_sensitivity_probe.py: does the
model repair a spatial-relation mistake, on real CLEVR scenes, when the
mistake is injected into the CONDITIONING rather than the image?

Sudoku's corruption probes splice a wrong digit crop into the noised IMAGE
while the conditioning (given clues) stays fixed and correct. CLEVR can't
easily do the same thing (there's no clean way to "paste a wrong object" into
a continuous 3-D render), so this probe corrupts the other side: the starting
image is the REAL scene noised to t_start (always correct), and the
CONDITIONING is edited to describe a scene that differs from it.

The dataset's "reduced" conditioning mode (what both the DM and the
painter-thinker checkpoints were trained with) encodes, per object, a sparse
set of left/right/front/behind edges — the transitive reduction of the true
per-axis total order, i.e. only pairs adjacent in x-rank (left-right) or
y-rank (front-behind) get an explicit edge; every other pair's relation is
only recoverable by chaining through intermediate objects. Two corruption
schemes were considered and rejected before landing on this one:

  1. Hand-flip a single edge bit for an already-adjacent pair, leaving
     attribute rows and the centroid mask untouched. This sounds like the
     direct sudoku analogue (flip one "given"), but the centroid mask is
     ANONYMOUS (an unordered union of position blobs, see
     datasets.clevr_dataset.make_presence_mask_from_scene) — it fixes the
     SET of (x, y) points, not the marginal x-values and y-values
     independently. Satisfying a flipped left-right-only edge while leaving
     front-behind untouched would require an object to appear at a NEW
     (x, y) point outside that fixed set — physically impossible given the
     mask. Resolving it as a full 2-D position swap instead over-corrects
     the front-behind relation too, which was never supposed to change.
  2. Insert a brand new edge for a NON-adjacent (implied-only) pair, leaving
     the real chain of direct edges untouched — the CLEVR analogue of
     violation_sensitivity_probe.py's "non_violating" condition. This is
     provably unsatisfiable: a non-adjacent pair's true relation is exactly
     what the untouched intervening direct edges already imply, so
     asserting the opposite is a straight logical contradiction (a cycle),
     not a "maybe valid under another solution" case like sudoku's sparse
     givens. There is no ambiguity to exploit here — CLEVR geometry has a
     single ground truth. So there is no CLEVR "non_violating" condition;
     that axis is dropped entirely (see module discussion with the user).

This probe instead corrupts at the SCENE level: pick disjoint groups of
object indices and cyclically rotate their true (3d_coords, pixel_coords)
among each other (a group of 2 = a plain swap; longer groups = a harder,
more-objects-affected permutation), keeping every object's own attributes
(color/shape/material/size/rotation) bound to its original index. Then
rebuild `relationships` and the "reduced" conditioning FRESH from this
permuted-but-still-real scene. This is always achievable (the blob position
SET, and hence the centroid mask, is completely unaffected — only WHICH
object's attributes claim which blob changes) and self-consistent (every
edge, not just the ones touching the swapped objects, is recomputed, so
there's no risk of a partial, contradictory edit).

The conditioning is the ground truth for a given run (it describes a fully
self-consistent, achievable alternate scene), and the starting sample (the
real photo) is what's mismatched with it -- same regime as the sudoku
corruption probes (conditioning always correct, intermediate sample carries
the mistake), just with a different mechanism for producing the mismatch.
Per corrupted object, after generating the final image, DINO+SigLIP
detection/matching (eval.evaluate_clevr._score_image, unaffected since
attributes are untouched by the corruption) locates it and classifies:
  success   — detected near the position the CONDITIONING asked for (the
              swap target) — the model corrected the mismatch
  stuck     — detected near its own ORIGINAL true position — the model
              failed to update away from the stale, mismatched content
  collapsed — not confidently detected/matched at all

Two orthogonal knobs replace the sudoku scripts' n_cells / violating axis:
  cycle_length  — objects rotated per corrupted group (2 = simple swap;
                  bigger = each object displaced further through more
                  neighbours, a harder edit)
  n_groups      — how many independent (disjoint) such groups per scene

Also reports each condition's aggregate clevr_spatial_acc (same c_rel/t_rel
computation as eval.clevr_eval_callbacks.ClevrMetricsCallback, but over the
corrupted-scene generation) as a global collateral-damage signal — it mixes
targeted and non-targeted relations, so a drop there beyond what the
targeted success/stuck/collapsed breakdown explains means the correction
disturbed relations that were never touched by the corruption.

Usage:
    python experiments/clevr_relation_swap_probe.py \\
      experiment=clevr_unet_concat_painter \\
      data.clevr_root=cache_dir \\
      +checkpoint=runs/clevr_unet_concat_big.pt \\
      +probe.num_samples=64 +probe.t_starts=[300,500,700] \\
      +probe.cycle_lengths=[2,3] +probe.n_groups_values=[1,2]

    # Options (all under +probe.*):
    #   num_samples       — default 64 (real CLEVR scenes sampled from val)
    #   seed              — default 0
    #   split             — CLEVR split to sample scenes from (default "val")
    #   min_objects       — default 6 (must fit max(cycle_lengths)*max(n_groups_values)
    #                        disjoint objects in every sampled scene)
    #   max_objects       — default 10
    #   image_size        — default 256
    #   t_starts          — denoising start points, must be members of the
    #                       num_inference_steps-step schedule (default [100,300,500,700,900])
    #   cycle_lengths     — default [2, 3]
    #   n_groups_values   — default [1, 2]
    #   batch_size        — generation batch size (default 8)
    #   cfg_scale / num_inference_steps — default from model.sampling_pipeline
    #   out               — json path (default: alongside checkpoint)
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ablate_trm_loop_budget import _load_checkpoint
from datasets.clevr_dataset import MAX_OBJECTS, ORIG_H, ORIG_W, make_presence_mask_from_scene, make_tensor_from_scene
from datasets.data_sample import DataSample, collate_data_samples
from eval.clevr_eval_callbacks import _load_scenes
from eval.evaluate_clevr import _score_image, calibrate_camera_and_size
from factory import build_model
from hydra.utils import instantiate
from painter_only_denoise_probe import _get_painter

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def _denoise_full(model, sample: DataSample, x_t: torch.Tensor, run_timesteps, cfg_scale: float = 1.0) -> torch.Tensor:
    """Runs the FULL model pipeline at every step -- model(sample), i.e.
    thinker reasoning + painter denoising when a thinker is present, or just
    the painter when mode=painter_base -- NOT
    painter_only_denoise_probe._denoise_from's steering=None thinker-bypass.
    "Does the model recover its own mistakes" means the whole model,
    including any thinker, the same way targeted_corruption_probe.py's
    _run_full_trajectory_targeted calls model(step_sample) rather than the
    frozen painter alone."""
    x = x_t
    for t in run_timesteps:
        t_batch = t.expand(x.shape[0]).to(x.device)
        step_sample = dataclasses.replace(sample, x_noisy=x, timesteps=t_batch)
        noise_pred = model(step_sample).pred
        if cfg_scale != 1.0:
            null_sample = model.null_condition_sample(step_sample)
            noise_pred_u = model(null_sample).pred
            noise_pred = noise_pred_u + cfg_scale * (noise_pred - noise_pred_u)
        x = model.scheduler.step(noise_pred, t, x).prev_sample
    return x


# ── Scene-level position permutation ────────────────────────────────────────


def _recompute_relationships(objects: list[dict]) -> dict:
    """Rebuild left/right/front/behind adjacency lists from objects'
    3d_coords -- same convention as datasets.clevr_dataset.sample_random_scene
    (and real CLEVR scene jsons): relationships[rel][i] lists every j
    satisfying rel relative to i."""
    n = len(objects)
    relationships = {
        "left": [[] for _ in range(n)], "right": [[] for _ in range(n)],
        "front": [[] for _ in range(n)], "behind": [[] for _ in range(n)],
    }
    for i, obj_a in enumerate(objects):
        pos_a = obj_a["3d_coords"]
        for j, obj_b in enumerate(objects):
            if i == j:
                continue
            pos_b = obj_b["3d_coords"]
            if pos_b[0] < pos_a[0]:
                relationships["left"][i].append(j)
            else:
                relationships["right"][i].append(j)
            if pos_b[1] < pos_a[1]:
                relationships["front"][i].append(j)
            else:
                relationships["behind"][i].append(j)
    return relationships


def _sample_permutation_groups(n_objects: int, cycle_length: int, n_groups: int, rng: random.Random) -> list[list[int]]:
    """n_groups disjoint cyclic groups of cycle_length object indices each,
    drawn from range(n_objects). Empty if there isn't room for
    cycle_length * n_groups disjoint indices -- callers should pre-filter
    scenes (via min_objects) so this never actually happens."""
    need = cycle_length * n_groups
    if need > n_objects or cycle_length < 2 or n_groups < 1:
        return []
    order = list(range(n_objects))
    rng.shuffle(order)
    chosen = order[:need]
    return [chosen[i * cycle_length:(i + 1) * cycle_length] for i in range(n_groups)]


def _apply_position_permutation(scene: dict, groups: list[list[int]]) -> tuple[dict, dict]:
    """Cyclically rotate 3d_coords/pixel_coords among each group of object
    indices (attributes stay bound to their original index -- only WHERE
    each object appears moves). Returns (new_scene, assigned_true_owner):
    assigned_true_owner[i] is the object index whose TRUE position object i
    now occupies (== i for every object outside every group).

    Rebuilding relationships from these permuted-but-real coordinates (not
    hand-edited relation bits) guarantees the result is always physically
    achievable: the blob POSITION SET is identical to the true scene's
    (make_presence_mask_from_scene draws an unordered union over
    objects[*]["pixel_coords"], so relabelling who owns which coordinate
    doesn't change the mask at all), and every relation -- not just edges
    touching the swapped objects -- is recomputed self-consistently.
    """
    objects = copy.deepcopy(scene["objects"])
    assigned_true_owner = {i: i for i in range(len(objects))}
    for group in groups:
        m = len(group)
        true_coords = [(objects[idx]["3d_coords"], objects[idx]["pixel_coords"]) for idx in group]
        for t in range(m):
            idx = group[t]
            src = group[(t + 1) % m]
            c3, cpix = true_coords[(t + 1) % m]
            objects[idx]["3d_coords"] = list(c3)
            objects[idx]["pixel_coords"] = list(cpix)
            assigned_true_owner[idx] = src
    new_scene = dict(scene)
    new_scene["objects"] = objects
    new_scene["relationships"] = _recompute_relationships(objects)
    new_scene.pop("reduced_variants", None)
    return new_scene, assigned_true_owner


def _scene_to_sample(scene: dict, image_t: torch.Tensor, mask_size: int) -> DataSample:
    scene = dict(scene)
    scene["mode"] = "reduced"
    cond_tensor, mask = make_tensor_from_scene(scene)
    spatial = make_presence_mask_from_scene(scene, mask_size)
    return DataSample(images=image_t, embedding_conditions=cond_tensor[0], embedding_mask=mask[0], spatial_conditions=spatial)


# ── Cached batches (real scenes + images, sampled once, reused everywhere) ──


def _build_cached_batches(scenes: list[dict], image_dir: str, transform, image_size: int, batch_size: int, num_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = list(scenes)
    rng.shuffle(pool)
    pool = pool[:num_samples]

    cached = []
    for start in range(0, len(pool), batch_size):
        chunk = pool[start:start + batch_size]
        images_t = []
        for scene in chunk:
            image = Image.open(os.path.join(image_dir, scene["image_filename"])).convert("RGB")
            images_t.append(transform(image))
        cached.append({"scenes": chunk, "images_t": torch.stack(images_t)})
    logger.info(f"Cached {len(pool)} real CLEVR scenes across {len(cached)} batches.")
    return cached


# ── Per-object readout ───────────────────────────────────────────────────────


def _classify_object(matched_3d: dict, obj_idx: int, true_objects: list[dict], assigned_true_owner: dict):
    """success / stuck / collapsed for one corrupted object index, given the
    detected-and-matched 3-D positions from _score_image(return_matches=True).

    The conditioning is the ground truth for this run (it describes a fully
    self-consistent, achievable alternate scene -- see module docstring), and
    the starting sample (the real photo) is the thing that's mismatched with
    it. So: success = final position matches what the CONDITIONING asked for
    (the swap target) -- the model corrected the mismatch. stuck = final
    position still matches the ORIGINAL real photo -- the model failed to
    update away from the stale, mismatched content.

    Also returns (d_true, d_target, pair_dist) -- the raw distances from the
    DETECTED position to the true and target reference points, and the
    distance BETWEEN those two references (the scale of the swap itself) --
    or None for all three if collapsed (undetected). The binary
    success/stuck call is just "which of d_true/d_target is smaller"; these
    raw distances let a caller check whether that call reflects a confident
    placement (d_target << pair_dist << d_true, or vice versa) or a
    near-coin-flip one (d_true and d_target both close to pair_dist/2, i.e.
    the detection landed roughly equidistant between the two references
    rather than clearly near either)."""
    if obj_idx not in matched_3d:
        return "collapsed", None, None, None
    detected = np.asarray(matched_3d[obj_idx], dtype=np.float64)
    true_pos = np.asarray(true_objects[obj_idx]["3d_coords"][:2], dtype=np.float64)
    target_idx = assigned_true_owner[obj_idx]
    target_pos = np.asarray(true_objects[target_idx]["3d_coords"][:2], dtype=np.float64)
    d_true = float(np.linalg.norm(detected - true_pos))
    d_target = float(np.linalg.norm(detected - target_pos))
    pair_dist = float(np.linalg.norm(true_pos - target_pos))
    outcome = "stuck" if d_true <= d_target else "success"
    return outcome, d_true, d_target, pair_dist


@torch.no_grad()
def _run_config(
    model, cached_batches: list, run_timesteps, t_start: int, cfg_scale: float, mask_size: int,
    cycle_length: int, n_groups: int, base_seed: int,
    calib, eval_models, scratch: bool = False,
) -> dict:
    H, l_vec, f_vec, sz_thresh = calib
    dino_proc, dino_mod, sig_proc, sig_mod, text_embeds = eval_models
    painter = _get_painter(model)

    n_corrupted = n_stuck = n_success = n_collapsed = 0
    total_c_rel = total_t_rel = 0
    n_scenes_scored = 0
    sum_d_true = sum_d_target = sum_pair_dist = 0.0
    n_dist = 0
    n_scenes_all_success = n_scenes_all_stuck = 0

    for bi, cb in enumerate(cached_batches):
        scenes, images_t = cb["scenes"], cb["images_t"]
        device = next(model.parameters()).device
        images_t = images_t.to(device)
        B = images_t.shape[0]
        seed = base_seed + bi

        latents = painter.encode(images_t) if painter.vae is not None else images_t
        torch.manual_seed(seed)
        if scratch:
            x_t_base = torch.randn_like(latents)
        else:
            noise = torch.randn_like(latents)
            t_start_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
            x_t_base = model.scheduler.add_noise(latents, noise, t_start_batch)

        rng = random.Random(seed)
        permuted_scenes, owners = [], []
        for scene in scenes:
            n_obj = len(scene["objects"])
            groups = _sample_permutation_groups(n_obj, cycle_length, n_groups, rng)
            new_scene, assigned_true_owner = _apply_position_permutation(scene, groups)
            permuted_scenes.append(new_scene)
            owners.append((groups, assigned_true_owner))

        samples = [
            _scene_to_sample(sc, img, mask_size).to(device)
            for sc, img in zip(permuted_scenes, images_t)
        ]
        batch_sample = collate_data_samples(samples)

        final_latents = _denoise_full(model, batch_sample, x_t_base, run_timesteps, cfg_scale=cfg_scale)
        final_pixels = model.decode_for_eval(final_latents)

        for i in range(B):
            scene = scenes[i]
            groups, assigned_true_owner = owners[i]
            if not groups:
                continue
            img_np = (final_pixels[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np).resize((ORIG_W, ORIG_H), Image.BILINEAR)

            m, matched_3d = _score_image(
                pil_img, scene["objects"], scene["relationships"], H, l_vec, f_vec, sz_thresh,
                dino_proc, dino_mod, sig_proc, sig_mod, text_embeds, return_matches=True,
            )
            total_c_rel += m["c_rel"]
            total_t_rel += m["t_rel"]
            n_scenes_scored += 1

            scene_all_success = True
            scene_all_stuck = True
            for group in groups:
                for obj_idx in group:
                    outcome, d_true, d_target, pair_dist = _classify_object(
                        matched_3d, obj_idx, scene["objects"], assigned_true_owner
                    )
                    n_corrupted += 1
                    if outcome == "stuck":
                        n_stuck += 1
                        scene_all_success = False
                    elif outcome == "success":
                        n_success += 1
                        scene_all_stuck = False
                    else:
                        n_collapsed += 1
                        scene_all_success = False
                        scene_all_stuck = False
                    if d_true is not None:
                        sum_d_true += d_true
                        sum_d_target += d_target
                        sum_pair_dist += pair_dist
                        n_dist += 1
            if scene_all_success:
                n_scenes_all_success += 1
            if scene_all_stuck:
                n_scenes_all_stuck += 1

    return {
        "cycle_length": cycle_length,
        "n_groups": n_groups,
        "n_corrupted": n_corrupted,
        "stuck_rate": (n_stuck / n_corrupted) if n_corrupted else None,
        "success_rate": (n_success / n_corrupted) if n_corrupted else None,
        "collapse_rate": (n_collapsed / n_corrupted) if n_corrupted else None,
        "n_scenes_scored": n_scenes_scored,
        "spatial_acc": (total_c_rel / total_t_rel) if total_t_rel else None,
        # Diagnostic: is the success/stuck split a confident placement or
        # a near-coin-flip? mean_pair_dist is the scale (true<->target
        # distance); mean_d_true/mean_d_target << mean_pair_dist on the
        # "winning" side indicates real confidence, both close to
        # mean_pair_dist/2 indicates the detection landed roughly equidistant
        # (i.e. the binary classification is closer to arbitrary than real).
        "mean_d_true": (sum_d_true / n_dist) if n_dist else None,
        "mean_d_target": (sum_d_target / n_dist) if n_dist else None,
        "mean_pair_dist": (sum_pair_dist / n_dist) if n_dist else None,
        # Per-SCENE (not per-object) "all mistakes fixed" / "nothing fixed"
        # rates -- the true (not independence-estimated) analogue of
        # sudoku's puzzle-level accuracy. all_success_rate = every corrupted
        # object in the scene individually classified "success" (zero stuck,
        # zero collapsed); all_stuck_rate = every one stayed "stuck" (zero
        # success, zero collapsed). Compare all_success_rate against
        # success_rate**need (the independence estimate) -- if the real rate
        # is notably higher, per-object outcomes are positively correlated
        # within a scene (the generation tends to succeed or fail as a
        # whole, not object-by-object).
        "all_success_rate": (n_scenes_all_success / n_scenes_scored) if n_scenes_scored else None,
        "all_stuck_rate": (n_scenes_all_stuck / n_scenes_scored) if n_scenes_scored else None,
    }


@torch.no_grad()
def _run_clean(model, cached_batches: list, run_timesteps, t_start: int, cfg_scale: float, mask_size: int, base_seed: int, calib, eval_models, scratch: bool = False) -> dict:
    H, l_vec, f_vec, sz_thresh = calib
    dino_proc, dino_mod, sig_proc, sig_mod, text_embeds = eval_models
    painter = _get_painter(model)

    total_c_rel = total_t_rel = 0
    n_scenes_scored = 0
    n_scenes_all_correct = 0

    for bi, cb in enumerate(cached_batches):
        scenes, images_t = cb["scenes"], cb["images_t"]
        device = next(model.parameters()).device
        images_t = images_t.to(device)
        B = images_t.shape[0]
        seed = base_seed + bi

        latents = painter.encode(images_t) if painter.vae is not None else images_t
        torch.manual_seed(seed)
        if scratch:
            x_t_base = torch.randn_like(latents)
        else:
            noise = torch.randn_like(latents)
            t_start_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
            x_t_base = model.scheduler.add_noise(latents, noise, t_start_batch)

        samples = [_scene_to_sample(sc, img, mask_size).to(device) for sc, img in zip(scenes, images_t)]
        batch_sample = collate_data_samples(samples)

        final_latents = _denoise_full(model, batch_sample, x_t_base, run_timesteps, cfg_scale=cfg_scale)
        final_pixels = model.decode_for_eval(final_latents)

        for i in range(B):
            scene = scenes[i]
            img_np = (final_pixels[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np).resize((ORIG_W, ORIG_H), Image.BILINEAR)
            m = _score_image(
                pil_img, scene["objects"], scene["relationships"], H, l_vec, f_vec, sz_thresh,
                dino_proc, dino_mod, sig_proc, sig_mod, text_embeds,
            )
            total_c_rel += m["c_rel"]
            total_t_rel += m["t_rel"]
            n_scenes_scored += 1
            # Strict, per-scene "every relation correct" -- stricter than
            # the aggregate spatial_acc below, which averages over all
            # relation-pairs across all scenes (so one wrong relation in an
            # otherwise-correct scene barely dents it). t_rel==0 (no
            # detected relation pairs at all) counts as NOT all-correct.
            if m["t_rel"] > 0 and m["c_rel"] == m["t_rel"]:
                n_scenes_all_correct += 1

    return {
        "n_scenes_scored": n_scenes_scored,
        "spatial_acc": (total_c_rel / total_t_rel) if total_t_rel else None,
        "all_relations_correct_rate": (n_scenes_all_correct / n_scenes_scored) if n_scenes_scored else None,
    }


def _load_eval_models(device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SiglipModel, SiglipProcessor
    from eval.evaluate_clevr import JOINT_PROMPTS

    dino_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_mod = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    sig_proc = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    sig_mod = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)
    with torch.no_grad():
        t_inputs = sig_proc(text=JOINT_PROMPTS, padding="max_length", return_tensors="pt").to(device)
        text_embeds = sig_mod.get_text_features(**t_inputs)
        text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)
    return dino_proc, dino_mod, sig_proc, sig_mod, text_embeds


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is None:
        raise SystemExit(
            "ERROR: No checkpoint specified.\n"
            "  Usage: python experiments/clevr_relation_swap_probe.py experiment=<name> "
            "+checkpoint=<path/to/checkpoint.pt> [+probe.xxx=...]"
        )

    pb = cfg.get("probe", {})
    num_samples: int = pb.get("num_samples", 64)
    seed: int = pb.get("seed", 0)
    split: str = pb.get("split", "val")
    min_objects: int = pb.get("min_objects", 3)
    max_objects: int = pb.get("max_objects", MAX_OBJECTS)
    image_size: int = pb.get("image_size", 256)
    batch_size: int = pb.get("batch_size", 8)
    t_starts: list[int] = list(pb.get("t_starts", [100, 300, 500, 700, 900]))
    include_scratch: bool = pb.get("include_scratch", False)
    cycle_lengths: list[int] = list(pb.get("cycle_lengths", [2, 3]))
    n_groups_values: list[int] = list(pb.get("n_groups_values", [1, 2]))
    out_path: str = pb.get("out", str(Path(checkpoint).parent / "clevr_relation_swap.json"))

    # (cycle_length, n_groups) needs cycle_length*n_groups DISJOINT real
    # objects in a scene -- CLEVR scenes only ever have up to MAX_OBJECTS
    # (10). Combos needing more than that are unsatisfiable by ANY real
    # scene, not just rare -- skip them rather than raising, and group the
    # rest by their shared `need` so combos that happen to need the same
    # object count (e.g. cycle_length=3,n_groups=2 and cycle_length=6,
    # n_groups=1 both need 6) reuse one scene pool + matched clean baseline
    # instead of redoing the same generation twice.
    feasible: dict[int, list[tuple[int, int]]] = {}
    skipped: list[tuple[int, int, int]] = []
    for cycle_length in cycle_lengths:
        if cycle_length < 2:
            skipped.append((cycle_length, 0, 0))  # cycle_length<2 is a no-op permutation
            continue
        for n_groups in n_groups_values:
            need = cycle_length * n_groups
            if need > max_objects:
                skipped.append((cycle_length, n_groups, need))
                continue
            feasible.setdefault(need, []).append((cycle_length, n_groups))

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Checkpoint: {checkpoint}")

    clevr_root = cfg.data.clevr_root
    all_scenes = _load_scenes(clevr_root, split, min_objects, max_objects)
    logger.info(f"{len(all_scenes)} scenes in split={split} with {min_objects}<=n_objects<={max_objects}")
    if skipped:
        for cl, ng, need in skipped:
            if ng == 0:
                logger.warning(f"skipping cycle_length={cl} (no-op permutation, identical to clean)")
            else:
                logger.warning(
                    f"skipping cycle_length={cl} n_groups={ng} (needs {need} disjoint objects per scene, "
                    f"but CLEVR scenes have at most {max_objects})"
                )

    filename_split = "val" if split == "validation" else split
    image_dir = os.path.join(clevr_root, "CLEVR_v1.0", "images", filename_split)
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize([0.5], [0.5])])
    mask_size = image_size // 8

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    _load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = model.to(device)
    model.eval()

    pipeline = model.sampling_pipeline
    cfg_scale: float = pb.get("cfg_scale", pipeline.cfg_scale)
    num_inference_steps: int = pb.get("num_inference_steps", pipeline.num_inference_steps)

    model.scheduler.set_timesteps(num_inference_steps, device=device)
    full_timesteps = model.scheduler.timesteps
    valid_t = set(int(t.item()) for t in full_timesteps)
    bad = [t for t in t_starts if t not in valid_t]
    if bad:
        raise SystemExit(
            f"t_starts {bad} are not members of the {num_inference_steps}-step schedule {sorted(valid_t, reverse=True)}."
        )
    logger.info(
        f"cfg_scale={cfg_scale}  num_inference_steps={num_inference_steps}  "
        f"t_starts={t_starts}  cycle_lengths={cycle_lengths}  n_groups_values={n_groups_values}"
    )

    logger.info("Loading DINO + SigLIP eval models, and calibrating camera/size...")
    calib = calibrate_camera_and_size(clevr_root, split=split)
    eval_models = _load_eval_models(device)

    # Global clean baseline: any object count, built once, reused for every t_start.
    global_clean_batches = _build_cached_batches(all_scenes, image_dir, transform, image_size, batch_size, num_samples, seed)

    # One scene pool (+ its own matched-clean baseline) per distinct `need`,
    # built once and reused across every t_start and every (cycle_length,
    # n_groups) combo that shares that need -- see the feasible/skipped
    # comment above main()'s combo-building loop.
    need_batches: dict[int, list] = {}
    for need in sorted(feasible):
        pool = [s for s in all_scenes if len(s["objects"]) >= need]
        if len(pool) < num_samples:
            logger.warning(f"need={need}: only {len(pool)} scenes have >={need} objects (< num_samples={num_samples}); using all of them")
        need_batches[need] = _build_cached_batches(pool, image_dir, transform, image_size, batch_size, num_samples, seed)
        logger.info(f"need={need}: pool={len(pool)} scenes, combos={feasible[need]}")

    # (label, t_start value passed through to _run_clean/_run_config for the
    # add_noise path -- unused when scratch=True, run_timesteps, scratch)
    sweep_points = [
        (str(t), t, full_timesteps[full_timesteps <= t], False) for t in t_starts
    ]
    if include_scratch:
        # x_init = pure noise, no real photo at all, full schedule -- the
        # genuine noise_lvl=100 endpoint (t_start tops out at 95 on this
        # schedule and always starts from a NOISED REAL PHOTO, so even
        # t_start=95 still carries a little real signal; scratch carries
        # none).
        sweep_points.append(("scratch", 0, full_timesteps, True))

    results: dict = {}
    for label, t_start, run_timesteps, scratch in sweep_points:
        logger.info(f"Running t_start={label} ({len(run_timesteps)} steps, scratch={scratch}) ...")

        clean = _run_clean(model, global_clean_batches, run_timesteps, t_start, cfg_scale, mask_size, seed, calib, eval_models, scratch=scratch)
        results[f"t_start={label}/clean"] = clean
        logger.info(f"  clean → {clean}")

        for need, combos in feasible.items():
            clean_matched = _run_clean(model, need_batches[need], run_timesteps, t_start, cfg_scale, mask_size, seed, calib, eval_models, scratch=scratch)
            results[f"t_start={label}/clean_matched/need={need}"] = clean_matched
            logger.info(f"  clean_matched/need={need} → {clean_matched}")

            for cycle_length, n_groups in combos:
                r = _run_config(
                    model, need_batches[need], run_timesteps, t_start, cfg_scale, mask_size,
                    cycle_length, n_groups, seed, calib, eval_models, scratch=scratch,
                )
                key = f"t_start={label}/cycle_length={cycle_length}/n_groups={n_groups}"
                results[key] = r
                logger.info(f"  {key} → {r}")

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(checkpoint), "num_samples": num_samples, "results": results}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
