import os
import json
import math
import random
import zipfile
from typing import Optional

import requests
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples

# Constants aligned with CLEVR
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIALS = ["rubber", "metal"]
SHAPES = ["cube", "sphere", "cylinder"]
SIZES = ["small", "large"]
RELATIONS = ["left", "right", "front", "behind"]

MAX_OBJECTS = 10

# Global mappings
COLOR2ID = {k: i for i, k in enumerate(COLORS)}
MAT2ID = {k: i for i, k in enumerate(MATERIALS)}
SHAPE2ID = {k: i for i, k in enumerate(SHAPES)}
SIZE2ID = {k: i for i, k in enumerate(SIZES)}

# Approximate boundaries for CLEVR scene generation
X_RANGE = (-3.0, 3.0)
Y_RANGE = (-3.0, 3.0)
MIN_DIST = 0.7  # Minimum distance between objects to avoid overlap
ORIG_W, ORIG_H = 480, 320

# Channels: [0:8] color, [8:11] shape, [11:13] material, [13:15] size, [15] presence
MASK_CHANNELS = 16
# Physical 3D radii used by CLEVR renderer (in scene units)
_CLEVR_RADIUS = {"small": 0.35, "large": 0.70}


def calibrate_mask_projection(scenes, num_scenes=150):
    """
    Fit a 2-D homography H that maps pixel coords (u, v) → 3-D ground-plane
    coords (x, y) using a subset of real CLEVR scenes.  Returns H_inv, the
    inverse mapping 3-D → pixel, as a (3, 3) float64 numpy array.

    Used to derive perspective-correct Gaussian blob sizes: an object at a
    position far from the camera will have a smaller projected radius than one
    that is close.
    """
    import cv2

    uv_pts, xy_pts = [], []
    for scene in scenes[:num_scenes]:
        for obj in scene["objects"]:
            uv_pts.append(obj["pixel_coords"][:2])
            xy_pts.append(obj["3d_coords"][:2])

    H, _ = cv2.findHomography(
        np.array(uv_pts, dtype=np.float32),
        np.array(xy_pts, dtype=np.float32),
    )
    return np.linalg.inv(H).astype(np.float64)


def _project_3d_to_pixel(x, y, H_inv):
    """Apply the inverse homography H_inv to a single 3-D ground-plane point."""
    pt = H_inv @ np.array([x, y, 1.0], dtype=np.float64)
    return pt[:2] / pt[2]


def make_mask_from_scene(scene_dict, mask_size=32, H_inv=None):
    """
    Convert a scene dict into a spatial conditioning mask tensor.

    Returns a float32 tensor of shape (MASK_CHANNELS, mask_size, mask_size).
    Each object is drawn as a soft 2-D Gaussian blob at its projected image
    position.  The blob sigma is derived from the object's 3-D radius:

    * If H_inv (3×3 inverse homography, 3-D→pixel) is supplied, the 3-D radius
      is projected to pixel space for a perspective-correct size.
    * Otherwise (synthetic scenes from sample_random_scene that use a simple
      linear projection) the radius is scaled analytically.

    Multiple overlapping objects take the per-channel maximum.

    Channel layout
    --------------
    0–7  : one-hot color    (8 classes, COLORS order)
    8–10 : one-hot shape    (3 classes, SHAPES order)
    11–12: one-hot material (2 classes, MATERIALS order)
    13–14: one-hot size     (2 classes, SIZES order)
    15   : presence (1.0 inside every blob)
    """
    objects = scene_dict["objects"]
    mask = torch.zeros(MASK_CHANNELS, mask_size, mask_size, dtype=torch.float32)

    ys = torch.arange(mask_size, dtype=torch.float32)
    xs = torch.arange(mask_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    for obj in objects:
        # Blob centre in mask coordinates
        cx = obj["pixel_coords"][0] / ORIG_W * mask_size
        cy = obj["pixel_coords"][1] / ORIG_H * mask_size

        r_3d = _CLEVR_RADIUS[obj["size"]]

        if H_inv is not None:
            # Perspective-correct sigma: project the object edge in 3-D and
            # measure the resulting pixel displacement.
            x3, y3 = obj["3d_coords"][0], obj["3d_coords"][1]
            uv_center = _project_3d_to_pixel(x3, y3, H_inv)
            uv_edge = _project_3d_to_pixel(x3 + r_3d, y3, H_inv)
            r_pix = float(np.linalg.norm(uv_edge - uv_center))
            # Scale from original pixel space to mask pixel space
            sigma = max(r_pix / ORIG_W * mask_size, 0.5)
        else:
            # Synthetic scenes use a linear projection: pixel ∝ (coord+3)/6.
            # The projected radius in mask pixels is therefore:
            sigma = max(r_3d / 6.0 * mask_size, 0.5)

        blob = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2.0 * sigma**2))

        mask[COLOR2ID[obj["color"]]] = mask[COLOR2ID[obj["color"]]].maximum(blob)
        mask[8 + SHAPE2ID[obj["shape"]]] = mask[8 + SHAPE2ID[obj["shape"]]].maximum(blob)
        mask[11 + MAT2ID[obj["material"]]] = mask[11 + MAT2ID[obj["material"]]].maximum(blob)
        mask[13 + SIZE2ID[obj["size"]]] = mask[13 + SIZE2ID[obj["size"]]].maximum(blob)
        mask[15] = mask[15].maximum(blob)

    return mask


def make_reveal_from_scene(
    image: torch.Tensor,
    scene_dict: dict,
    n_reveal: int,
    radius_frac: float = 0.12,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Zero everywhere except circular patches around ``n_reveal`` randomly
    chosen objects' true pixel positions, cropped directly from the real
    (already-transformed) target image tensor.

    An exact-pixel, exact-position anchor for a handful of objects — CLEVR's
    analogue of MNIST-Sudoku's "given cells": a diagnostic to test whether a
    few perfect anchors let the TRM place the rest via relations. Diagnostic
    only, not a deployable mechanism — it requires the real target image,
    which doesn't exist yet at actual generation time.

    Args:
        image: (C, H, W) already-transformed target image tensor.
        scene_dict: raw scene dict (has "objects" with "pixel_coords").
        n_reveal: number of objects to reveal (clamped to the scene's count).
        radius_frac: reveal-circle radius as a fraction of image size.
        rng: optional random.Random for reproducible object selection.
    """
    C, H, W = image.shape
    revealed = torch.zeros_like(image)
    objects = scene_dict["objects"]
    if not objects or n_reveal <= 0:
        return revealed

    rng = rng or random
    n = min(n_reveal, len(objects))
    chosen = rng.sample(range(len(objects)), n)

    radius = radius_frac * min(H, W)
    ys = torch.arange(H, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, -1)

    for idx in chosen:
        obj = objects[idx]
        cx = obj["pixel_coords"][0] / ORIG_W * W
        cy = obj["pixel_coords"][1] / ORIG_H * H
        keep = ((ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2).unsqueeze(0)  # (1, H, W)
        revealed = torch.where(keep, image, revealed)

    return revealed


def _crop_object_swatch(image: torch.Tensor, obj: dict, H_inv: np.ndarray, swatch_size: int, margin: float = 1.6) -> torch.Tensor:
    """Crop a tight, perspective-correct square patch around one object's
    true rendered position from its real, full-resolution source image
    tensor, resized to (3, swatch_size, swatch_size). ``H_inv`` — see
    calibrate_mask_projection — gives a perspective-correct pixel radius
    from the object's known 3-D size, the same projection make_mask_from_scene
    uses for its Gaussian blob sigma.

    The crop WINDOW is sized from the "large" reference radius regardless of
    this object's own size (``size_override="large"`` below), not from the
    object's own radius: sizing the window to the object's own radius would
    make every crop fill the tile after resizing, erasing the small/large
    size cue entirely. With a fixed window, a "small" object naturally
    occupies less of the tile than a "large" one, so the resized swatch
    still visually encodes relative size."""
    C, H, W = image.shape
    cx, cy, _ = _object_pixel_circle(obj, W, H, H_inv)
    _, _, r_pix = _object_pixel_circle(obj, W, H, H_inv, size_override="large")
    r_pix = max(r_pix * margin, 4.0)

    x0, x1 = int(round(cx - r_pix)), int(round(cx + r_pix))
    y0, y1 = int(round(cy - r_pix)), int(round(cy + r_pix))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return torch.zeros(C, swatch_size, swatch_size, dtype=image.dtype)

    crop = image[:, y0:y1, x0:x1].unsqueeze(0)
    return F.interpolate(crop, size=(swatch_size, swatch_size), mode="bilinear", align_corners=False)[0]


def _object_pixel_circle(obj: dict, image_w: int, image_h: int, H_inv: np.ndarray, size_override: Optional[str] = None):
    """(cx, cy, r) — object's projected center and perspective-correct
    radius, in pixel coordinates scaled to (image_w, image_h). Pass
    ``size_override`` (e.g. "large") to compute r for a hypothetical object
    of that size AT this object's position, instead of its own true size —
    used by _crop_object_swatch to get a size-invariant crop window."""
    cx = obj["pixel_coords"][0] / ORIG_W * image_w
    cy = obj["pixel_coords"][1] / ORIG_H * image_h
    r_3d = _CLEVR_RADIUS[size_override or obj["size"]]
    x3, y3 = obj["3d_coords"][0], obj["3d_coords"][1]
    uv_center = _project_3d_to_pixel(x3, y3, H_inv)
    uv_edge = _project_3d_to_pixel(x3 + r_3d, y3, H_inv)
    r = float(np.linalg.norm(uv_edge - uv_center)) / ORIG_W * image_w
    return cx, cy, r


def _is_isolated(
    target_obj: dict,
    scene_objects: list,
    image_w: int,
    image_h: int,
    H_inv: np.ndarray,
    margin: float = 1.6,
    size_override: Optional[str] = None,
) -> bool:
    """True if no other object's real silhouette intrudes into target_obj's
    CROP WINDOW — radius = size_override's reference radius * margin, i.e.
    exactly the window _crop_object_swatch will actually use (pass the same
    margin/size_override to both, as extract_clevr_swatch_table does).
    Checking against target_obj's own true radius instead — the previous
    behavior — under-margins whenever the actual crop window is bigger than
    the object itself, which is always true here since the window is sized
    to "large" regardless of the object's own size: a neighbor could sit
    just outside a same-size isolation check yet still land inside the
    size-invariant window used for the real crop. Objects can visually
    overlap in 2-D even when spaced apart in 3-D, so this checks in
    projected pixel space, not 3-D distance."""
    tcx, tcy, tr = _object_pixel_circle(target_obj, image_w, image_h, H_inv, size_override=size_override)
    tr *= margin
    for other in scene_objects:
        if other is target_obj:
            continue
        ocx, ocy, orad = _object_pixel_circle(other, image_w, image_h, H_inv)
        if math.hypot(tcx - ocx, tcy - ocy) < tr + orad:
            return False
    return True


def extract_clevr_swatch_table(
    scenes: list,
    image_dir: str,
    H_inv: np.ndarray,
    swatch_size: int = 32,
    margin: float = 1.6,
    seed: int = 0,
) -> torch.Tensor:
    """Scan real CLEVR scenes once and, for each unique (color, shape,
    material, size) combination, crop a tight square patch around one real,
    unoccluded object instance's true position from its actual rendered
    image — a real photorealistic anchor (same renderer/lighting engine as
    the rest of the dataset), not a synthetic icon (a hand-drawn shape has
    essentially no visual correspondence to a Blender render). Candidates
    are preferred by: isolated (no overlapping neighbor) first, then fewest
    total objects in the source scene, to avoid a crop contaminated by
    clutter or occlusion. The result is a fixed lookup table reused at both
    train and inference time: same attribute combination -> the same real
    pixels, every time — CLEVR's analogue of MNISTSudokuDataset's
    same_digit_images.

    Returns:
        table: (len(COLORS)*len(SHAPES)*len(MATERIALS)*len(SIZES), 3, S, S)
            float32 tensor in [0, 1], ordered color->shape->material->size
            (must match models.condition_encoders._clevr_swatch_indices).
        isolated_mask: bool tensor, same length as table's first dim. False
            means that combination had no isolated candidate anywhere in the
            scanned scenes and fell back to a possibly-overlapping instance
            (or is all-zero because the combination never appeared at all —
            check the table for that separately). For inspection only, not
            needed by ObjectFeatureEncoderV1Swatch.
    """
    rng = random.Random(seed)
    candidates: dict = {}
    for scene in scenes:
        objects = scene["objects"]
        n_objects = len(objects)
        for obj in objects:
            key = (obj["color"], obj["shape"], obj["material"], obj["size"])
            isolated = _is_isolated(obj, objects, ORIG_W, ORIG_H, H_inv, margin=margin, size_override="large")
            candidates.setdefault(key, []).append((scene["image_filename"], obj, n_objects, isolated))

    def _pick(options):
        isolated_opts = [o for o in options if o[3]]
        pool = isolated_opts if isolated_opts else options
        min_n = min(o[2] for o in pool)
        best = [o for o in pool if o[2] == min_n]
        return rng.choice(best)

    to_tensor = T.ToTensor()
    image_cache: dict = {}
    table = []
    isolated_mask = []
    missing = []
    n_occluded_fallback = 0
    for color in COLORS:
        for shape in SHAPES:
            for material in MATERIALS:
                for size in SIZES:
                    key = (color, shape, material, size)
                    options = candidates.get(key)
                    if not options:
                        missing.append(key)
                        table.append(torch.zeros(3, swatch_size, swatch_size))
                        isolated_mask.append(False)
                        continue
                    filename, obj, _n, isolated = _pick(options)
                    if not isolated:
                        n_occluded_fallback += 1
                    isolated_mask.append(isolated)
                    if filename not in image_cache:
                        img = Image.open(os.path.join(image_dir, filename)).convert("RGB")
                        image_cache[filename] = to_tensor(img)
                    table.append(_crop_object_swatch(image_cache[filename], obj, H_inv, swatch_size, margin))
    if missing:
        print(f"extract_clevr_swatch_table: no example found for {len(missing)} combinations, filled with zeros: {missing}")
    if n_occluded_fallback:
        print(f"extract_clevr_swatch_table: {n_occluded_fallback} combinations had no isolated instance, fell back to a possibly-occluded one")
    return torch.stack(table), torch.tensor(isolated_mask, dtype=torch.bool)


def _adj_lists_to_matrix(adj_lists, n):
    """Convert a list-of-lists adjacency representation to a numpy bool matrix."""
    mat = np.zeros((n, n), dtype=np.bool_)
    for i, neighbours in enumerate(adj_lists[:n]):
        for j in neighbours:
            if j < n:
                mat[i, j] = True
    return mat


def transitive_reduce_direction(adj_mat, object_order):
    """
    Single-pass greedy transitive reduction for one spatial direction.

    The algorithm processes objects in *object_order* and, for each object a,
    removes any edge a→b that is already reachable via some other object c
    currently in a's neighbour set.  Because edges removed from c's list
    (when c is processed earlier) are no longer visible when a is processed,
    different orderings produce different — but all valid — sparse graphs.
    This is the source of structural diversity across variants.

    Object indices in the returned edges always refer to the original object
    numbering, so they remain consistent across variants.

    Args:
        adj_mat:      numpy bool array (n, n); adj_mat[i, j] = True means j is
                      in this direction of i.
        object_order: permutation of range(n) controlling processing order.

    Returns:
        List of (a, b) int tuples — the surviving edges.
    """
    mat = adj_mat.copy()
    n = mat.shape[0]

    for a in object_order:
        if not mat[a].any():
            continue
        # Two-hop reachability from a via its current neighbours (vectorised).
        # two_hop[b] is True if any current neighbour c of a has mat[c, b]=True.
        two_hop = mat[a] @ mat  # shape (n,), bool
        # Keep only edges a→b that are NOT reachable via another neighbour.
        # "Another neighbour" means we must exclude the direct b→b self-loop
        # that would appear if mat[b, b] were set — it isn't, so mat[a] & two_hop
        # gives exactly the redundant edges (reachable via at least one c≠b).
        redundant = mat[a] & two_hop
        mat[a] &= ~redundant

    return [(int(a), int(b)) for a in range(n) for b in range(n) if mat[a, b]]


def compute_reduced_variants(relationships, n, n_variants):
    """
    Pre-compute *n_variants* differently-ordered transitive reductions for both
    spatial axes (left/right and front/behind).

    Each variant uses an independently shuffled object-processing order, giving
    a different sparse-but-valid representation of the same scene's geometry.

    Returns:
        List of (left_mat, front_mat) pairs of numpy bool (n, n) arrays.
        Stored as arrays (not edge-tuple lists) to avoid Python GC pressure at
        training time from millions of small tuple objects.
    """
    # Build adjacency matrices once — reused across all variants.
    left_mat = _adj_lists_to_matrix(relationships["left"], n)
    front_mat = _adj_lists_to_matrix(relationships["front"], n)

    variants = []
    for _ in range(n_variants):
        order = list(range(n))
        random.shuffle(order)
        # transitive_reduce_direction already returns a copy, so each variant
        # gets its own independent matrix.
        reduced_left = _edges_to_matrix(transitive_reduce_direction(left_mat, order), n)
        reduced_front = _edges_to_matrix(transitive_reduce_direction(front_mat, order), n)
        variants.append((reduced_left, reduced_front))
    return variants


def _edges_to_matrix(edges, n):
    """Convert a list of (a, b) edge tuples back to a numpy bool (n, n) matrix."""
    mat = np.zeros((n, n), dtype=np.bool_)
    for a, b in edges:
        mat[a, b] = True
    return mat


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------


def sample_random_scene(num_objects=None, mode="absolute"):
    """
    Generates a random, physically valid scene dictionary.

    Args:
        num_objects (int): Number of objects. If None, random (3 to 10).
        mode (str): "absolute", "relative", or "reduced".
    """
    if num_objects is None:
        num_objects = random.randint(3, 10)  # TODO change low back to 3

    objects = []
    positions = []  # Store (x, y) for distance checks

    for _ in range(num_objects):
        valid_pos = False
        attempts = 0
        while not valid_pos and attempts < 100:
            x = random.uniform(*X_RANGE)
            y = random.uniform(*Y_RANGE)

            too_close = False
            for px, py in positions:
                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                if dist < MIN_DIST:
                    too_close = True
                    break

            if not too_close:
                valid_pos = True
                positions.append((x, y))

                obj = {
                    "color": random.choice(COLORS),
                    "material": random.choice(MATERIALS),
                    "shape": random.choice(SHAPES),
                    "size": random.choice(SIZES),
                    "rotation": random.uniform(0, 360),
                    "3d_coords": [x, y, 0.0],
                    "pixel_coords": [(x + 3) / 6 * 480, (y + 3) / 6 * 320, 10.0],
                }
                objects.append(obj)
            attempts += 1

    relationships = {
        "left": [[] for _ in objects],
        "right": [[] for _ in objects],
        "front": [[] for _ in objects],
        "behind": [[] for _ in objects],
    }

    for i in range(len(objects)):
        obj_A = objects[i]
        pos_A = obj_A["3d_coords"]

        for j in range(len(objects)):
            if i == j:
                continue

            obj_B = objects[j]
            pos_B = obj_B["3d_coords"]

            if pos_B[0] < pos_A[0]:
                relationships["left"][i].append(j)
            else:
                relationships["right"][i].append(j)

            if pos_B[1] < pos_A[1]:
                relationships["front"][i].append(j)
            else:
                relationships["behind"][i].append(j)

    return {"objects": objects, "relationships": relationships, "mode": mode}


def make_tensor_from_scene(scene_dict):
    """
    Converts a scene dictionary into model-ready tensors.

    Returns:
        cond_tensor (Tensor): Shape (1, MAX_OBJECTS, Feature_Dim)
        mask (Tensor): Shape (1, MAX_OBJECTS) - 1.0 for real objects, 0.0 for padding
    """
    mode = scene_dict["mode"]
    objects = scene_dict["objects"]
    relationships = scene_dict["relationships"]

    # For "reduced" mode, build the 4 relation grids before the per-object loop.
    #
    # Step 1 — pick a structural variant (which edges survived the greedy reduction):
    #   • Dataset path:  one of the pre-computed variants is chosen at random.
    #   • Eval/sampling: a single variant is computed on-the-fly with a shuffled order.
    #
    # Step 2 — on-the-fly complementary assignment:
    #   For each surviving edge (a, b) in the left axis we toss a coin:
    #     heads → store as left[a, b]   ("b is left of a")
    #     tails → store as right[b, a]  (equivalent: "a is right of b")
    #   Likewise for the front/behind axis.  This forces the model to understand
    #   both directions rather than always finding information in the same slot.
    if mode == "reduced":
        n = len(objects)

        if "reduced_variants" in scene_dict:
            left_mat_r, front_mat_r = random.choice(scene_dict["reduced_variants"])
        else:
            order = list(range(n))
            random.shuffle(order)
            left_edges = transitive_reduce_direction(_adj_lists_to_matrix(relationships["left"], n), order)
            front_edges = transitive_reduce_direction(_adj_lists_to_matrix(relationships["front"], n), order)
            left_mat_r = _edges_to_matrix(left_edges, n)
            front_mat_r = _edges_to_matrix(front_edges, n)

        # All 4 grids stacked as (4, MAX_OBJECTS, MAX_OBJECTS) — row i gives row i
        # of each grid, so per-object slice is rel_flat[i] = 40 dims flat.
        rel_all = torch.zeros(4, MAX_OBJECTS, MAX_OBJECTS, dtype=torch.float32)
        left_g, right_g, front_g, behind_g = rel_all[0], rel_all[1], rel_all[2], rel_all[3]

        # Vectorised coin flips: for each surviving edge, randomly assign it to
        # the canonical or complementary slot.
        left_coords = np.argwhere(left_mat_r)  # (E, 2)
        if len(left_coords):
            coins = np.random.random(len(left_coords)) < 0.5
            for (a, b), heads in zip(left_coords, coins):
                if a < MAX_OBJECTS and b < MAX_OBJECTS:
                    if heads:
                        left_g[a, b] = 1.0  # "b is left of a"
                    else:
                        right_g[b, a] = 1.0  # equivalent: "a is right of b"

        front_coords = np.argwhere(front_mat_r)
        if len(front_coords):
            coins = np.random.random(len(front_coords)) < 0.5
            for (a, b), heads in zip(front_coords, coins):
                if a < MAX_OBJECTS and b < MAX_OBJECTS:
                    if heads:
                        front_g[a, b] = 1.0  # "b is in front of a"
                    else:
                        behind_g[b, a] = 1.0  # equivalent: "a is behind b"

        # Pre-flatten to (MAX_OBJECTS, 40): rel_flat[i] = object i's relation row
        rel_flat = rel_all.permute(1, 0, 2).reshape(MAX_OBJECTS, 4 * MAX_OBJECTS)

    obj_vectors = []

    for i, obj in enumerate(objects):
        # --- 1. Common Attributes (15 dims) ---
        rot_rad = math.radians(obj["rotation"])
        rot_vec = torch.tensor([math.sin(rot_rad), math.cos(rot_rad)], dtype=torch.float32)

        sz_vec = torch.tensor([SIZE2ID[obj["size"]]], dtype=torch.float32)
        mat_vec = torch.tensor([MAT2ID[obj["material"]]], dtype=torch.float32)
        sh_vec = F.one_hot(torch.tensor(SHAPE2ID[obj["shape"]]), num_classes=3).float()
        col_vec = F.one_hot(torch.tensor(COLOR2ID[obj["color"]]), num_classes=8).float()

        base = torch.cat([rot_vec, sz_vec, mat_vec, sh_vec, col_vec])

        # --- 2. Mode Specifics ---
        if mode == "absolute":
            px, py, pz = obj["pixel_coords"]
            p_vec = torch.tensor([px / 480.0, py / 320.0, pz / 20.0], dtype=torch.float32)

            x3, y3, z3 = obj["3d_coords"]
            c3_vec = torch.tensor([x3 / 5.0, y3 / 5.0, z3 / 5.0], dtype=torch.float32)

            spatial = torch.cat([p_vec, c3_vec])
            full_vec = torch.cat([base, spatial])

        elif mode == "relative":
            rel_grid = torch.zeros((4, MAX_OBJECTS), dtype=torch.float32)

            for r_idx, rel_name in enumerate(RELATIONS):
                if i < len(relationships[rel_name]):
                    target_indices = relationships[rel_name][i]
                    for t_idx in target_indices:
                        if t_idx < MAX_OBJECTS:
                            rel_grid[r_idx, t_idx] = 1.0

            rels = rel_grid.flatten()
            full_vec = torch.cat([base, rels])

        elif mode == "reduced":
            # 4 × MAX_OBJECTS = 40 dims — same shape as "relative" for architecture
            # compatibility, but sparser and with randomised direction encoding.
            full_vec = torch.cat([base, rel_flat[i]])

        obj_vectors.append(full_vec)

    # --- 3. Stacking and Padding ---
    dim = 21 if mode == "absolute" else 55  # "relative" and "reduced" both use 55

    padded_objs = torch.zeros((1, MAX_OBJECTS, dim), dtype=torch.float32)
    mask = torch.zeros((1, MAX_OBJECTS), dtype=torch.float32)

    limit = min(len(obj_vectors), MAX_OBJECTS)
    if limit > 0:
        stacked = torch.stack(obj_vectors[:limit])
        padded_objs[0, :limit, :] = stacked
        mask[0, :limit] = 1.0

    return padded_objs, mask


class CLEVRHybridDataset(Dataset):
    URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    collate_fn = staticmethod(collate_data_samples)

    def __init__(
        self,
        root_dir,
        split="train",
        mode="absolute",
        image_size=256,
        download=True,
        n_reduced_samples=16,
        reveal_n_objects: int = 0,
        reveal_radius_frac: float = 0.12,
        include_centroid_mask: bool = False,
    ):
        """
        Args:
            mode (str): "absolute", "relative", "reduced", or "mask".
                "mask" returns a spatial (MASK_CHANNELS, H, W) conditioning tensor
                instead of a per-object feature sequence.
            n_reduced_samples (int): Number of differently-ordered reductions to
                pre-compute per scene when mode="reduced". One is chosen
                randomly in each __getitem__ call, followed by an on-the-fly
                left↔right / front↔behind coin flip per edge.
            reveal_n_objects: if > 0, ALSO populate ``spatial_conditions`` with
                a real-image reveal around this many randomly chosen objects'
                true pixel positions (rest zeroed) — see
                make_reveal_from_scene. Additive on top of whatever `mode`
                already produces in `embedding_conditions`. Diagnostic only
                (uses the real target image, unavailable at real inference).
            reveal_radius_frac: reveal-circle radius, as a fraction of image
                size, when reveal_n_objects > 0.
            include_centroid_mask: if True, ALSO populate `spatial_conditions`
                with the per-attribute Gaussian-blob mask (make_mask_from_scene)
                at every object's true position — additive on top of `mode`'s
                own `embedding_conditions`, unlike mode="mask" which replaces
                it entirely. Mutually exclusive with reveal_n_objects (only one
                spatial_conditions source at a time).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.mask_size = image_size // 8  # matches VAE 8× downsampling
        self.dataset_path = os.path.join(root_dir, "CLEVR_v1.0")
        self.reveal_n_objects = reveal_n_objects
        self.reveal_radius_frac = reveal_radius_frac
        self.include_centroid_mask = include_centroid_mask
        if reveal_n_objects > 0 and include_centroid_mask:
            raise ValueError("reveal_n_objects and include_centroid_mask are mutually exclusive.")

        if download and not os.path.exists(self.dataset_path):
            self._download_and_extract()

        # Load Scenes
        filename_split = "val" if split == "validation" else split
        scene_path = os.path.join(self.dataset_path, "scenes", f"CLEVR_{filename_split}_scenes.json")

        print(f"Loading {mode} scenes from {scene_path}...")
        with open(scene_path, "r") as f:
            self.scenes = json.load(f)["scenes"]

        # Pre-compute perspective calibration matrix for mask mode.
        # H_inv maps 3-D ground-plane coords → pixel coords so we can derive
        # perspective-correct Gaussian blob radii for each object.
        self.H_inv = None
        if mode == "mask" or include_centroid_mask:
            print("Calibrating perspective projection for mask generation...")
            self.H_inv = calibrate_mask_projection(self.scenes)

        # Pre-compute reduction variants once so __getitem__ stays cheap:
        # each call only does a random.choice + a small number of grid writes.
        if mode == "reduced":
            print(f"Pre-computing {n_reduced_samples} reduced graph variants " f"for {len(self.scenes)} scenes...")
            for scene in self.scenes:
                n = len(scene["objects"])
                scene["reduced_variants"] = compute_reduced_variants(scene["relationships"], n, n_reduced_samples)

        self.image_dir = os.path.join(self.dataset_path, "images", filename_split)

        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize([0.5], [0.5])])

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.scenes)

    def _download_and_extract(self):
        print(f"Downloading CLEVR to {self.root_dir}...")
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        zip_path = os.path.join(self.root_dir, "CLEVR_v1.0.zip")
        if not os.path.exists(zip_path):
            r = requests.get(self.URL, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root_dir)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        image = Image.open(os.path.join(self.image_dir, scene["image_filename"])).convert("RGB")
        image_t = self.transform(image)

        if self.mode == "mask":
            spatial_mask = make_mask_from_scene(scene, self.mask_size, self.H_inv)
            return DataSample(images=image_t, spatial_conditions=spatial_mask)

        # Temporarily inject the mode so our shared function knows how to process it
        scene["mode"] = self.mode

        # Get tensors. make_tensor_from_scene adds a batch dim of 1, so we strip it off here.
        cond_tensor, mask = make_tensor_from_scene(scene)

        spatial_conditions = None
        if self.reveal_n_objects > 0:
            spatial_conditions = make_reveal_from_scene(image_t, scene, self.reveal_n_objects, self.reveal_radius_frac)
        elif self.include_centroid_mask:
            spatial_conditions = make_mask_from_scene(scene, self.mask_size, self.H_inv)

        return DataSample(
            images=image_t,
            embedding_conditions=cond_tensor[0],
            embedding_mask=mask[0],
            spatial_conditions=spatial_conditions,
        )
