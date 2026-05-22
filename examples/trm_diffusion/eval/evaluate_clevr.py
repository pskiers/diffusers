import os
import cv2
import json
import torch
import torchvision
import numpy as np
import argparse
import itertools
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
from scipy.optimize import linear_sum_assignment

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SiglipProcessor, SiglipModel

from datasets.clevr_dataset import CLEVRHybridDataset, COLORS, SHAPES, ORIG_H, ORIG_W


# Joint Prompts for Material/Color context
JOINT_COMBOS = list(itertools.product(COLORS, ["matte rubber", "shiny metal"], SHAPES))
JOINT_PROMPTS = [f"a 3d render of a {c} {m} {s}" for c, m, s in JOINT_COMBOS]


# 2D to 3D Clevr calibration
def calibrate_camera_and_size(data_dir, split="val"):
    """
    Uses the official CLEVR dataset to calibrate the 3D projection for evaluation.
    """
    print("Calibrating 3D Geometry and Size Thresholds from official CLEVR...")
    dataset = CLEVRHybridDataset(root_dir=data_dir, split=split, download=False)
    scenes = dataset.scenes[:150]

    uv_points, xy_points, left_diffs, front_diffs = [], [], [], []
    small_widths_3d, large_widths_3d = [], []

    for scene in scenes:
        for obj in scene["objects"]:
            xy_points.append(obj["3d_coords"][:2])
            uv_points.append(obj["pixel_coords"][:2])
            w_3d = 0.7 if obj["size"] == "small" else 1.4
            if obj["size"] == "small":
                small_widths_3d.append(w_3d)
            else:
                large_widths_3d.append(w_3d)
        for r_type in ["left", "front"]:
            for i, targets in enumerate(scene["relationships"][r_type]):
                for j in targets:
                    diff = np.array(scene["objects"][j]["3d_coords"][:2]) - np.array(
                        scene["objects"][i]["3d_coords"][:2]
                    )
                    if r_type == "left":
                        left_diffs.append(diff)
                    else:
                        front_diffs.append(diff)

    H, _ = cv2.findHomography(np.array(uv_points, dtype=np.float32), np.array(xy_points, dtype=np.float32))
    size_thresh = (np.median(small_widths_3d) + np.median(large_widths_3d)) / 2
    l_vec = np.mean(left_diffs, axis=0)
    l_vec /= np.linalg.norm(l_vec)
    f_vec = np.mean(front_diffs, axis=0)
    f_vec /= np.linalg.norm(f_vec)
    return H, l_vec, f_vec, size_thresh


def get_3d_center(bbox, H):
    u, v = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    return cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)[0][0]


def get_3d_width(bbox, H):
    p1 = cv2.perspectiveTransform(np.array([[[bbox[0], bbox[3]]]], dtype=np.float32), H)[0][0]
    p2 = cv2.perspectiveTransform(np.array([[[bbox[2], bbox[3]]]], dtype=np.float32), H)[0][0]
    return np.linalg.norm(p1 - p2)


def check_spatial_relationship_3d(rel_type, pos_s, pos_t, l_vec, f_vec):
    diff = pos_t - pos_s
    if rel_type == "left":
        return np.dot(l_vec, diff) > 0
    if rel_type == "right":
        return np.dot(-l_vec, diff) > 0
    if rel_type == "front":
        return np.dot(f_vec, diff) > 0
    if rel_type == "behind":
        return np.dot(-f_vec, diff) > 0
    return False


def score_all_relations(anchors_3d, gt_relationships, l_vec, f_vec):
    score = 0
    for rel_type, rel_lists in gt_relationships.items():
        for source_idx, targets in enumerate(rel_lists):
            for target_idx in targets:
                if source_idx in anchors_3d and target_idx in anchors_3d:
                    if check_spatial_relationship_3d(
                        rel_type, anchors_3d[source_idx], anchors_3d[target_idx], l_vec, f_vec
                    ):
                        score += 1
    return score


def print_report(m):
    print("\n" + "=" * 70 + "\n GENERATIVE MODEL PERFORMANCE REPORT\n" + "=" * 70)
    print(f"\n[1] CONDITIONING ADHERENCE")
    print(f"    Precision (What % of generated objects were requested?): {m['v_matches']/max(1, m['t_pred'])*100:.1f}%")
    print(f"    Recall    (What % of requested objects were drawn?):     {m['v_matches']/max(1, m['t_req'])*100:.1f}%")
    print(f"    Hallucinations: {m['hallucinations']} | Misses: {m['t_req'] - m['v_matches']}")

    v = max(1, m["v_matches"])
    print(f"\n[2] ATTRIBUTE FIDELITY")
    print(f"    Color: {m['c_col']/v*100:.1f}% | Shape: {m['c_sh']/v*100:.1f}% | Material: {m['c_mat']/v*100:.1f}%")
    print(f"    Size:  {m['c_sz']/v*100:.1f}% | PERFECT GEN: {m['perf']/v*100:.1f}%")

    rel_acc = (m["c_rel"] / max(1, m["t_rel"])) * 100 if m["t_rel"] > 0 else 0.0
    print(f"\n[3] SPATIAL/RELATIONAL ACCURACY: {rel_acc:.1f}% ({m['c_rel']}/{m['t_rel']})\n" + "=" * 70)


def evaluate_model_samples(samples_dir, clevr_dir, limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples_path = Path(samples_dir)
    metadata_path = samples_path / "metadata_rank0.jsonl"

    print(f"Loading generated samples metadata from {metadata_path}...")
    with open(metadata_path, "r") as f:
        samples = [json.loads(line) for line in f]

    H, l_vec, f_vec, sz_thresh = calibrate_camera_and_size(clevr_dir)

    print("Loading Models (DINO & SigLIP)...")
    dino_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_mod = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    sig_proc = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    sig_mod = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device)

    with torch.no_grad():
        t_inputs = sig_proc(text=JOINT_PROMPTS, padding="max_length", return_tensors="pt").to(device)
        text_embeds = sig_mod.get_text_features(**t_inputs)
        text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)

    m = {
        "t_req": 0,
        "t_pred": 0,
        "v_matches": 0,
        "hallucinations": 0,
        "c_col": 0,
        "c_sh": 0,
        "c_mat": 0,
        "c_sz": 0,
        "perf": 0,
        "t_rel": 0,
        "c_rel": 0,
    }

    loop_limit = limit if limit is not None else len(samples)
    random.shuffle(samples)
    for i, item in enumerate(tqdm(samples[:loop_limit])):
        img_path = samples_path / item["file_name"]
        raw_image = Image.open(img_path).convert("RGB")

        if raw_image.size != (ORIG_W, ORIG_H):
            image = raw_image.resize((ORIG_W, ORIG_H), Image.BILINEAR)
        else:
            image = raw_image

        gt_objects = item.get("objects", [])

        # 1. BBOX DETECTION
        gt_phrases = [f"{o['size']} {o['color']} {o['material']} {o['shape']}" for o in gt_objects]
        text_in = " . ".join(gt_phrases) + " . cube . sphere . cylinder ."

        inputs = dino_proc(images=image, text=text_in, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_mod(**inputs)
        res = dino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.3, text_threshold=0.25, target_sizes=[image.size[::-1]]
        )[0]

        bboxes = []
        if len(res["boxes"]) > 0:
            keep = torchvision.ops.nms(res["boxes"], res["scores"], iou_threshold=0.65)
            n_boxes = res["boxes"][keep].tolist()
            n_scores = res["scores"][keep].tolist()
            sorted_boxes = [b for _, b in sorted(zip(n_scores, n_boxes), reverse=True)]

            for b in sorted_boxes:
                cx1, cy1 = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                is_ghost = False
                for kb in bboxes:
                    cx2, cy2 = (kb[0] + kb[2]) / 2, (kb[1] + kb[3]) / 2
                    if (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 < 15**2:
                        is_ghost = True
                        break
                if not is_ghost:
                    bboxes.append(b)

        m["t_req"] += len(gt_objects)
        m["t_pred"] += len(bboxes)

        # 2. BATCHED ATTRIBUTE CLASSIFICATION
        pred_objs = []
        all_crops = []
        for b in bboxes:
            x1, y1, x2, y2 = map(int, b)
            all_crops.extend(
                [
                    image.crop(
                        (max(0, x1 - 5), max(0, y1 - 5), min(image.width, x2 + 5), min(image.height, y2 + 5))
                    ),  # Full view
                    image.crop(
                        (x1 + (x2 - x1) // 4, y1 + (y2 - y1) // 4, x2 - (x2 - x1) // 4, y2 - (y2 - y1) // 4)
                    ),  # Center view
                ]
            )

        if all_crops:
            s_inputs = sig_proc(images=all_crops, return_tensors="pt").to(device)
            with torch.no_grad():
                img_embeds = sig_mod.get_image_features(**s_inputs)
                img_embeds /= img_embeds.norm(p=2, dim=-1, keepdim=True)
                logits_batch = (img_embeds @ text_embeds.T) * sig_mod.logit_scale.exp()

            for idx, b in enumerate(bboxes):
                avg_logits = (logits_batch[idx * 2] + logits_batch[idx * 2 + 1]) / 2.0
                win = avg_logits.argmax().item()
                color, mat_str, shape = JOINT_COMBOS[win]

                w_3d = get_3d_width(b, H)
                pred_objs.append(
                    {
                        "bbox": b,
                        "size": "large" if w_3d > sz_thresh else "small",
                        "color": color,
                        "material": "rubber" if "matte" in mat_str else "metal",
                        "shape": shape,
                        "center_3d": get_3d_center(b, H),
                    }
                )

        # 3. HUNGARIAN MATCHING
        cost = np.zeros((len(gt_objects), len(pred_objs)))
        for gi, go in enumerate(gt_objects):
            for pi, po in enumerate(pred_objs):
                c = sum([go[k] != po[k] for k in ["color", "shape", "material", "size"]])
                cost[gi, pi] = c

        g_idx, p_idx = linear_sum_assignment(cost) if cost.size > 0 else ([], [])
        matched_3d = {}

        for gi, pi in zip(g_idx, p_idx):
            if cost[gi, pi] <= 2:
                m["v_matches"] += 1
                po, go = pred_objs[pi], gt_objects[gi]
                matched_3d[gi] = po["center_3d"]

                if go["color"] == po["color"]:
                    m["c_col"] += 1
                if go["shape"] == po["shape"]:
                    m["c_sh"] += 1
                if go["material"] == po["material"]:
                    m["c_mat"] += 1
                if go["size"] == po["size"]:
                    m["c_sz"] += 1
                if cost[gi, pi] == 0:
                    m["perf"] += 1

        m["hallucinations"] += len(pred_objs) - len(matched_3d)

        # 4. SPATIAL PERMUTATION SOLVER
        gr = item.get("relationships", {})
        phrase_to_gt = {}
        for gi, ph in enumerate(gt_phrases):
            phrase_to_gt.setdefault(ph, []).append(gi)

        ambig = [inds for inds in phrase_to_gt.values() if len(inds) > 1 and all(idx in matched_3d for idx in inds)]
        for g_group in ambig:
            best_a = matched_3d.copy()
            best_s = score_all_relations(best_a, gr, l_vec, f_vec)
            orig_a = [matched_3d[idx] for idx in g_group]

            for perm in itertools.permutations(orig_a):
                test_a = matched_3d.copy()
                for idx, new_a in zip(g_group, perm):
                    test_a[idx] = new_a
                cur_s = score_all_relations(test_a, gr, l_vec, f_vec)
                if cur_s > best_s:
                    best_s = cur_s
                    best_a = test_a.copy()
            matched_3d = best_a

        # 5. SCORE FINAL SPATIAL RELATIONSHIPS
        m["c_rel"] += score_all_relations(matched_3d, gr, l_vec, f_vec)
        for r_lists in gr.values():
            for si, targets in enumerate(r_lists):
                for ti in targets:
                    if si in matched_3d and ti in matched_3d:
                        m["t_rel"] += 1

        if (i + 1) % 50 == 0:
            print_report(m)

    print_report(m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True, help="Path to your generated samples and JSONL")
    parser.add_argument("--clevr_dir", type=str, required=True, help="Path to original CLEVR for calibration")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate_model_samples(args.samples_dir, args.clevr_dir, limit=args.limit)
