"""Probe pixel colours of solid vs faint paths and the dataset metadata layout."""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

TRM = Path("/net/scratch/hscra/plgrid/plgmgrzanka/diffusers/examples/trm_diffusion")
sys.path.insert(0, str(TRM))
from datasets.amaze_dataset import AmazeDataset

GEN = TRM / "runs/bagel_infer/bagel_maze_square_2026.09.22_15.57.52_50/generated_images"
pid = "02653c68-ce56-4e6e-b39b-38c4c65d112b"

ds = AmazeDataset(str(TRM / "data/amaze/maze/square/n7_test.parquet"), split="test", image_size=144,
                  condition_field="m_original_img", target_field="sol_img", num_channels=3, include_metadata=True)
idx = [i for i in range(len(ds)) if ds[i].metadata.get("id") == pid][0]
md = ds[idx].metadata
print("metadata keys:", list(md.keys()))
for k, v in md.items():
    if isinstance(v, Image.Image):
        print(f"  {k}: PIL {v.size} {v.mode}")
meta = json.loads(md["metadata"]) if isinstance(md["metadata"], str) else md["metadata"]
print("inner metadata keys:", list(meta.keys()))
print("path_cell_ids:", meta.get("path_cell_ids"))

gen = np.array(Image.open(GEN / f"7x7_{pid}_attempt001.jpg").convert("RGB"))
print("gen size", gen.shape)
hsv = cv2.cvtColor(gen, cv2.COLOR_RGB2HSV)
# faint branch: horizontal at y~575, x~480..640 ; solid path: horizontal at y~678
for name, (y, x0, x1) in {"faint_h": (575, 480, 640), "solid_h": (678, 100, 600)}.items():
    ys = slice(y - 6, y + 7)
    patch_rgb = gen[ys, x0:x1].reshape(-1, 3)
    patch_hsv = hsv[ys, x0:x1].reshape(-1, 3)
    nonwhite = patch_rgb.min(1) < 245
    print(f"{name}: n_nonwhite={nonwhite.sum()}")
    if nonwhite.any():
        print("   RGB  p10/50/90:", np.percentile(patch_rgb[nonwhite], [10, 50, 90], axis=0).round().tolist())
        print("   HSV  p10/50/90:", np.percentile(patch_hsv[nonwhite], [10, 50, 90], axis=0).round().tolist())
bg = gen[20:90, 120:400].reshape(-1, 3)
bgh = hsv[20:90, 120:400].reshape(-1, 3)
print("background RGB p1/50/99:", np.percentile(bg, [1, 50, 99], axis=0).round().tolist())
print("background HSV p50/99/99.9:", np.percentile(bgh, [50, 99, 99.9], axis=0).round().tolist())
