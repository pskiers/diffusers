import os
import json
import math
import random
import zipfile
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


# Constants aligned with CLEVR
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
MATERIALS = ['rubber', 'metal']
SHAPES = ['cube', 'sphere', 'cylinder']
SIZES = ['small', 'large']
RELATIONS = ['left', 'right', 'front', 'behind']

MAX_OBJECTS = 10

# Global mappings
COLOR2ID = {k: i for i, k in enumerate(COLORS)}
MAT2ID   = {k: i for i, k in enumerate(MATERIALS)}
SHAPE2ID = {k: i for i, k in enumerate(SHAPES)}
SIZE2ID  = {k: i for i, k in enumerate(SIZES)}

# Approximate boundaries for CLEVR scene generation
X_RANGE = (-3.0, 3.0)
Y_RANGE = (-3.0, 3.0)
MIN_DIST = 0.7  # Minimum distance between objects to avoid overlap


def sample_random_scene(num_objects=None, mode="absolute"):
    """
    Generates a random, physically valid scene dictionary.

    Args:
        num_objects (int): Number of objects. If None, random (3 to 10).
        mode (str): "absolute" or "relative".
    """
    if num_objects is None:
        num_objects = random.randint(3, 10)

    objects = []
    positions = [] # Store (x, y) for distance checks

    for _ in range(num_objects):
        valid_pos = False
        attempts = 0
        while not valid_pos and attempts < 100:
            x = random.uniform(*X_RANGE)
            y = random.uniform(*Y_RANGE)

            too_close = False
            for px, py in positions:
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                if dist < MIN_DIST:
                    too_close = True
                    break

            if not too_close:
                valid_pos = True
                positions.append((x, y))

                obj = {
                    'color': random.choice(COLORS),
                    'material': random.choice(MATERIALS),
                    'shape': random.choice(SHAPES),
                    'size': random.choice(SIZES),
                    'rotation': random.uniform(0, 360),
                    '3d_coords': [x, y, 0.0],
                    'pixel_coords': [
                        (x + 3) / 6 * 480,
                        (y + 3) / 6 * 320,
                        10.0
                    ]
                }
                objects.append(obj)
            attempts += 1

    relationships = {
        'left':  [[] for _ in objects],
        'right': [[] for _ in objects],
        'front': [[] for _ in objects],
        'behind': [[] for _ in objects]
    }

    for i in range(len(objects)):
        obj_A = objects[i]
        pos_A = obj_A['3d_coords']

        for j in range(len(objects)):
            if i == j: continue

            obj_B = objects[j]
            pos_B = obj_B['3d_coords']

            if pos_B[0] < pos_A[0]:
                relationships['left'][i].append(j)
            else:
                relationships['right'][i].append(j)

            if pos_B[1] < pos_A[1]:
                relationships['front'][i].append(j)
            else:
                relationships['behind'][i].append(j)

    return {
        "objects": objects,
        "relationships": relationships,
        "mode": mode
    }

def make_tensor_from_scene(scene_dict):
    """
    Converts a scene dictionary into model-ready tensors.

    Returns:
        cond_tensor (Tensor): Shape (1, MAX_OBJECTS, Feature_Dim)
        mask (Tensor): Shape (1, MAX_OBJECTS) - 1.0 for real objects, 0.0 for padding
    """
    mode = scene_dict['mode']
    objects = scene_dict['objects']
    relationships = scene_dict['relationships']

    obj_vectors = []

    for i, obj in enumerate(objects):
        # --- 1. Common Attributes (15 dims) ---
        rot_rad = math.radians(obj['rotation'])
        rot_vec = torch.tensor([math.sin(rot_rad), math.cos(rot_rad)], dtype=torch.float32)

        sz_vec = torch.tensor([SIZE2ID[obj['size']]], dtype=torch.float32)
        mat_vec = torch.tensor([MAT2ID[obj['material']]], dtype=torch.float32)
        sh_vec = F.one_hot(torch.tensor(SHAPE2ID[obj['shape']]), num_classes=3).float()
        col_vec = F.one_hot(torch.tensor(COLOR2ID[obj['color']]), num_classes=8).float()

        base = torch.cat([rot_vec, sz_vec, mat_vec, sh_vec, col_vec])

        # --- 2. Mode Specifics ---
        if mode == "absolute":
            px, py, pz = obj['pixel_coords']
            p_vec = torch.tensor([px/480.0, py/320.0, pz/20.0], dtype=torch.float32)

            x3, y3, z3 = obj['3d_coords']
            c3_vec = torch.tensor([x3/5.0, y3/5.0, z3/5.0], dtype=torch.float32)

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

        obj_vectors.append(full_vec)

    # --- 3. Stacking and Padding ---
    dim = 21 if mode == "absolute" else 55

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

    def __init__(self, root_dir, split="train", mode="absolute", image_size=256, download=True):
        """
        Args:
            mode (str): "absolute" (uses coordinates) or "relative" (uses relationships)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.dataset_path = os.path.join(root_dir, "CLEVR_v1.0")

        if download and not os.path.exists(self.dataset_path):
            self._download_and_extract()

        # Load Scenes
        filename_split = "val" if split == "validation" else split
        scene_path = os.path.join(self.dataset_path, "scenes", f"CLEVR_{filename_split}_scenes.json")

        print(f"Loading {mode} scenes from {scene_path}...")
        with open(scene_path, 'r') as f:
            self.scenes = json.load(f)['scenes']

        self.image_dir = os.path.join(self.dataset_path, "images", filename_split)

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

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
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(self.root_dir)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        image = Image.open(os.path.join(self.image_dir, scene['image_filename'])).convert("RGB")

        # Temporarily inject the mode so our shared function knows how to process it
        scene['mode'] = self.mode

        # Get tensors. make_tensor_from_scene adds a batch dim of 1, so we strip it off here.
        cond_tensor, mask = make_tensor_from_scene(scene)

        return {
            "img": self.transform(image),
            "obj_features": cond_tensor[0],
            "obj_mask": mask[0]
        }
