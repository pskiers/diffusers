import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
import json
import zipfile
import requests
import random
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T
import accelerate
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.configuration_utils import register_to_config

from model_utils import _extract_into_tensor, trunc_normal_init_
from data_utils import SafeIterator, LimitedLoader


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--channels", nargs="+", type=int, default=[128, 128], required=False, help="Channels in each UNet block."
    )
    parser.add_argument(
        "--down_block_types",
        nargs="+",
        type=str,
        default=["AttnDownBlock2D", "DownBlock2D"],
        required=False,
        help="Down block types."
    )
    parser.add_argument(
        "--up_block_types",
        nargs="+",
        type=str,
        default=["UpBlock2D", "AttnUpBlock2D"],
        required=False,
        help="Up block types."
    )
    parser.add_argument("--T", type=int, default=3, help="T")
    parser.add_argument("--n", type=int, default=6, help="n")
    parser.add_argument("--N_supervision", type=int, default=4, help="N_supervision")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes")
    parser.add_argument("--image_key", type=str, default="img", help="Image key in the dataset")
    parser.add_argument("--class_key", type=str, default="fine_label", help="Image key in the dataset")
    parser.add_argument(
        "--vae_name",
        type=str,
        required=False,
        default=None,
        help="If doing ldm pass path to VAE here, otherwise don't set it to anything."
    )
    parser.add_argument("--input_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--test_split_name", type=str, default="test", help="Name of split to use for testing")
    parser.add_argument("--epoch_max_batches_train", type=int, default=1000, help="Max number of batches per epoch for train")
    parser.add_argument("--epoch_max_batches_eval", type=int, default=250, help="Max number of batches per epoch for eval")
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="relative",
        choices=["absolute", "relative"],
        help="Whether to use absolute (coordinates) or relative (relationships) scene representation."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


# --- Constants aligned with CLEVR ---
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
MATERIALS = ['rubber', 'metal']
SHAPES = ['cube', 'sphere', 'cylinder']
SIZES = ['small', 'large']

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

    # 1. Place objects (Simple rejection sampling for collision)
    positions = [] # Store (x, y) for distance checks

    for _ in range(num_objects):
        valid_pos = False
        attempts = 0
        while not valid_pos and attempts < 100:
            # Random 3D world coordinates
            x = random.uniform(*X_RANGE)
            y = random.uniform(*Y_RANGE)

            # Check collision
            too_close = False
            for px, py in positions:
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                if dist < MIN_DIST:
                    too_close = True
                    break

            if not too_close:
                valid_pos = True
                positions.append((x, y))

                # Create the object dict
                obj = {
                    'color': random.choice(COLORS),
                    'material': random.choice(MATERIALS),
                    'shape': random.choice(SHAPES),
                    'size': random.choice(SIZES),
                    'rotation': random.uniform(0, 360),
                    '3d_coords': [x, y, 0.0], # Z is usually 0 on the floor
                    # Approximate pixel coords (Just for consistency, strictly we need a camera matrix)
                    # Mapping world (x,y) -> image (u,v) roughly:
                    'pixel_coords': [
                        (x + 3) / 6 * 480, # Map -3..3 to 0..480
                        (y + 3) / 6 * 320, # Map -3..3 to 0..320
                        10.0 # Arbitrary depth
                    ]
                }
                objects.append(obj)
            attempts += 1

    # 2. Compute Relationships (Validity comes from Geometry!)
    # In CLEVR:
    # Right: x > x_ref, Left: x < x_ref
    # Front: y < y_ref, Behind: y > y_ref

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

            # X-axis logic
            if pos_B[0] < pos_A[0]: # B is left of A
                relationships['left'][i].append(j)
            else:
                relationships['right'][i].append(j)

            # Y-axis logic
            if pos_B[1] < pos_A[1]: # B is in front of A
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
    Converts a scene dictionary into a model-ready tensor.
    Returns: (1, 10, Feature_Dim)
    """
    mode = scene_dict['mode']
    objects = scene_dict['objects']
    relationships = scene_dict['relationships']

    # Mappings (Must match Dataset class exactly)
    color2id = {k: i for i, k in enumerate(COLORS)}
    mat2id   = {k: i for i, k in enumerate(MATERIALS)}
    shape2id = {k: i for i, k in enumerate(SHAPES)}
    size2id  = {k: i for i, k in enumerate(SIZES)}
    RELATIONS = ['left', 'right', 'front', 'behind']

    MAX_OBJECTS = 10
    obj_vectors = []

    for i, obj in enumerate(objects):
        # --- 1. Common Attributes (15 dims) ---
        # Rotation
        rot_rad = math.radians(obj['rotation'])
        rot_vec = torch.tensor([math.sin(rot_rad), math.cos(rot_rad)])

        # Categorical
        sz_vec = torch.tensor([size2id[obj['size']]])
        mat_vec = torch.tensor([mat2id[obj['material']]])
        sh_vec = F.one_hot(torch.tensor(shape2id[obj['shape']]), num_classes=3)
        col_vec = F.one_hot(torch.tensor(color2id[obj['color']]), num_classes=8)

        base = torch.cat([rot_vec, sz_vec, mat_vec, sh_vec, col_vec])

        # --- 2. Mode Specifics ---
        if mode == "absolute":
            # Pixel (normalized)
            px, py, pz = obj['pixel_coords']
            p_vec = torch.tensor([px/480.0, py/320.0, pz/20.0])

            # 3D (normalized)
            x3, y3, z3 = obj['3d_coords']
            c3_vec = torch.tensor([x3/5.0, y3/5.0, z3/5.0])

            spatial = torch.cat([p_vec, c3_vec])
            full_vec = torch.cat([base, spatial]) # 15 + 6 = 21 dims

        elif mode == "relative":
            # Relationship Grid (40 dims)
            rel_grid = torch.zeros((4, MAX_OBJECTS))

            for r_idx, rel_name in enumerate(RELATIONS):
                target_indices = relationships[rel_name][i]
                for t_idx in target_indices:
                    if t_idx < MAX_OBJECTS:
                        rel_grid[r_idx, t_idx] = 1.0

            rels = rel_grid.flatten()
            full_vec = torch.cat([base, rels]) # 15 + 40 = 55 dims

        obj_vectors.append(full_vec)

    # --- 3. Stacking and Padding ---
    if mode == "absolute":
        dim = 21
    else:
        dim = 55

    padded_objs = torch.zeros((1, MAX_OBJECTS, dim)) # Batch size 1

    # Stack valid objects
    limit = min(len(obj_vectors), MAX_OBJECTS)
    if limit > 0:
        stacked = torch.stack(obj_vectors[:limit])
        padded_objs[0, :limit, :] = stacked

    return padded_objs


def make_tensor_from_scene(scene_dict):
    """
    Converts a scene dictionary into model-ready tensors.

    Returns:
        cond_tensor (Tensor): Shape (1, 10, Feature_Dim)
        mask (Tensor): Shape (1, 10) - 1.0 for real objects, 0.0 for padding
    """
    mode = scene_dict['mode']
    objects = scene_dict['objects']
    relationships = scene_dict['relationships']

    # Mappings (Must match Dataset class exactly)
    COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
    MATERIALS = ['rubber', 'metal']
    SHAPES = ['cube', 'sphere', 'cylinder']
    SIZES = ['small', 'large']
    RELATIONS = ['left', 'right', 'front', 'behind']

    color2id = {k: i for i, k in enumerate(COLORS)}
    mat2id   = {k: i for i, k in enumerate(MATERIALS)}
    shape2id = {k: i for i, k in enumerate(SHAPES)}
    size2id  = {k: i for i, k in enumerate(SIZES)}

    MAX_OBJECTS = 10
    obj_vectors = []

    for i, obj in enumerate(objects):
        # --- 1. Common Attributes (15 dims) ---
        # Rotation
        rot_rad = math.radians(obj['rotation'])
        rot_vec = torch.tensor([math.sin(rot_rad), math.cos(rot_rad)])

        # Categorical
        sz_vec = torch.tensor([size2id[obj['size']]])
        mat_vec = torch.tensor([mat2id[obj['material']]])
        sh_vec = F.one_hot(torch.tensor(shape2id[obj['shape']]), num_classes=3)
        col_vec = F.one_hot(torch.tensor(color2id[obj['color']]), num_classes=8)

        base = torch.cat([rot_vec, sz_vec, mat_vec, sh_vec, col_vec])

        # --- 2. Mode Specifics ---
        if mode == "absolute":
            # Pixel (normalized)
            px, py, pz = obj['pixel_coords']
            p_vec = torch.tensor([px/480.0, py/320.0, pz/20.0])

            # 3D (normalized)
            x3, y3, z3 = obj['3d_coords']
            c3_vec = torch.tensor([x3/5.0, y3/5.0, z3/5.0])

            spatial = torch.cat([p_vec, c3_vec])
            full_vec = torch.cat([base, spatial]) # 15 + 6 = 21 dims

        elif mode == "relative":
            # Relationship Grid (40 dims)
            rel_grid = torch.zeros((4, MAX_OBJECTS))

            for r_idx, rel_name in enumerate(RELATIONS):
                # Ensure the relationship list exists for this object index
                if i < len(relationships[rel_name]):
                    target_indices = relationships[rel_name][i]
                    for t_idx in target_indices:
                        if t_idx < MAX_OBJECTS:
                            rel_grid[r_idx, t_idx] = 1.0

            rels = rel_grid.flatten()
            full_vec = torch.cat([base, rels]) # 15 + 40 = 55 dims

        obj_vectors.append(full_vec)

    # --- 3. Stacking and Padding ---
    if mode == "absolute":
        dim = 21
    else:
        dim = 55

    padded_objs = torch.zeros((1, MAX_OBJECTS, dim)) # Batch size 1
    mask = torch.zeros((1, MAX_OBJECTS))             # Batch size 1

    # Fill tensors
    limit = min(len(obj_vectors), MAX_OBJECTS)
    if limit > 0:
        stacked = torch.stack(obj_vectors[:limit])
        padded_objs[0, :limit, :] = stacked
        mask[0, :limit] = 1.0 # Mark as valid

    return padded_objs, mask


class CLEVRHybridDataset(Dataset):
    URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"

    # Standard CLEVR attributes
    COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
    MATERIALS = ['rubber', 'metal']
    SHAPES = ['cube', 'sphere', 'cylinder']
    SIZES = ['small', 'large']
    RELATIONS = ['left', 'right', 'front', 'behind']

    MAX_OBJECTS = 10

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

        # Mappings
        self.color2id = {k: i for i, k in enumerate(self.COLORS)}
        self.mat2id   = {k: i for i, k in enumerate(self.MATERIALS)}
        self.shape2id = {k: i for i, k in enumerate(self.SHAPES)}
        self.size2id  = {k: i for i, k in enumerate(self.SIZES)}

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

    def encode_common_attributes(self, obj):
        """Encodes attributes common to both modes (Color, Shape, etc.)"""
        # 1. Rotation (Sin/Cos)
        rot_deg = obj['rotation']
        rot_rad = math.radians(rot_deg)
        rot_vec = torch.tensor([math.sin(rot_rad), math.cos(rot_rad)], dtype=torch.float32)

        # 2. Categorical (One-Hot / Binary)
        sz_vec = torch.tensor([self.size2id[obj['size']]], dtype=torch.float32)
        mat_vec = torch.tensor([self.mat2id[obj['material']]], dtype=torch.float32)
        sh_vec = F.one_hot(torch.tensor(self.shape2id[obj['shape']]), num_classes=3).float()
        col_vec = F.one_hot(torch.tensor(self.color2id[obj['color']]), num_classes=8).float()

        # Total Common Dims: 2 + 1 + 1 + 3 + 8 = 15 dims
        return torch.cat([rot_vec, sz_vec, mat_vec, sh_vec, col_vec])

    def get_absolute_features(self, obj):
        """Adds Pixel Coords and 3D Coords"""
        # [x, y, z] pixel (normalized)
        px, py, pz = obj['pixel_coords']
        p_vec = torch.tensor([px/480.0, py/320.0, pz/20.0], dtype=torch.float32)

        # [x, y, z] 3d
        x3, y3, z3 = obj['3d_coords']
        c3_vec = torch.tensor([x3/5.0, y3/5.0, z3/5.0], dtype=torch.float32)

        # 3 + 3 = 6 dims
        return torch.cat([p_vec, c3_vec])

    def get_relative_features(self, obj_index, relationships):
        """
        Creates a 'Relation Fingerprint'.
        We have 4 relations. We have MAX 10 objects.
        This creates a 40-dim vector:
        [ is_left_of_obj0, is_left_of_obj1... is_behind_obj9 ]
        """
        # Shape: [4, 10] flattened to [40]
        rel_grid = torch.zeros((4, self.MAX_OBJECTS), dtype=torch.float32)

        # relationships is a dict: {'left': [[1,2], [], ...], 'right': ...}
        # relationships['left'][obj_index] gives the list of objects to the left of current obj

        for r_idx, rel_name in enumerate(self.RELATIONS):
            target_indices = relationships[rel_name][obj_index]
            for t_idx in target_indices:
                if t_idx < self.MAX_OBJECTS: # Safety check
                    rel_grid[r_idx, t_idx] = 1.0

        return rel_grid.flatten()

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        image = Image.open(os.path.join(self.image_dir, scene['image_filename'])).convert("RGB")
        objects = scene['objects']
        relationships = scene['relationships']

        obj_vectors = []

        for i, obj in enumerate(objects):
            # Base attributes (15 dims)
            base = self.encode_common_attributes(obj)

            if self.mode == "absolute":
                # Add coords (6 dims) -> Total 21
                spatial = self.get_absolute_features(obj)
                full_vec = torch.cat([base, spatial])

            elif self.mode == "relative":
                # Add relationship grid (40 dims) -> Total 55
                rels = self.get_relative_features(i, relationships)
                full_vec = torch.cat([base, rels])

            obj_vectors.append(full_vec)

        dim = 21 if self.mode == "absolute" else 55

        padded_objs = torch.zeros((self.MAX_OBJECTS, dim))
        mask = torch.zeros(self.MAX_OBJECTS)

        limit = min(len(obj_vectors), self.MAX_OBJECTS)
        if limit > 0:
            padded_objs[:limit] = torch.stack(obj_vectors[:limit])
            mask[:limit] = 1.0

        return {
            "img": self.transform(image),
            "obj_features": padded_objs,
            "obj_mask": mask
        }


class CLEVRDiffusionModel(UNet2DConditionModel):
    @register_to_config
    def __init__(
        self,
        raw_dim=21,
        # 2. The Standard UNet Arguments you are customizing
        sample_size=64,
        in_channels=3,
        out_channels=3,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D"
        ),
        block_out_channels=(128, 128, 256, 256, 512, 512),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=512,
        attention_head_dim=8,
        **kwargs,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            **kwargs,
        )

        self.projector = nn.Sequential(
            nn.Linear(raw_dim, cross_attention_dim),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim)
        )

    def forward(self, sample, timestep, raw_objects, obj_mask=None, **kwargs):
        """
        Args:
            sample: The noisy image tensor (Batch, 3, H, W)
            timestep: The current timestep (Batch,)
            raw_objects: Object data (Batch, 10, raw_dim)
            obj_mask: (Optional) Mask for padding (Batch, 10)
        """
        object_embeddings = self.projector(raw_objects)

        return super().forward(
            sample,
            timestep,
            encoder_hidden_states=object_embeddings,
            encoder_attention_mask=obj_mask,
            **kwargs,
        )


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), CLEVRDiffusionModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = CLEVRDiffusionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Initialize the model

    T=args.T
    n=args.n
    N_supervision=args.N_supervision

    sample_size = args.resolution if args.vae_name is None else args.resolution // 8
    in_channels = args.input_channels * 3  # x, y, z - noisy input, pred noise, reasoning token
    out_channels = args.input_channels * 2  # y, z - pred noise, reasoning token
    layers_per_block = 1
    block_out_channels = args.channels
    down_block_types = args.down_block_types
    up_block_types = args.up_block_types

    if args.model_config_name_or_path is None:
        model = CLEVRDiffusionModel(
            raw_dim=21 if args.dataset_mode == "absolute" else 55,
            cross_attention_dim=512,
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        model.y_init = trunc_normal_init_(torch.empty((1, args.input_channels, sample_size, sample_size), dtype=model.dtype), std=1)
        model.z_init = trunc_normal_init_(torch.empty((1, args.input_channels, sample_size, sample_size), dtype=model.dtype), std=1)

        torch.save(model.y_init, os.path.join(args.output_dir, "y_init.pt"))
        torch.save(model.z_init, os.path.join(args.output_dir, "z_init.pt"))
    else:
        config = CLEVRDiffusionModel.load_config(args.model_config_name_or_path)
        model = CLEVRDiffusionModel.from_config(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    vae = None
    if args.vae_name is not None:
        vae = AutoencoderKL.from_pretrained(args.vae_name, cache_dir=args.cache_dir)
        vae.requires_grad_(False)
        vae_scaling_factor = vae.config.scaling_factor
        vae.to(accelerator.device, dtype=torch.float32)
        vae.eval()

    logger.info(f"Total number of parameters: {total_params}")
    logger.info(f"Number of trainable parameters: {trainable_params}")

    if accelerator.is_main_process and args.logger == "wandb":
        accelerator.init_trackers(
            project_name="small-llm-diffusion",
            config={
                "sample_size": sample_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "layers_per_block": layers_per_block,
                "block_out_channels": block_out_channels,
                "down_block_types": down_block_types,
                "up_block_types": up_block_types,
                "learning_rate": args.learning_rate,
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
            init_kwargs={
                "wandb": {
                    "name": args.output_dir
                }
            }
        )

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=CLEVRDiffusionModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #     dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #         split="train",
    #     )
    #     test_dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #         split=args.test_split_name,
    #     )
    # else:
    #     dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
    #     test_dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="test")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    dataset = CLEVRHybridDataset(root_dir=args.train_data_dir, split="train", mode=args.dataset_mode, image_size=args.resolution)
    test_dataset = CLEVRHybridDataset(root_dir=args.train_data_dir, split="validation", mode=args.dataset_mode, image_size=args.resolution)


    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    test_augmentations = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # def transform_images(examples):
    #     images = [augmentations(image.convert("RGB")) for image in examples[args.image_key]]
    #     return {"input": images, "class": examples[args.class_key]}

    # def test_transform_images(examples):
    #     images = [test_augmentations(image.convert("RGB")) for image in examples[args.image_key]]
    #     return {"input": images, "class": examples[args.class_key]}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(augmentations)
    test_dataset.set_transform(test_augmentations)
    train_dataloader = LimitedLoader(
        torch.utils.data.DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True
        ),
        limit_batches=args.epoch_max_batches_train,
    )
    test_dataloader = LimitedLoader(
        torch.utils.data.DataLoader(
            test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.dataloader_num_workers, drop_last=True
        ),
        limit_batches=args.epoch_max_batches_eval,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * N_supervision,
        num_training_steps=(len(train_dataloader) * args.num_epochs * N_supervision),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in SafeIterator(enumerate(train_dataloader), logger=logger):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["img"]
            cond = batch["obj_features"].to(model.device)
            mask = batch["obj_mask"].to(model.device)

            if vae is not None:
                clean_images = clean_images.to(device=accelerator.device, dtype=vae.dtype)
                with torch.no_grad():
                    dist = vae.encode(clean_images).latent_dist
                    latents = dist.sample()
                    clean_images = latents * vae_scaling_factor
                    clean_images = clean_images.to(device=accelerator.device, dtype=weight_dtype)
            else:
                clean_images = clean_images.to(device=accelerator.device, dtype=weight_dtype)


            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            with accelerator.accumulate(model):
                model.old_forward = model.forward
                def latent_recursion(model, x, y, z, timesteps, cond, mask, n=6):
                    for _ in range(n):
                        _, z = model.old_forward(
                            torch.cat([x, y, z], dim=1),
                            timesteps,
                            raw_objects=cond,
                            obj_mask=mask,
                        ).sample.chunk(2, dim=1)
                    y, _ = model.old_forward(
                        torch.cat([x, y, z], dim=1),
                        timesteps,
                        raw_objects=cond,
                        obj_mask=mask,
                    ).sample.chunk(2, dim=1)
                    return y, z

                def deep_recursion(model, x, y, z, timesteps, cond, mask, n=6, T=3):
                    # x = x[:, :args.input_channels]
                    with torch.no_grad():
                        for _ in range(T - 1):
                            y, z = latent_recursion(model, x, y, z, timesteps, cond, mask, n)
                    y, z = latent_recursion(model, x, y, z, timesteps, cond, mask, n)
                    return y, y.detach(), z.detach()
                # Predict the noise residual
                # y, z = model.module.get_init_y_z(args.train_batch_size)

                y = torch.cat([model.module.y_init for _ in range(args.train_batch_size)], dim=0).to(model.device)
                z = torch.cat([model.module.z_init for _ in range(args.train_batch_size)], dim=0).to(model.device)
                for _ in range(N_supervision):

                    # model_output, y, z = model(noisy_images, timesteps, y=y, z=z)
                    model_output, y, z = deep_recursion(model, noisy_images, y, z, timesteps, cond, mask, n, T)

                    if args.prediction_type == "epsilon":
                        loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif args.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"train/loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        model.eval()
        progress_bar = tqdm(total=len(test_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Eval epoch {epoch}")
        for step, batch in SafeIterator(enumerate(test_dataloader), logger=logger):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["img"]
            cond = batch["obj_features"].to(model.device)
            mask = batch["obj_mask"].to(model.device)

            if vae is not None:
                clean_images = clean_images.to(device=accelerator.device, dtype=vae.dtype)
                with torch.no_grad():
                    dist = vae.encode(clean_images).latent_dist
                    latents = dist.sample()
                    clean_images = latents * vae_scaling_factor
                    clean_images = clean_images.to(device=accelerator.device, dtype=weight_dtype)
            else:
                clean_images = clean_images.to(device=accelerator.device, dtype=weight_dtype)

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                # y, z = model.module.get_init_y_z(args.train_batch_size)

                y = torch.cat([model.module.y_init for _ in range(args.train_batch_size)], dim=0).to(model.device)
                z = torch.cat([model.module.z_init for _ in range(args.train_batch_size)], dim=0).to(model.device)
                for _ in range(N_supervision):

                    # model_output, y, z = model(noisy_images, timesteps, y=y, z=z)
                    with torch.no_grad():
                        model_output, y, z = deep_recursion(model, noisy_images, y, z, timesteps, cond, mask, n, T)

                    if args.prediction_type == "epsilon":
                        loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    elif args.prediction_type == "sample":
                        alpha_t = _extract_into_tensor(
                            noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        # use SNR weighting from distillation paper
                        loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

            logs = {"val/loss": loss.detach().item()}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                class OutputUnet:
                    pass
                def new_forward(sample, timesteps, *args, **kwargs):
                    y = torch.cat([model.module.y_init for _ in range(sample.shape[0])], dim=0).to(unet.device)
                    z = torch.cat([model.module.z_init for _ in range(sample.shape[0])], dim=0).to(unet.device)
                    for _ in range(N_supervision):
                        model_output, y, z = deep_recursion(unet, sample, y, z, timesteps, kwargs["raw_objects"], kwargs["obj_mask"], n, T)
                    output = OutputUnet()
                    output.sample = model_output
                    return output
                unet.old_forward = unet.forward
                unet.forward = new_forward
                unet.config.in_channels = args.input_channels

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)

                conds = []
                masks = []
                scene_dict = sample_random_scene(num_objects=4, mode=args.dataset_mode)
                # scene_dict = {
                #     "objects": [
                #         {"color": "gray", "shape": "cube", "size": "small", "material": "rubber", "rotation": 0, "3d_coords": [0, 0, 0]},
                #         {"color": "red", "shape": "sphere", "size": "small", "material": "rubber", "rotation": 0, "3d_coords": [0, 0, 0]},
                #         {"color": "blue", "shape": "cylinder", "size": "large", "material": "metal", "rotation": 0, "3d_coords": [0, 0, 0]},
                #         {"color": "green", "shape": "cylinder", "size": "large", "material": "metal", "rotation": 0, "3d_coords": [0, 0, 0]},
                #     ],
                #     "relationships": {
                #         "left": [[1,2,3], [3], [1,3], []],
                #         "right": [[], [0, 2], [0], [0,1,2]],
                #         "front": [[1,2,3], [2,3], [3], []],
                #         "behind": [[], [0], [0,1], [0,1,2]],
                #     },
                #     "mode": "relative"
                # }
                for _ in range(args.eval_batch_size):
                    cond, mask = make_tensor_from_scene(scene_dict)
                    conds.append(cond)
                    masks.append(mask)
                cond_tensor = torch.cat(conds, dim=0).to(pipeline.device)
                mask = torch.cat(masks, dim=0).to(pipeline.device)

                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="pt",
                    raw_objects=cond_tensor,
                    obj_mask=mask,
                ).images
                if vae is not None:
                    latents = images / vae_scaling_factor
                    latents = latents.to(torch.float32)

                    with torch.no_grad():
                        images = vae.decode(latents).sample
                images = (images / 2 + 0.5).clamp(0, 1).cpu().float()

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                unet.forward = unet.old_forward
                unet.config.in_channels = args.input_channels * 3

                # denormalize the images and save to tensorboard
                # images_processed = (images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    n_images = len(images)
                    n_cols = math.ceil(math.sqrt(n_images) * 1.5)
                    image_grid = make_grid(images, nrow=n_cols, padding=2, normalize=True)
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": wandb.Image(image_grid), "epoch": epoch},
                        step=global_step,
                    )
                del pipeline
                del images
                if 'latents' in locals():
                    del latents
                torch.cuda.empty_cache()

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
