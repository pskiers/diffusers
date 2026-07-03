import os
from glob import glob
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from sokoban.dataset.fields_states import FieldStates


class SokobanDatasetTokens(Dataset):
    def __init__(
        self,
        mode: str,
        data_path: str,
        total_dataset_size: int,
        samples: Optional[List] = None,
        bot_removal_prob: float = 0.75,
    ):
        self.mode = mode
        self.data_path = data_path
        self.total_dataset_size = total_dataset_size
        self.bot_removal_prob = bot_removal_prob
        self.samples = samples if samples is not None else []

        self._surface_cache = {}

    @staticmethod
    def _adjust_box_count(board: np.ndarray, target_boxes: int) -> np.ndarray:
        """Remove solved (BOX_ON_TARGET) pairs to reduce box count.

        Only removes BOX_ON_TARGET cells — this preserves solvability because
        a solved box+target pair is no longer part of the remaining puzzle.
        The board must have enough BOX_ON_TARGET cells to reach target_boxes.
        """
        board_mod = board.copy()
        current_boxes = int(np.sum(
            (board_mod == FieldStates.BOX.id) | (board_mod == FieldStates.BOX_ON_TARGET.id)
        ))
        to_remove = current_boxes - target_boxes
        if to_remove <= 0:
            return board_mod

        bot_coords = np.argwhere(board_mod == FieldStates.BOX_ON_TARGET.id)
        if len(bot_coords) < to_remove:
            return board_mod  # not enough solved pairs to reduce safely

        indices = np.random.choice(len(bot_coords), to_remove, replace=False)
        for idx in indices:
            r, c = bot_coords[idx]
            board_mod[r, c] = FieldStates.FLOOR.id  # remove both box and target

        return board_mod

    @classmethod
    def for_conditioning_num_boxes_generation(cls, data_path: str, total_dataset_size: int):
        data_dir_files = glob(os.path.join(data_path, "*"))
        if not data_dir_files:
            raise ValueError(f"No files in {data_path}")

        candidates_by_bot: dict = {0: [], 1: [], 2: [], 3: [], 4: []}
        max_candidates = total_dataset_size * 4
        total_loaded = 0
        with tqdm(total=max_candidates, desc="Loading Num Boxes candidates") as pbar:
            for f in data_dir_files:
                if total_loaded >= max_candidates:
                    break
                data: dict = joblib.load(f)
                for traj in data.values():
                    if total_loaded >= max_candidates:
                        break
                    traj_np = np.argmax(traj, axis=3).astype(np.uint8)
                    for board in traj_np:
                        bot_count = int(np.sum(board == FieldStates.BOX_ON_TARGET.id))
                        candidates_by_bot[bot_count].append(board.copy())
                        total_loaded += 1
                        pbar.update(1)
                        if total_loaded >= max_candidates:
                            break

        samples = []
        with tqdm(total=total_dataset_size, desc="Building Num Boxes Conditional") as pbar:
            while len(samples) < total_dataset_size:
                num_boxes = (len(samples) % 4) + 1
                required_bot = 4 - num_boxes

                pool = []
                for bot_count in range(required_bot, 5):
                    pool.extend(candidates_by_bot[bot_count])

                if not pool:
                    break

                board = pool[np.random.randint(len(pool))]
                samples.append({"target": cls._adjust_box_count(board, num_boxes), "num_boxes": num_boxes})
                pbar.update(1)

        return cls("conditional_num_boxes", data_path, len(samples), samples=samples)

    @classmethod
    def for_conditioning_k_steps_generation(cls, data_path: str, total_dataset_size: int, k_values: List[int], bot_removal_prob: float = 0.75):
        data_dir_files = glob(os.path.join(data_path, "*"))
        samples = []

        with tqdm(total=total_dataset_size, desc="Loading K Steps") as pbar:
            for f in data_dir_files:
                data: dict = joblib.load(f)

                for traj_key, trajectory in data.items():
                    traj_np = np.argmax(trajectory, axis=3).astype(np.uint8)
                    traj_len = len(traj_np)

                    for start_idx in range(traj_len):
                        if len(samples) >= total_dataset_size:
                            return cls("conditional_k_steps", data_path, len(samples), samples=samples,
                                       bot_removal_prob=bot_removal_prob)

                        for k in k_values:
                            target_idx = start_idx + k
                            if target_idx < traj_len:
                                samples.append({
                                    "condition": traj_np[start_idx].copy(),
                                    "target": traj_np[target_idx].copy(),
                                    "k": k,
                                })
                                pbar.update(1)

        return cls("conditional_k_steps", data_path, len(samples), samples=samples,
                   bot_removal_prob=bot_removal_prob)

    @classmethod
    def for_new_k_steps_conditioning_generation(cls, data_path: str, total_dataset_size: int, k_values: List[int], bot_removal_prob: float = 0.75, seed=None):
        data_dir_files = glob(os.path.join(data_path, "*"))
        if not data_dir_files:
            raise ValueError(f"No files in {data_path}")

        candidates_by_k = {k: [] for k in k_values}
        trajectories = []

        with tqdm(desc="Loading K-step candidates [random boards version]") as pbar:
            for f in data_dir_files:
                data: dict = joblib.load(f)

                for trajectory in data.values():
                    traj_np = np.argmax(trajectory, axis=3).astype(np.uint8)
                    traj_len = len(traj_np)

                    traj_id = len(trajectories)
                    trajectories.append(traj_np)

                    for start_idx in range(traj_len):
                        for k in k_values:
                            if start_idx + k < traj_len:
                                candidates_by_k[k].append((traj_id, start_idx))

                        pbar.update(1)

        available_k = [k for k, pool in candidates_by_k.items() if len(pool) > 0]
        if not available_k:
            raise ValueError("No valid (condition, target) pairs found")

        samples = []

        with tqdm(total=total_dataset_size, desc="Building K-step Conditional") as pbar:
            while len(samples) < total_dataset_size:
                k = np.random.choice(available_k)
                traj_id, start_idx = candidates_by_k[k][
                    np.random.randint(len(candidates_by_k[k]))
                ]
                traj_np = trajectories[traj_id]
                samples.append(
                    {
                        "condition": traj_np[start_idx].copy(),
                        "target": traj_np[start_idx + k].copy(),
                        "k": int(k),
                    }
                )
                pbar.update(1)

        return cls("conditional_k_steps", data_path, len(samples), samples=samples, bot_removal_prob=bot_removal_prob)

    @classmethod
    def for_unconditional(cls, data_path: str, total_dataset_size: int, bot_removal_prob: float = 0.75):
        data_dir_files = glob(os.path.join(data_path, "*"))
        if not data_dir_files:
            raise ValueError(f"No files in {data_path}")

        # Load all candidate boards from trajectories
        candidates = []
        max_candidates = total_dataset_size * 4
        with tqdm(total=max_candidates, desc="Loading Unconditional candidates") as pbar:
            for f in data_dir_files:
                if len(candidates) >= max_candidates:
                    break
                data: dict = joblib.load(f)
                for traj in data.values():
                    if len(candidates) >= max_candidates:
                        break
                    traj_np = np.argmax(traj, axis=3).astype(np.uint8)
                    for board in traj_np:
                        candidates.append(board.copy())
                        pbar.update(1)
                        if len(candidates) >= max_candidates:
                            break

        # Sample randomly from candidates
        samples = []
        indices = np.random.choice(len(candidates), total_dataset_size, replace=len(candidates) < total_dataset_size)
        for idx in indices:
            samples.append({"target": candidates[idx]})

        return cls("unconditional", data_path, len(samples), samples=samples,
                   bot_removal_prob=bot_removal_prob)

    def __len__(self):
        return self.total_dataset_size

    def __getitem__(self, index):
        if self.mode == "conditional_num_boxes":
            num_boxes = (index % 4) + 1
            return {"target": torch.from_numpy(self.samples[index]["target"]).long(), "num_boxes": self.samples[index]["num_boxes"]}

        elif self.mode == "conditional_k_steps":
            sample = self.samples[index]
            cond = sample["condition"].copy()
            tgt = sample["target"].copy()
            k = sample["k"]

            # Randomly remove each common BOX_ON_TARGET with probability p
            common_bot = (cond == FieldStates.BOX_ON_TARGET.id) & (tgt == FieldStates.BOX_ON_TARGET.id)
            if common_bot.any():
                remove = np.random.random(common_bot.sum()) < self.bot_removal_prob
                bot_coords = np.argwhere(common_bot)
                # Ensure at least 1 box remains in condition after removal
                num_non_bot_boxes_cond = int(np.sum(
                    (cond == FieldStates.BOX.id) | ((cond == FieldStates.BOX_ON_TARGET.id) & ~common_bot)
                ))
                max_removable = common_bot.sum() - (1 if num_non_bot_boxes_cond == 0 else 0)
                if remove.sum() > max_removable:
                    true_indices = np.where(remove)[0]
                    keep_idx = np.random.choice(true_indices)
                    remove[keep_idx] = False
                for coord in bot_coords[remove]:
                    cond[coord[0], coord[1]] = FieldStates.FLOOR.id
                    tgt[coord[0], coord[1]] = FieldStates.FLOOR.id

            num_boxes = int(np.sum(
                (cond == FieldStates.BOX.id) | (cond == FieldStates.BOX_ON_TARGET.id)
            ))

            return {
                "condition": torch.from_numpy(cond).long(),
                "target": torch.from_numpy(tgt).long(),
                "k": torch.tensor(k, dtype=torch.long),
                "num_boxes": num_boxes,
            }

        elif self.mode == "unconditional":
            board = self.samples[index]["target"].copy()

            # Randomly remove each BOX_ON_TARGET with probability p
            bot_mask = board == FieldStates.BOX_ON_TARGET.id
            if bot_mask.any():
                remove = np.random.random(bot_mask.sum()) < self.bot_removal_prob
                bot_coords = np.argwhere(bot_mask)
                # Ensure at least 1 box remains after removal
                num_non_bot_boxes = int(np.sum(board == FieldStates.BOX.id))
                max_removable = bot_mask.sum() - (1 if num_non_bot_boxes == 0 else 0)
                if remove.sum() > max_removable:
                    true_indices = np.where(remove)[0]
                    keep_idx = np.random.choice(true_indices)
                    remove[keep_idx] = False
                for coord in bot_coords[remove]:
                    board[coord[0], coord[1]] = FieldStates.FLOOR.id

            num_boxes = int(np.sum(
                (board == FieldStates.BOX.id) | (board == FieldStates.BOX_ON_TARGET.id)
            ))

            return {
                "target": torch.from_numpy(board).long(),
                "num_boxes": num_boxes,
            }

        else:
            raise ValueError("Wrong mode - it must be 'conditional_num_boxes', 'conditional_k_steps' or 'unconditional'")

    def render_board(self, board_tokens):
        """Converts matrix 12x12 with 0-6 fields to RGB image."""
        w, h = board_tokens.shape
        render_surface = self._load_surface(board_tokens.shape)
        res = np.empty((w**2, h**2, 3), dtype=np.uint8)
        for i in range(w):
            for j in range(h):
                res[i * w : (i + 1) * w, j * h : (j + 1) * h] = render_surface[board_tokens[i, j] % len(render_surface)]
        return res

    def _load_surface(self, shape: tuple[int, int]):
        if shape in self._surface_cache:
            return self._surface_cache[shape]

        surface_dir = Path(__file__).resolve().parent.parent / "data" / "surface"
        asset_file_names = [field_state.asset_file_name for field_state in FieldStates]
        surface = []
        for asset_file_name in asset_file_names:
            asset_path = surface_dir / asset_file_name
            asset_np_array = np.array(Image.open(asset_path).convert("RGB").resize(shape))
            surface.append(asset_np_array)

        self._surface_cache[shape] = np.stack(surface)
        return self._surface_cache[shape]


class SokobanBitsDataset(SokobanDatasetTokens):
    """Subclass for Bit-Diffusion. Converts discrete token matrices into continuous num_bits-channel bit tensors. For example: token 3 (BOX_TARGET) -> [0, 1, 1]

    Supports dihedral augmentation (D4 group: 4 rotations × 2 flips = 8 transforms).
    When use_dihedral_aug=True:
      - Dataset is virtually expanded 8× (len multiplied by 8)
      - Each index maps to (base_sample, transform_id) deterministically
      - Useful for TRM small-dataset regime (e.g. 1000 base samples × 8 = 8000)
    """
    N_DIHEDRAL = 8

    def __init__(self,
                 mode: str,
                 data_path: str,
                 total_dataset_size: int,
                 num_bits: int = 3,
                 use_dihedral_aug: bool = False, **kwargs):
        super().__init__(mode=mode, data_path=data_path, total_dataset_size=total_dataset_size, **kwargs)
        self.num_bits = num_bits
        self.use_dihedral_aug = use_dihedral_aug

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        if self.use_dihedral_aug:
            transform_id: int = int(torch.randint(0, self.N_DIHEDRAL, (1,)).item())
        else:
            transform_id = 0

        res = {}
        for key, val in sample.items():
            if key in ["target", "condition"]:
                mask = 2 ** torch.arange(self.num_bits - 1, -1, -1).to(val.device)
                bits = val.unsqueeze(-1).bitwise_and(mask).ne(0).float()
                bits = bits * 2.0 - 1.0  # {0, 1} to {-1.0, 1.0}
                bits = bits.permute(2, 0, 1)  # [num_bits, 12, 12]
                if self.use_dihedral_aug:
                    bits = self._apply_dihedral(bits, transform_id)
                res[key] = bits
            else:
                res[key] = val  # leave k, num_boxes as-is

        return res

    def _apply_dihedral(self, tensor: torch.Tensor, transform_id: int) -> torch.Tensor:
        """Apply one of 8 dihedral transforms to a (C, H, W) tensor.

        transform_id 0-3: rot90 by k=0,1,2,3
        transform_id 4-7: horizontal flip + rot90 by k=0,1,2,3
        """
        if transform_id >= 4:
            tensor = tensor.flip(-1)  # horizontal flip
            transform_id -= 4
        if transform_id > 0:
            tensor = torch.rot90(tensor, k=transform_id, dims=(-2, -1))
        return tensor

    @staticmethod
    def bits_to_tokens(board_bits: torch.Tensor) -> torch.Tensor:
        """Convert bit tensor to integer token board. Thresholds floats at 0: >0 → 1, ≤0 → 0, then converts binary to int.
        """
        binary_bits = (board_bits > 0).long()

        if binary_bits.dim() == 3:  # single: (num_bits, H, W)
            num_bits = binary_bits.shape[0]
            mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=binary_bits.device, dtype=torch.long)
            return (binary_bits * mask.view(num_bits, 1, 1)).sum(dim=0)
        elif binary_bits.dim() == 4:  # batch: (B, num_bits, H, W)
            num_bits = binary_bits.shape[1]
            mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=binary_bits.device, dtype=torch.long)
            return (binary_bits * mask.view(1, num_bits, 1, 1)).sum(dim=1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {binary_bits.dim()}D")

    def render_bit_boards(self, board_bits: torch.Tensor):
        """Render a single (num_bits, H, W) bit tensor to RGB image."""
        board_tokens = self.bits_to_tokens(board_bits).numpy()
        return self.render_board(board_tokens)
