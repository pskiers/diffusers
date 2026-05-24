from torch.utils.data import Dataset, Sampler
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import joblib
import torch
import functools
from typing import List, Dict, Optional
import pkg_resources
from PIL import Image

from examples.trm_diffusion.sokoban.dataset.fields_states import FieldStates


class SokobanDatasetTokens(Dataset):
    def __init__(
        self,
        mode: str,
        data_path: str,
        total_dataset_size: int,
        lazy: bool,
        samples: Optional[List] = None,
        file_list: Optional[List[str]] = None
    ):
        self.mode = mode
        self.data_path = data_path
        self.total_dataset_size = total_dataset_size
        self.lazy = lazy

        self.file_list = file_list if file_list else []
        self.num_files = len(self.file_list)
        self.samples = samples if samples is not None else []

        self._surface_cache = {}

    @functools.lru_cache(maxsize=16)
    def _lazy_load(self, file_path: str) -> dict:
        """Prevents I/O trashing"""
        return joblib.load(file_path)

    @staticmethod
    def _adjust_box_count(board: np.ndarray, target_boxes: int) -> np.ndarray:
        """Bezpiecznie redukuje liczbę skrzynek i celów uwzględniając ich stany."""
        board_mod = board.copy()

        # BOXES
        box_coords = np.argwhere(
            (board_mod == FieldStates.BOX.id) |
            (board_mod == FieldStates.BOX_ON_TARGET.id)
        )
        if len(box_coords) > target_boxes:
            indices_to_remove = np.random.choice(len(box_coords), len(box_coords) - target_boxes, replace=False)
            for idx in indices_to_remove:
                r, c = box_coords[idx]
                if board_mod[r, c] == FieldStates.BOX.id:
                    board_mod[r, c] = FieldStates.FLOOR.id
                elif board_mod[r, c] == FieldStates.BOX_ON_TARGET.id:   # target stays, possibly deleted in targets reduction
                    board_mod[r, c] = FieldStates.BOX_TARGET.id

        # TARGETS
        total_targets = np.sum(
            (board_mod == FieldStates.BOX_TARGET.id) |
            (board_mod == FieldStates.BOX_ON_TARGET.id) |
            (board_mod == FieldStates.PLAYER_ON_TARGET.id)
        )
        if total_targets > target_boxes:
            removable_target_coords = np.argwhere(              # only delete where there is no bo on target
                (board_mod == FieldStates.BOX_TARGET.id) |
                (board_mod == FieldStates.PLAYER_ON_TARGET.id)
            )

            targets_to_remove = total_targets - target_boxes
            indices_to_remove = np.random.choice(len(removable_target_coords), int(targets_to_remove), replace=False)

            for idx in indices_to_remove:
                r, c = removable_target_coords[idx]
                if board_mod[r, c] == FieldStates.BOX_TARGET.id:
                    board_mod[r, c] = FieldStates.FLOOR.id
                elif board_mod[r, c] == FieldStates.PLAYER_ON_TARGET.id:
                    board_mod[r, c] = FieldStates.PLAYER.id

        return board_mod

    @classmethod
    def for_unconditional_generation(cls, data_path: str, total_dataset_size: int, lazy: bool = True):
        data_dir_files = glob(os.path.join(data_path, "*"))
        if not data_dir_files:
            raise ValueError(f"No files in {data_path}")

        if lazy:
            return cls("unconditional", data_path, total_dataset_size, lazy=True, file_list=data_dir_files)

        samples = []
        file_idx = 0
        with tqdm(total=total_dataset_size, desc="Ładowanie Unconditional (Eager)") as pbar:
            while len(samples) < total_dataset_size:
                f = data_dir_files[file_idx % len(data_dir_files)]
                data: dict = joblib.load(f)

                for traj in data.values():
                    if len(samples) >= total_dataset_size:
                        break
                    traj_np = np.argmax(traj, axis=3).astype(np.uint8)
                    state_idx = np.random.randint(0, len(traj_np))
                    board = traj_np[state_idx]

                    num_boxes = (len(samples) % 4) + 1
                    samples.append({"target": cls._adjust_box_count(board, num_boxes)})
                    pbar.update(1)

                file_idx += 1
        return cls("unconditional", data_path, total_dataset_size, lazy=False, samples=samples)

    @classmethod
    def for_conditional_generation(cls, data_path: str, total_dataset_size: int, max_k: int = 10, lazy: bool = True):
        data_dir_files = glob(os.path.join(data_path, "*"))
        samples = []

        for f in tqdm(data_dir_files, desc=f"Indeksowanie Conditional Gen (Lazy: {lazy})"):
            data: dict = joblib.load(f)

            for traj_key, trajectory in data.items():
                traj_len = len(trajectory)
                for start_idx in range(traj_len):
                    if len(samples) >= total_dataset_size:
                        return cls("conditional_gen", data_path, len(samples), lazy=lazy, samples=samples)

                    for k in range(1, max_k):
                        target_idx = start_idx + k
                        if target_idx < traj_len:
                            if lazy:
                                samples.append({
                                    "file": f,
                                    "traj_key": traj_key,
                                    "start": start_idx,
                                    "target": target_idx,
                                    "k": k
                                })
                            else:
                                traj_np = np.argmax(trajectory, axis=3).astype(np.uint8)
                                samples.append({
                                    "condition": traj_np[start_idx],
                                    "target": traj_np[target_idx],
                                    "k": k
                                })

        return cls("conditional_gen", data_path, len(samples), lazy=lazy, samples=samples)

    @classmethod
    def for_conditional_trm(cls, data_path: str, total_dataset_size: int, lazy: bool = False):
        data_dir_files = glob(os.path.join(data_path, "*"))
        samples = []
        seen_hashes = set()

        for f in tqdm(data_dir_files, desc=f"Indeksowanie Conditional TRM (Lazy: {lazy})"):
            data: dict = joblib.load(f)

            for traj_key, trajectory in data.items():
                traj_np = np.argmax(trajectory, axis=3).astype(np.uint8)

                for step_idx, board in enumerate(traj_np):
                    if len(samples) >= total_dataset_size:
                        return cls("conditional_trm", data_path, len(samples), lazy=lazy, samples=samples)

                    board_bytes = board.tobytes()
                    if board_bytes not in seen_hashes:
                        seen_hashes.add(board_bytes)

                        if lazy:
                            samples.append({
                                "file": f,
                                "traj_key": traj_key,
                                "step": step_idx
                            })
                        else:
                            samples.append({"target": board})

        return cls("conditional_trm", data_path, len(samples), lazy=lazy, samples=samples)

    def __len__(self):
        return self.total_dataset_size

    def __getitem__(self, index):
        if self.mode == "unconditional":
            num_boxes = (index % 4) + 1

            if self.lazy:
                file_path = self.file_list[(index // 100) % self.num_files]
                data = self._lazy_load(file_path)

                keys = list(data.keys())
                traj_key = keys[index % len(keys)]
                traj_np = np.argmax(data[traj_key], axis=3).astype(np.uint8)

                state_idx = torch.randint(0, len(traj_np), (1,)).item()
                board = traj_np[state_idx]

                board = self._adjust_box_count(board, num_boxes)
                return {"target": torch.from_numpy(board).long()}
            else:
                return {"target": torch.from_numpy(self.samples[index]["target"]).long()}

        elif self.mode == "conditional_gen":
            if self.lazy:
                ptr = self.samples[index]
                data = self._lazy_load(ptr["file"])
                traj_np = np.argmax(data[ptr["traj_key"]], axis=3).astype(np.uint8)

                return {
                    "condition": torch.from_numpy(traj_np[ptr["start"]]).long(),
                    "target": torch.from_numpy(traj_np[ptr["target"]]).long(),
                    "k": torch.tensor(ptr["k"], dtype=torch.long)
                }
            else:
                sample = self.samples[index]
                return {
                    "condition": torch.from_numpy(sample["condition"]).long(),
                    "target": torch.from_numpy(sample["target"]).long(),
                    "k": torch.tensor(sample["k"], dtype=torch.long)
                }

        elif self.mode == "conditional_trm":
            if self.lazy:
                ptr = self.samples[index]
                data = self._lazy_load(ptr["file"])
                traj_np = np.argmax(data[ptr["traj_key"]], axis=3).astype(np.uint8)
                board = traj_np[ptr["step"]]

                return {"target": torch.from_numpy(board).long()}
            else:
                return {"target": torch.from_numpy(self.samples[index]["target"]).long()}

        else:
            raise ValueError("Wrong mode - it must be 'unconditional', 'conditional_gen' or 'conditional_trm'")

    def render_board(self, board_tokens):
        """Converts matrix 12x12 with 0-6 fields to 0-1 image ready to display"""
        w, h = board_tokens.shape
        render_surface = self._load_surface(board_tokens.shape)
        res = np.empty((w**2, h**2, 3))
        for i in range(w):
            for j in range(h):
                res[i * w : (i + 1) * w, j * h : (j + 1) * h] = render_surface[x[i, j] % len(render_surface)]
        return res

    def _load_surface(self, shape: tuple[int, int]):
        if shape in self._surface_cache:
            return self._surface_cache[shape]

        asset_file_names = [field_state.asset_file_name for field_state in FieldStates]
        resource_package = __name__
        surface = []
        for asset_file_name in asset_file_names:
            asset_path = pkg_resources.resource_filename(resource_package, "/".join(("surface", asset_file_name)))  # type: ignore
            asset_np_array = np.array(Image.open(asset_path).convert("RGB").resize(shape))
            surface.append(asset_np_array)

        self._surface_cache[shape] = np.stack(surface)
        return self._surface_cache[shape]


class SokobanBitsDataset(SokobanDatasetTokens):
    """Subclass for Bit-Diffusion. Converts discrete token matrices into continuous num_bits-channel bit tensors.
    For example: token 3 (BOX_TARGET) -> [0, 1, 1]
    """
    def __init__(self, mode: str, data_path: str, total_dataset_size: int, lazy: bool, num_bits: int = 3, **kwargs):
        super().__init__(mode=mode, data_path=data_path, total_dataset_size=total_dataset_size, lazy=lazy, **kwargs)
        self.num_bits = num_bits

    def __getitem__(self, index):
        sample = super().__getitem__(index) # Get the discrete tokens from the parent class

        res = {}
        for key, val in sample.items():
            if key in ["target", "condition"]:
                mask = 2 ** torch.arange(self.num_bits - 1, -1, -1).to(val.device)
                bits = val.unsqueeze(-1).bitwise_and(mask).ne(0).float()
                bits = bits * 2.0 - 1.0 # {0, 1} to {-1.0, 1.0}
                res[key] = bits.permute(2, 0, 1) # [num_bits, 12, 12]
            else:
                res[key] = val  # leave k as it is

        return res

    def render_bit_boards(self, board_bits: torch.Tensor):
        """Gets a [num_bits x 12 x 12] matrix of noisy floats and returns a valid integer board.
        Quantize continuous floats (e.g., -0.2, 0.8) back to hard discrete bits {0, 1}. Anything > 0 becomes 1, anything <= 0 becomes 0
        """
        binary_bits = (board_bits > 0).long()

        if binary_bits.dim() == 3:
            binary_bits = binary_bits.permute(1, 2, 0)  # from [num_bits, 12, 12] back to [12, 12, num_bits]

        num_bits = binary_bits.shape[-1]

        mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(binary_bits.device)
        board_tokens = (binary_bits * mask).sum(dim=-1).long()

        return self.render_board(board_tokens)
