from omegaconf import ListConfig
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Union, Optional, List
import random
from tqdm import tqdm
import joblib
import os
from glob import glob


class SokobanBitDataset(Dataset):
    def __init__(self, sokoban_dataset, num_bits, clip_sample_range=1.0, weight_dtype=torch.float32, device="cuda"):
        self.dataset = sokoban_dataset
        self.num_bits = num_bits
        self.clip_sample_range = clip_sample_range
        self.weight_dtype = weight_dtype
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        result = {}

        result["images"] = self._board_to_tensor(item["target"])

        if "state" in item:
            result["conditions"] = self._board_to_tensor(item["state"])

        if "distance_label" in item and item["distance_label"] is not None:
            result["class_labels"] = item["distance_label"]

        return result

    def _board_to_tensor(self, board_np):
        board_tensor = torch.from_numpy(board_np).unsqueeze(0).to(self.device)
        images = self._int2bits(board_tensor, self.num_bits, self.device, self.weight_dtype)
        images = (images * 2 - 1.0) * self.clip_sample_range
        images = images.squeeze(0).permute(2, 0, 1)  #(num_bits, H, W)
        images = images.to(self.weight_dtype)
        return images

    def _int2bits(self, x, n, device, out_dtype=None):
        """Convert an integer x in (...) into bits in (..., n)."""
        x = torch.bitwise_right_shift(torch.unsqueeze(x, -1), torch.arange(n).to(device))
        x = torch.remainder(x, 2)
        if out_dtype and out_dtype != x.dtype:
            x = x.to(out_dtype)
        return x


class SokobanDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        encoding: str = "bits",
        k: Optional[Union[int, List[int]]] = None,
        max_trajectories: Optional[int] = None,
    ):
        self.data_path = data_path
        if encoding != "bits":
            raise ValueError("Zoptymalizowany dataset wspiera wyłącznie kodowanie 'bits'.")
        self.encoding = encoding

        if isinstance(k, (tuple, list, ListConfig)):
            k = list(k)
            if len(k) > 1:
                self.k_label = {k_val: i for i, k_val in enumerate(k)}
            else:
                k = k[0]
        self.k = k
        self.max_trajectories = max_trajectories

        self.boards, self.trajectory_start_idx, self.trajectory_length = self._load_boards(data_path)

    def __len__(self) -> int:
        if isinstance(self.k, list):
            return min(len(indices) for indices in self.valid_indices.values())
        return len(self.valid_indices[self.k])

    @property
    def num_trajectories(self) -> int:
        return len(set(self.trajectory_start_idx))

    def __getitem__(self, idx: int) -> dict:
        if self.k is None or self.k == 0:
            return {"target": self.boards[idx]}

        # If k is a list, sample a random value from the list
        k = random.choice(self.k) if isinstance(self.k, list) else self.k
        k_label = self.k_label[k] if isinstance(self.k, list) else None

        board_idx = self.valid_indices[k][idx]
        board = self.boards[board_idx]

        target = self.boards[board_idx + k]

        trajectory_idx = self.trajectory_start_idx[board_idx]
        current_step = board_idx - trajectory_idx

        return {
            "state": board,
            "target": target,
            "distance": k,
            "distance_label": k_label,
            "trajectory_timestep": current_step,  # Absolute time step in trajectory
        }

    def _load_boards(self, data_path: str) -> tuple[np.ndarray, list[int], list[int]]:
        boards = []
        trajectory_start_idx = []
        trajectory_length = []
        valid_indices = {}  # Dictionary mapping k -> list of valid indices

        # Initialize valid_indices for each k
        if isinstance(self.k, list):
            for k_val in self.k:
                valid_indices[k_val] = []
        else:
            valid_indices[self.k] = []

        data_dir_files = glob(os.path.join(data_path, "*"))
        for f in tqdm(data_dir_files, total=len(list(data_dir_files)), desc="Loading the data"):
            data = joblib.load(f)
            for trajectory in data.values():
                trajectory = np.argmax(trajectory, axis=3).astype(np.uint8)

                start_idx = len(boards)
                traj_len = len(trajectory)

                # For each state in trajectory, check if k-step future state exists
                for i in range(traj_len):
                    trajectory_start_idx.append(start_idx)
                    trajectory_length.append(traj_len)

                    # Check for each k value
                    if isinstance(self.k, list):
                        for k_val in self.k:
                            if i + k_val < traj_len:
                                valid_indices[k_val].append(len(boards))
                    else:
                        if i + self.k < traj_len:
                            valid_indices[self.k].append(len(boards))

                boards.extend(trajectory)

                if self.max_trajectories is not None and len(boards) > self.max_trajectories:
                    break

        # Store valid indices as class attribute
        self.valid_indices = valid_indices

        return (
            np.array(boards).astype(np.uint8),
            trajectory_start_idx,
            trajectory_length,
        )
