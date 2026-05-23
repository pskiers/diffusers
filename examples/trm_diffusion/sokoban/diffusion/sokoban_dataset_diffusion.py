from omegaconf import ListConfig
from torch.utils.data import Dataset, Sampler
import torch
import numpy as np
from typing import Union, Optional, List, Iterator
from tqdm import tqdm
import joblib
import os
from glob import glob


class GroupBatchSampler(Sampler[list[int]]):
    """Batch sampler that picks one random board per trajectory per batch.

    Each batch contains ``batch_size`` samples, each from a different trajectory.
    Trajectories are shuffled every epoch. When all trajectories are exhausted
    the iterator ends (one epoch = one pass through all groups).
    """

    def __init__(
        self,
        group_boundaries: list[int],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.group_boundaries = group_boundaries
        self.n_groups = len(group_boundaries) - 1
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = rng.permutation(self.n_groups)

        batch: list[int] = []
        for g in group_order:
            lo = self.group_boundaries[g]
            hi = self.group_boundaries[g + 1]
            if lo >= hi:
                continue  # skip empty group
            idx = int(rng.integers(lo, hi))
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_groups // self.batch_size
        return (self.n_groups + self.batch_size - 1) // self.batch_size


class SokobanBitDataset(Dataset):
    def __init__(self, sokoban_dataset, num_bits, clip_sample_range=1.0, weight_dtype=torch.float32):
        self.dataset = sokoban_dataset
        self.num_bits = num_bits
        self.clip_sample_range = clip_sample_range
        self.weight_dtype = weight_dtype

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        result = {}

        result["images"] = self._board_to_bits_tensor(item["target"])

        if "state" in item:
            result["conditions"] = self._board_to_bits_tensor(item["state"])

        if "distance_label" in item and item["distance_label"] is not None:
            result["class_labels"] = torch.tensor(item["distance_label"], dtype=torch.long)

        return result

    def _board_to_bits_tensor(self, board_np):
        board_tensor = torch.from_numpy(board_np).unsqueeze(0)
        images = self._int2bits(board_tensor, self.num_bits, self.weight_dtype)
        images = (images * 2 - 1.0) * self.clip_sample_range
        images = images.squeeze(0).permute(2, 0, 1)  # (num_bits, H, W)
        images = images.to(self.weight_dtype)
        return images

    def _int2bits(self, x, n, out_dtype=None):
        """Convert an integer x in (...) into bits in (..., n)."""
        x = torch.bitwise_right_shift(torch.unsqueeze(x, -1), torch.arange(n))
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
        max_boards_per_trajectory: int = 50,
        max_boards: Optional[int] = None,
    ):
        self.data_path = data_path
        if encoding != "bits":
            raise ValueError("Only 'bits' encoding supported")
        self.encoding = encoding

        if isinstance(k, (tuple, list, ListConfig)):
            self.k = list(k)
            if len(self.k) > 1:
                self.k_label = {k_val: i for i, k_val in enumerate(self.k)}
            else:
                self.k = self.k[0]
                self.k_label = None
        else:
            self.k = k
            self.k_label = None

        self.max_trajectories = max_trajectories
        self.max_boards_per_trajectory = max_boards_per_trajectory
        self.max_boards = max_boards

        self.boards, self.trajectory_start_idx, self.samples = self._load_boards(data_path)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectory_start_idx)

    def __getitem__(self, idx: int) -> dict:
        if self.k is None or self.k == 0:
            board_idx = self.samples[idx]
            return {"target": self.boards[board_idx]}

        board_idx, target_idx, k_val, k_lbl, current_step = self.samples[idx]

        return {
            "state": self.boards[board_idx],
            "target": self.boards[target_idx],
            "distance": k_val,
            "distance_label": k_lbl,
            "trajectory_timestep": current_step,
        }

    def _load_boards(self, data_path: str) -> tuple[np.ndarray, list[int], list]:
        boards_list = []
        trajectory_start_idx = []
        samples = []    # (board_idx, target_idx, k_value, k_label)
        group_boundaries = [0]  # cumulative sample count per trajectory

        current_global_idx = 0
        trajectories_loaded = 0

        rng = np.random.default_rng(42)
        data_dir_files = glob(os.path.join(data_path, "*"))

        for f in tqdm(data_dir_files, total=len(data_dir_files), desc="Loading the data"):
            data = joblib.load(f)

            for trajectory in data.values():
                trajectory_np = np.argmax(trajectory, axis=3).astype(np.uint8)
                traj_len = len(trajectory_np)

                # Subsample: take at most max_boards_per_trajectory boards
                sample_size = min(self.max_boards_per_trajectory, traj_len)
                selected_indices = rng.choice(traj_len, size=sample_size, replace=False)
                selected_indices.sort()
                trajectory_np = trajectory_np[selected_indices]

                boards_list.append(trajectory_np)
                trajectory_start_idx.append(current_global_idx)

                if self.k is None or self.k == 0:
                    for i in range(sample_size):
                        samples.append(current_global_idx + i)
                else:
                    k_values = self.k if isinstance(self.k, list) else [self.k]

                    for i in range(sample_size):
                        for k_val in k_values:
                            if i + k_val < sample_size:
                                b_idx = current_global_idx + i
                                t_idx = current_global_idx + i + k_val
                                k_lbl = self.k_label[k_val] if self.k_label else None
                                current_step = int(selected_indices[i])
                                samples.append((b_idx, t_idx, k_val, k_lbl, current_step))

                current_global_idx += sample_size
                group_boundaries.append(len(samples))
                trajectories_loaded += 1

                if self.max_trajectories is not None and trajectories_loaded >= self.max_trajectories:
                    break
                if self.max_boards is not None and current_global_idx >= self.max_boards:
                    break

            if self.max_trajectories is not None and trajectories_loaded >= self.max_trajectories:
                break
            if self.max_boards is not None and current_global_idx >= self.max_boards:
                break

        final_boards = np.concatenate(boards_list, axis=0)
        self.group_boundaries = group_boundaries

        return final_boards, trajectory_start_idx, samples
