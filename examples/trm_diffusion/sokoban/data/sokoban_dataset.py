from torch.utils.data import Dataset, Sampler
class
import numpy as np
import os
from glob import glob
import joblib
from tqdm import tqdm


class SokobanDatasetTokens(Dataset):
    def __init__(self, data_path: str, total_dataset_size: int, num_files=None):
        self.data_path = data_path
        self.num_files = num_files
        self.total_dataset_size = total_dataset_size

    @classmethod
    def for_unconditional_generation(cls, data_path: str, total_dataset_size: int, lazy=True):
        num_files = 0
        total_trajectories = 0
        data_dir_files = glob(os.path.join(data_path, "*"))

        for f in tqdm(data_dir_files, total=len(data_dir_files), desc="Loading the data"):
            num_files += 1

            data: dict = joblib.load(f)

            total_trajectories += len(data.keys())
            if total_trajectories >= total_dataset_size:
                return cls(data_path=data_path, total_dataset_size=total_dataset_size, num_files=num_files)

        raise ValueError(f"Not enough data to construct {total_dataset_size} boards")

    @classmethod
    def for_conditional_generation(cls, data_path, total_dataset_size: int):
        ...

    def __len__(self):
        return self.total_dataset_size

    def __getitem__(self, index):
        ...

    @staticmethod
    def render_board(board_tokens):
        """Gets 12x12 matrix with bits and returns valid image to output"""

    def _load_boards_from_files(self, data_path: str, total_dataset_size: int, one_per_tragectory=False):
        """
        Total dataset length: 1000 for TRM, ?? for standard diffusion
        total dataset length / 4 (box numbers)
        one per trajectory - for uncond generation, true fo cond

        150 files with 1000 trajectories
        each trajectory has around 30 boards (states)
        Unconditional generation (lazy evaluation per batch):
        -> 1, random board from each trajectory
        -> number of trajectories configurable (8 augmentation per one trajectory)
            -> trm diffusion needs 1000 trajectories * 8 augmentations
            -> diffusion will need more
        Conditional generation
        -> all boards from each trajectory (around 30 boards)
        ->
        """
        per_box_length = total_dataset_size // 4    # 1, 2, 3 or 4 boxes

        rng = np.random.default_rng(42)
        data_dir_files = glob(os.path.join(data_path, "*"))

        if one_per_tragectory:
            self.num_files = 0

        for f in tqdm(data_dir_files, total=len(data_dir_files), desc="Loading the data"):
            data: dict = joblib.load(f)

            trajectories_in_file = len(data.keys())


class SokobanBitsDataset(SokobanDatasetTokens):
    """For Bit-Diffusion"""
    def __init__(self, data_path: str, total_dataset_size: int, num_files=None):
        super().__init__(data_path, total_dataset_size, num_files)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        tokens = super().__getitem__(index)
        # change this tokens to bits

    @staticmethod
    def render_bit_boards(board_bits):
        """Gets 12x12 matrix with bits and returns valid image to output"""
        board_tokens = ... # transformation logic here
        return SokobanDatasetTokens.render_board(board_tokens)

