# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import base64
import io
from PIL import Image


class MazeDataset(Dataset):
    """Dataset class for loading maze data from parquet files."""

    def __init__(self, dataset_path: str, split: str = 'train'):
        """
        Initialize the maze dataset.

        Args:
            dataset_path: Path to the directory containing maze dataset files
            split: 'train' or 'test'
        """
        self.dataset_path = dataset_path
        self.split = split

        # Load the appropriate parquet file
        if split == 'train':
            file_path = os.path.join(dataset_path, 'maze_dataset_train.parquet')
        else:
            file_path = os.path.join(dataset_path, 'maze_dataset_test.parquet')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load the parquet file
        self.data = pd.read_parquet(file_path)

        # Extract prompts and metadata
        # Note: Adjust these column names based on actual maze dataset structure
        if 'instruction' in self.data.columns:
            self.prompts = self.data['instruction'].tolist()
        elif 'text' in self.data.columns:
            self.prompts = self.data['text'].tolist()
        else:
            # If no specific column found, create dummy prompts
            self.prompts = [f"Navigate through maze {i}" for i in range(len(self.data))]

        # Store metadata for each sample
        self.metadata = []
        for idx, row in self.data.iterrows():
            metadata = {
                'index': idx,
                'split': split
            }
            # Add any additional columns as metadata
            for col in self.data.columns:
                if col not in ['instruction', 'text']:
                    metadata[col] = row[col]
            self.metadata.append(metadata)

    @staticmethod
    def decode_base64_image(base64_str: str) -> Optional[Image.Image]:
        """
        Decode base64 string to PIL Image.

        Args:
            base64_str: Base64 encoded image string

        Returns:
            PIL Image object or None if decoding fails
        """
        if not base64_str or pd.isna(base64_str):
            return None

        try:
            # Remove data URL prefix if present (e.g., "data:image/png;base64,")
            if base64_str.startswith('data:'):
                base64_str = base64_str.split(',', 1)[1]

            # Decode base64 string
            image_data = base64.b64decode(base64_str)

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))

            return image
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        metadata = self.metadata[idx].copy()
        row = self.data.iloc[idx]

        # 获取基本信息
        prompt = self.prompts[idx]

        # 获取ID
        sample_id = row.get('id', f"maze_{idx}")

        # 解码所有图像
        original_img = self.decode_base64_image(row.get('original_img'))
        m_original_img = self.decode_base64_image(row.get('m_original_img'))
        sol_img = self.decode_base64_image(row.get('sol_img'))
        mask_img = self.decode_base64_image(row.get('mask_img'))
        cell_map = self.decode_base64_image(row.get('cell_map'))

        # 为图像编辑任务添加source_image_id
        metadata['source_image_id'] = str(sample_id)

        return {
            "id": sample_id,
            "prompt": prompt,
            "original_img": original_img,      # 无标记迷宫
            "m_original_img": m_original_img,  # 带标记迷宫
            "sol_img": sol_img,                # 解答图像
            "mask_img": mask_img,              # 解空间mask
            "cell_map": cell_map,              # 格子分割图
            "metadata": metadata
        }

    @staticmethod
    def collate_fn(examples):
        """Collate function for batching - compatible with GRPO training script."""
        # Extract prompts for compatibility with training script
        prompts = [example["prompt"] for example in examples]

        # Create enhanced metadata that includes all image data
        metadatas = []
        for example in examples:
            metadata = example["metadata"].copy()

            # Add images to metadata for reward function access
            metadata.update({
                'original_image': example["original_img"],    # For reward function compatibility
                'original_img': example["original_img"],      # Keep both naming conventions
                'm_original_img': example["m_original_img"],
                'sol_img': example["sol_img"],
                'mask_img': example["mask_img"],
                'cell_map': example["cell_map"],
                'id': example["id"]
            })

            metadatas.append(metadata)

        # Return format expected by training script: (prompts, metadatas)
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    """
    Distributed sampler that repeats each unique sample k times.
    This is used for GRPO where we need multiple samples per prompt for comparison.
    """

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        """
        Initialize the sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Batch size per replica
            k: Number of repetitions per sample
            num_replicas: Total number of replicas (GPUs)
            rank: Current replica rank
            seed: Random seed for synchronization
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k cannot divide n*b, k={k}, num_replicas={num_replicas}, "
            f"batch_size={batch_size}"
        )
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # Repeat each index k times
            repeated_indices = []
            for idx in indices:
                repeated_indices.extend([idx] * self.k)

            # Shuffle the repeated indices
            shuffled_indices = torch.tensor(repeated_indices)[
                torch.randperm(len(repeated_indices), generator=g)
            ].tolist()

            # Split among replicas
            samples_per_replica = len(shuffled_indices) // self.num_replicas
            start_idx = self.rank * samples_per_replica
            end_idx = start_idx + samples_per_replica

            replica_indices = shuffled_indices[start_idx:end_idx]

            # Yield in batches
            for i in range(0, len(replica_indices), self.batch_size):
                batch_indices = replica_indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size:
                    yield batch_indices

    def set_epoch(self, epoch):
        """Used to synchronize random state across epochs."""
        self.epoch = epoch


def create_maze_dataloader(
    config,
    split: str = 'train',
    num_replicas: int = 1,
    rank: int = 0
) -> DataLoader:
    """
    Create a DataLoader for the maze dataset.

    Args:
        config: GRPO configuration object
        split: 'train' or 'test'
        num_replicas: Number of distributed processes
        rank: Current process rank

    Returns:
        DataLoader for the maze dataset
    """
    # Create dataset
    dataset = MazeDataset(config.dataset_path, split=split)

    if split == 'train':
        # For training, use the distributed k-repeat sampler
        # This ensures we get k samples for each unique prompt for GRPO comparison
        k = getattr(config, 'num_image_per_prompt', 8)  # Default to 8 if not specified
        sampler = DistributedKRepeatSampler(
            dataset=dataset,
            batch_size=config.train_batch_size,
            k=k,
            num_replicas=num_replicas,
            rank=rank,
            seed=42
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config.train_batch_size,
            sampler=sampler,
            collate_fn=MazeDataset.collate_fn,
            num_workers=0,  # Set to 0 for simplicity
            pin_memory=True,
            drop_last=True
        )
    else:
        # For evaluation, use simple sequential sampling
        dataloader = DataLoader(
            dataset,
            batch_size=config.train_batch_size,
            shuffle=False,
            collate_fn=MazeDataset.collate_fn,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

    return dataloader


def get_maze_prompts(dataset_path: str, split: str = 'train', max_samples: int = None) -> List[str]:
    """
    Utility function to extract prompts from maze dataset.

    Args:
        dataset_path: Path to maze dataset
        split: 'train' or 'test'
        max_samples: Maximum number of samples to return (None for all)

    Returns:
        List of prompt strings
    """
    dataset = MazeDataset(dataset_path, split=split)
    prompts = dataset.prompts

    if max_samples is not None:
        prompts = prompts[:max_samples]

    return prompts