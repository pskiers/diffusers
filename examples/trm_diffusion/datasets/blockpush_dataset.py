"""
Block Push multi-modal dataset for Diffusion Policy (Chi et al. 2023).

DataSample field mapping:
  images               — (T, 2)  normalized action sequence  (denoising target)
  embedding_conditions — (T, 16) normalized state observations (conditioning)

Download:
  wget https://diffusion-policy.cs.columbia.edu/data/training/multimodal_push_seed.zip
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples
from datasets.diffusion_policy_utils import LinearNormalizer, SequenceSampler, load_zarr_data


def _ensure_blockpush_data(zarr_path, download):
    """Download the BlockPush zarr store if missing and download=True."""
    if os.path.exists(zarr_path):
        return
    if not download:
        raise FileNotFoundError(
            f"BlockPush zarr not found at: {zarr_path}\n"
            "Pass download=True, or download manually with:\n"
            "  python -c \"from datasets.blockpush_dataset import download_block_push; download_block_push('data')\""
        )
    data_dir = zarr_path.split('/block_pushing/')[0] or 'data'
    download_block_push(data_dir)
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"download_block_push() completed but {zarr_path} still doesn't exist — "
            "check that zarr_path matches the downloaded layout "
            "(<data_dir>/block_pushing/multimodal_push_seed.zarr)."
        )


class BlockPushDataset(Dataset):
    """Block Push multi-modal dataset (16D state, 2D action).

    DataSample fields:
      images               — (T, 2) normalized action sequence (denoising target)
      embedding_conditions — (T, 16) normalized state observations

    Download: wget https://diffusion-policy.cs.columbia.edu/data/training/block_pushing.zip
    or from https://diffusion-policy.cs.columbia.edu/data/training/multimodal_push_seed.zip
    """

    collate_fn = staticmethod(collate_data_samples)

    def __init__(
        self,
        zarr_path,
        split='train',
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.02,
        seed=42,
        download=False,
    ):
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        _ensure_blockpush_data(zarr_path, download)

        # ------------------------------------------------------------------ #
        # 1. Load full zarr data                                              #
        # ------------------------------------------------------------------ #
        full_data_dict, episode_ends = load_zarr_data(zarr_path)
        # full_data_dict keys: 'obs' (N, 16), 'action' (N, 2)
        # episode_ends: (E,) cumulative exclusive end indices

        total_episodes = len(episode_ends)

        # ------------------------------------------------------------------ #
        # 2. Fit normalizers on the FULL data (before splitting)              #
        # ------------------------------------------------------------------ #
        self.normalizer = LinearNormalizer()
        self.normalizer.fit({
            'obs': full_data_dict['obs'],
            'action': full_data_dict['action'],
        })

        # ------------------------------------------------------------------ #
        # 3. Train / val episode split                                        #
        # ------------------------------------------------------------------ #
        n_val = min(max(1, round(total_episodes * val_ratio)), total_episodes - 1)
        rng = np.random.default_rng(seed)
        val_indices = set(rng.choice(total_episodes, size=n_val, replace=False).tolist())

        if split == 'train':
            selected = [i for i in range(total_episodes) if i not in val_indices]
        else:
            selected = [i for i in range(total_episodes) if i in val_indices]

        # Build split-specific data dict by concatenating selected episode slices
        ep_starts = np.concatenate([[0], episode_ends[:-1]])

        obs_parts = []
        action_parts = []
        new_episode_ends = []
        cursor = 0

        for ep_idx in selected:
            start = int(ep_starts[ep_idx])
            end = int(episode_ends[ep_idx])
            obs_parts.append(full_data_dict['obs'][start:end])
            action_parts.append(full_data_dict['action'][start:end])
            cursor += end - start
            new_episode_ends.append(cursor)

        if len(obs_parts) == 0:
            # Edge case: empty split
            split_obs = np.zeros((0, 16), dtype=np.float32)
            split_action = np.zeros((0, 2), dtype=np.float32)
            split_episode_ends = np.array([], dtype=np.int64)
        else:
            split_obs = np.concatenate(obs_parts, axis=0)
            split_action = np.concatenate(action_parts, axis=0)
            split_episode_ends = np.array(new_episode_ends, dtype=np.int64)

        self.split_data_dict = {
            'obs': split_obs,
            'action': split_action,
        }

        # ------------------------------------------------------------------ #
        # 4. Build sequence sampler                                           #
        # ------------------------------------------------------------------ #
        self.sampler = SequenceSampler(
            split_episode_ends,
            horizon,
            pad_before,
            pad_after,
        )

    def get_normalizer(self):
        """Return the fitted LinearNormalizer."""
        return self.normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        seq = self.sampler.sample_sequence(idx, self.split_data_dict)
        # seq['obs']:    (T, 16) float32
        # seq['action']: (T, 2)  float32

        obs_norm = self.normalizer['obs'].normalize(seq['obs'])
        act_norm = self.normalizer['action'].normalize(seq['action'])

        return DataSample(
            images=torch.from_numpy(act_norm).float(),
            embedding_conditions=torch.from_numpy(obs_norm).float(),
        )


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_block_push(data_dir='data'):
    """Download BlockPush zarr dataset from diffusion_policy servers."""
    import urllib.request, zipfile, pathlib
    out_dir = pathlib.Path(data_dir) / 'block_pushing'
    out_dir.mkdir(parents=True, exist_ok=True)
    url = 'https://diffusion-policy.cs.columbia.edu/data/training/multimodal_push_seed.zip'
    zip_path = out_dir / 'block_push.zip'
    print(f'Downloading BlockPush from {url} ...')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    print('Done.')
