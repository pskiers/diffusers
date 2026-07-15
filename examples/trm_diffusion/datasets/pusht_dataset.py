"""
datasets/pusht_dataset.py — PyTorch Dataset classes for the PushT robot
manipulation task following Chi et al. 2023 "Diffusion Policy".

Three variants are provided:
  PushTImageDataset   — RGB image observations + 2-D continuous actions
  PushTLowdimDataset  — Low-dimensional state observations + 2-D actions
  PushTHybridDataset  — Image + agent-pos observations + 2-D actions

DataSample field mapping (no new fields introduced):
  images               → normalized action sequence  (T, 2)
  spatial_conditions   → normalized image obs        (T, 3, 96, 96)
  embedding_conditions → normalized state / obs      (T, D)

Paper hyperparameters (Chi et al. 2023):
  horizon          = 16
  n_obs_steps      = 2
  n_action_steps   = 8
  pad_before       = 1
  pad_after        = 7
  val_ratio        = 0.02
  max_train_eps    = 90
  zarr source      = pusht_cchi_v7_replay.zarr

Download:
  wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples
from datasets.diffusion_policy_utils import (
    LinearNormalizer,
    get_image_range_normalizer,
    SequenceSampler,
    load_zarr_data,
)


# ---------------------------------------------------------------------------
# Auto-download helper
# ---------------------------------------------------------------------------

def _ensure_pusht_data(zarr_path, download):
    """Download the PushT zarr store if missing and download=True."""
    if os.path.exists(zarr_path):
        return
    if not download:
        raise FileNotFoundError(
            f"PushT zarr not found at: {zarr_path}\n"
            "Pass download=True, or download manually with:\n"
            "  python -c \"from datasets.pusht_dataset import download_pusht; download_pusht('data')\""
        )
    data_dir = zarr_path.split('/pusht/')[0] or 'data'
    download_pusht(data_dir)
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"download_pusht() completed but {zarr_path} still doesn't exist — "
            "check that zarr_path matches the downloaded layout "
            "(<data_dir>/pusht/pusht_cchi_v7_replay.zarr)."
        )


# ---------------------------------------------------------------------------
# Shared split logic
# ---------------------------------------------------------------------------

def _split_episodes(episode_ends, split, val_ratio=0.02, seed=42, max_train_episodes=None):
    """Partition episode indices into train / val sets.

    Parameters
    ----------
    episode_ends : np.ndarray
        Shape (E,) cumulative exclusive end indices from the zarr store.
    split : str
        'train' or 'val'.
    val_ratio : float
        Fraction of episodes reserved for validation.
    seed : int
        RNG seed for reproducibility.
    max_train_episodes : int or None
        If given, cap the training set to this many episodes (first N after
        excluding val episodes).

    Returns
    -------
    selected_indices : list[int]
        Episode indices (into episode_ends) belonging to the requested split.
    """
    total_episodes = len(episode_ends)
    n_val = min(max(1, round(total_episodes * val_ratio)), total_episodes - 1)

    rng = np.random.default_rng(seed)
    val_mask = np.zeros(total_episodes, dtype=bool)
    val_idx = rng.choice(total_episodes, size=n_val, replace=False)
    val_mask[val_idx] = True

    if split == 'val':
        selected = np.where(val_mask)[0].tolist()
    else:
        # train: all non-val episodes, optionally capped
        train_idx = np.where(~val_mask)[0].tolist()
        if max_train_episodes is not None:
            train_idx = train_idx[:max_train_episodes]
        selected = train_idx

    return selected


def _build_split_data(data_dict, episode_ends, selected_episode_indices):
    """Concatenate the data slices belonging to the selected episodes.

    Parameters
    ----------
    data_dict : dict
        Full buffer arrays keyed by field name, each shape (N, *feat).
    episode_ends : np.ndarray
        Shape (E,) cumulative exclusive end indices.
    selected_episode_indices : list[int]
        Which episodes to include.

    Returns
    -------
    split_data : dict
        Concatenated slices for the requested episodes.
    split_episode_ends : np.ndarray
        Recomputed cumulative end indices for the split buffer.
    """
    slices = []
    ep_starts = np.concatenate([[0], episode_ends[:-1]])

    for ep_i in selected_episode_indices:
        slices.append((int(ep_starts[ep_i]), int(episode_ends[ep_i])))

    split_data = {}
    for key, arr in data_dict.items():
        parts = [arr[s:e] for s, e in slices]
        split_data[key] = np.concatenate(parts, axis=0)

    # Recompute episode_ends as cumulative lengths
    ep_lengths = [e - s for s, e in slices]
    split_episode_ends = np.cumsum(ep_lengths, dtype=np.int64)

    return split_data, split_episode_ends


# ---------------------------------------------------------------------------
# PushTImageDataset
# ---------------------------------------------------------------------------

class PushTImageDataset(Dataset):
    """PushT visual dataset (96x96 RGB images + 2-D continuous actions).

    DataSample fields:
      images              — (T, 2) float32 normalized action sequence (denoising target)
      spatial_conditions  — (T, 3, 96, 96) float32 normalized image observations

    Normalizers are fitted on the FULL dataset before splitting to ensure
    consistent statistics regardless of the train/val split.

    Download:
        wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
    """

    def __init__(
        self,
        zarr_path,
        split='train',
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.02,
        max_train_episodes=None,
        seed=42,
        download=False,
    ):
        super().__init__()

        _ensure_pusht_data(zarr_path, download)

        # ── Load full zarr buffer ─────────────────────────────────────────────
        data, episode_ends = load_zarr_data(zarr_path)
        # data['img']:    (N, 96, 96, 3) uint8
        # data['action']: (N, 2)         float32

        # ── Fit normalizers on the FULL dataset for consistent statistics ─────
        self.normalizer = LinearNormalizer()
        self.normalizer['action'] = LinearNormalizer()
        # Reuse a fresh LinearNormalizer to fit action only
        _action_norm = LinearNormalizer()
        _action_norm.fit({'action': data['action']})
        self.normalizer['action'] = _action_norm['action']

        # Image normalizer: maps float [0, 1] -> [-1, 1]
        self.normalizer['image'] = get_image_range_normalizer()

        # ── Partition episodes into train / val ───────────────────────────────
        selected = _split_episodes(
            episode_ends,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            max_train_episodes=max_train_episodes,
        )

        # Build split-specific data dict and recomputed episode_ends
        self.split_data, split_episode_ends = _build_split_data(
            data, episode_ends, selected
        )

        # ── Sliding-window sampler ────────────────────────────────────────────
        self.sampler = SequenceSampler(
            episode_ends=split_episode_ends,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

    # ------------------------------------------------------------------

    def get_normalizer(self):
        """Return the fitted LinearNormalizer (action + image)."""
        return self.normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        # Sample a raw window of length T from the split buffer
        seq = self.sampler.sample_sequence(idx, self.split_data)
        # seq['img']:    (T, 96, 96, 3) uint8
        # seq['action']: (T, 2)         float32

        # ── Normalize image ───────────────────────────────────────────────────
        # Convert uint8 [0, 255] -> float32 [0, 1] first, then apply [-1, 1] norm
        img_float = seq['img'].astype(np.float32) / 255.0   # (T, H, W, C)
        img_norm = self.normalizer['image'].normalize(img_float)  # (T, H, W, C)

        # Transpose to (T, C, H, W) and make a contiguous float32 tensor
        img_tensor = torch.from_numpy(
            np.transpose(img_norm, (0, 3, 1, 2)).copy()
        ).float()  # (T, 3, 96, 96)

        # ── Normalize action ──────────────────────────────────────────────────
        act_norm = self.normalizer['action'].normalize(seq['action'])  # (T, 2)
        act_tensor = torch.from_numpy(act_norm.copy()).float()         # (T, 2)

        return DataSample(
            images=act_tensor,              # denoising target: (T, 2)
            spatial_conditions=img_tensor,  # visual conditioning: (T, 3, 96, 96)
        )

    collate_fn = staticmethod(collate_data_samples)


# ---------------------------------------------------------------------------
# PushTLowdimDataset
# ---------------------------------------------------------------------------

class PushTLowdimDataset(Dataset):
    """PushT low-dimensional state dataset.

    DataSample fields:
      images               — (T, 2)      float32 normalized action sequence
      embedding_conditions — (T, obs_dim) float32 normalized state observations

    obs_dim:
      20  if 'keypoint' key is present (9 keypoints x 2 + agent_pos 2)
       2  otherwise (agent_pos only from state[:, :2])

    Download:
        wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
    """

    def __init__(
        self,
        zarr_path,
        split='train',
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.02,
        max_train_episodes=None,
        seed=42,
        download=False,
    ):
        super().__init__()

        _ensure_pusht_data(zarr_path, download)

        # ── Load full zarr buffer ─────────────────────────────────────────────
        data, episode_ends = load_zarr_data(zarr_path)
        # data['state']:    (N, 5)      float32  [agent_pos(2), block_pos+angle(3)]
        # data['action']:   (N, 2)      float32
        # data['keypoint']: (N, 9, 2)   float32  (optional)

        # ── Build observation array from available keys ───────────────────────
        agent_pos = data['state'][:, :2]  # (N, 2) always available

        if 'keypoint' in data:
            # Flatten keypoints (N, 9, 2) -> (N, 18), then concat agent_pos -> (N, 20)
            keypoints_flat = data['keypoint'].reshape(len(data['keypoint']), -1)
            obs = np.concatenate([keypoints_flat, agent_pos], axis=-1)  # (N, 20)
        else:
            obs = agent_pos  # (N, 2)

        # ── Fit normalizers on the FULL dataset ───────────────────────────────
        self.normalizer = LinearNormalizer()
        _norm = LinearNormalizer()
        _norm.fit({'obs': obs, 'action': data['action']})
        self.normalizer['obs'] = _norm['obs']
        self.normalizer['action'] = _norm['action']

        # ── Store obs in a combined data dict for the sampler ─────────────────
        full_data = {'obs': obs, 'action': data['action']}

        # ── Partition episodes into train / val ───────────────────────────────
        selected = _split_episodes(
            episode_ends,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            max_train_episodes=max_train_episodes,
        )

        self.split_data, split_episode_ends = _build_split_data(
            full_data, episode_ends, selected
        )

        # ── Sliding-window sampler ────────────────────────────────────────────
        self.sampler = SequenceSampler(
            episode_ends=split_episode_ends,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

    # ------------------------------------------------------------------

    def get_normalizer(self):
        """Return the fitted LinearNormalizer (obs + action)."""
        return self.normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        seq = self.sampler.sample_sequence(idx, self.split_data)
        # seq['obs']:    (T, obs_dim)
        # seq['action']: (T, 2)

        # ── Normalize ─────────────────────────────────────────────────────────
        obs_norm = self.normalizer['obs'].normalize(seq['obs'])
        act_norm = self.normalizer['action'].normalize(seq['action'])

        obs_tensor = torch.from_numpy(obs_norm.copy()).float()  # (T, obs_dim)
        act_tensor = torch.from_numpy(act_norm.copy()).float()  # (T, 2)

        return DataSample(
            images=act_tensor,                   # denoising target: (T, 2)
            embedding_conditions=obs_tensor,     # state conditioning: (T, obs_dim)
        )

    collate_fn = staticmethod(collate_data_samples)


# ---------------------------------------------------------------------------
# PushTHybridDataset
# ---------------------------------------------------------------------------

class PushTHybridDataset(Dataset):
    """PushT hybrid dataset (image + agent-pos observations + 2-D actions).

    DataSample fields:
      images               — (T, 2)        float32 normalized action sequence
      spatial_conditions   — (T, 3, 96, 96) float32 normalized image observations
      embedding_conditions — (T, 2)        float32 normalized agent-pos observations

    Agent position is always taken from state[:, :2] regardless of whether
    a 'keypoint' key is present.

    Download:
        wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
    """

    def __init__(
        self,
        zarr_path,
        split='train',
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.02,
        max_train_episodes=None,
        seed=42,
        download=False,
    ):
        super().__init__()

        _ensure_pusht_data(zarr_path, download)

        # ── Load full zarr buffer ─────────────────────────────────────────────
        data, episode_ends = load_zarr_data(zarr_path)
        # data['img']:    (N, 96, 96, 3) uint8
        # data['state']:  (N, 5)         float32
        # data['action']: (N, 2)         float32

        # Agent position: always 2-D regardless of keypoint availability
        agent_pos = data['state'][:, :2]  # (N, 2)

        # ── Fit normalizers on the FULL dataset ───────────────────────────────
        self.normalizer = LinearNormalizer()

        _norm = LinearNormalizer()
        _norm.fit({'agent_pos': agent_pos, 'action': data['action']})
        self.normalizer['agent_pos'] = _norm['agent_pos']
        self.normalizer['action'] = _norm['action']

        # Image normalizer: float [0, 1] -> [-1, 1]
        self.normalizer['image'] = get_image_range_normalizer()

        # ── Combine into a single data dict for the sampler ───────────────────
        full_data = {
            'img': data['img'],
            'agent_pos': agent_pos,
            'action': data['action'],
        }

        # ── Partition episodes into train / val ───────────────────────────────
        selected = _split_episodes(
            episode_ends,
            split=split,
            val_ratio=val_ratio,
            seed=seed,
            max_train_episodes=max_train_episodes,
        )

        self.split_data, split_episode_ends = _build_split_data(
            full_data, episode_ends, selected
        )

        # ── Sliding-window sampler ────────────────────────────────────────────
        self.sampler = SequenceSampler(
            episode_ends=split_episode_ends,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

    # ------------------------------------------------------------------

    def get_normalizer(self):
        """Return the fitted LinearNormalizer (action + agent_pos + image)."""
        return self.normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        seq = self.sampler.sample_sequence(idx, self.split_data)
        # seq['img']:       (T, 96, 96, 3) uint8
        # seq['agent_pos']: (T, 2)         float32
        # seq['action']:    (T, 2)         float32

        # ── Normalize image ───────────────────────────────────────────────────
        img_float = seq['img'].astype(np.float32) / 255.0   # (T, H, W, C)
        img_norm = self.normalizer['image'].normalize(img_float)  # (T, H, W, C)
        img_tensor = torch.from_numpy(
            np.transpose(img_norm, (0, 3, 1, 2)).copy()
        ).float()  # (T, 3, 96, 96)

        # ── Normalize agent_pos and action ────────────────────────────────────
        pos_norm = self.normalizer['agent_pos'].normalize(seq['agent_pos'])
        act_norm = self.normalizer['action'].normalize(seq['action'])

        pos_tensor = torch.from_numpy(pos_norm.copy()).float()  # (T, 2)
        act_tensor = torch.from_numpy(act_norm.copy()).float()  # (T, 2)

        return DataSample(
            images=act_tensor,               # denoising target: (T, 2)
            spatial_conditions=img_tensor,   # visual conditioning: (T, 3, 96, 96)
            embedding_conditions=pos_tensor, # state conditioning: (T, 2)
        )

    collate_fn = staticmethod(collate_data_samples)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_pusht(data_dir='data'):
    """Download PushT zarr dataset."""
    import urllib.request, zipfile, pathlib
    out_dir = pathlib.Path(data_dir) / 'pusht'
    out_dir.mkdir(parents=True, exist_ok=True)
    url = 'https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip'
    zip_path = out_dir / 'pusht.zip'
    print(f'Downloading PushT from {url} ...')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    print('Done.')
