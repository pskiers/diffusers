"""
datasets/toolhang_dataset.py — ToolHang datasets from robomimic (tool_hang/ph).

Two variants:
  ToolHangLowdimDataset  — state observations (object + robot)
  ToolHangImageDataset   — two camera views (sideview + robot0_eye_in_hand) plus
                           low-dim robot proprioception

DataSample field mapping:
  images               — (T, 7)  normalized action sequence (denoising target)
  spatial_conditions   — dict of per-view (T, 3, H, W) tensors  (image variant only)
  embedding_conditions — (T, obs_dim)  (lowdim variant, and robot proprioception
                          for the image variant)

Download helper: see download_tool_hang() at the bottom of this file.
"""

import os
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples
from datasets.diffusion_policy_utils import LinearNormalizer, get_image_range_normalizer, SequenceSampler
from datasets.pusht_dataset import _split_episodes, _build_split_data


# ---------------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------------

def _load_hdf5_data(hdf5_path, load_images=False, download=False):
    """Load robomimic-format HDF5 into memory.

    Parameters
    ----------
    hdf5_path : str or Path
        Path to the .hdf5 file.
    load_images : bool
        When True also load agentview and hand camera images.
    download : bool
        When True and hdf5_path is missing, download it automatically via
        download_tool_hang() (variant inferred from load_images).

    Returns
    -------
    data_dict : dict
        'action'        : (N_total, 7)        float32
        'obs_robot'     : (N_total, 9)        float32  (eef_pos 3 + eef_quat 4 + gripper 2)
        'obs_object'    : (N_total, 44)       float32  or None if key absent
        'sideview_image': (N_total, H, W, 3) uint8    only when load_images=True
        'hand_image'    : (N_total, H, W, 3)  uint8    only when load_images=True
    episode_ends : np.ndarray
        Shape (E,) exclusive cumulative lengths.
    """
    import h5py

    hdf5_path = str(hdf5_path)
    if not os.path.exists(hdf5_path) and download:
        variant = 'image' if load_images else 'lowdim'
        data_dir = hdf5_path.split('/robomimic/')[0] or 'data'
        download_tool_hang(data_dir, variant=variant)

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(
            f"ToolHang HDF5 not found at: {hdf5_path}\n"
            "Pass download=True, or download manually with:\n"
            "  python -c \"from datasets.toolhang_dataset import download_tool_hang; "
            f"download_tool_hang('data', '{'image' if load_images else 'lowdim'}')\"\n"
            "Or directly from:\n"
            "  https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip\n"
            "  https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip"
        )

    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = sorted(
            [k for k in f['data'].keys() if k.startswith('demo')],
            key=lambda x: int(x.split('_')[1])
        )

        actions_list = []
        obs_robot_list = []
        obs_object_list = []
        has_object = None

        if load_images:
            sideview_list = []
            hand_list = []

        for demo_key in demo_keys:
            demo = f['data'][demo_key]
            obs = demo['obs']

            # --- actions ---
            actions_list.append(demo['actions'][:].astype(np.float32))

            # --- robot state: eef_pos (3) + eef_quat (4) + gripper_qpos (2) ---
            eef_pos = obs['robot0_eef_pos'][:].astype(np.float32)    # (L, 3)
            eef_quat = obs['robot0_eef_quat'][:].astype(np.float32)  # (L, 4)
            gripper = obs['robot0_gripper_qpos'][:].astype(np.float32)  # (L, 2)
            robot_state = np.concatenate([eef_pos, eef_quat, gripper], axis=-1)  # (L, 9)
            obs_robot_list.append(robot_state)

            # --- object state (optional, 44D) ---
            if has_object is None:
                has_object = 'object' in obs
            if has_object:
                obs_object_list.append(obs['object'][:].astype(np.float32))  # (L, 44)

            # --- images ---
            if load_images:
                # sideview camera: try different key names (ToolHang uses sideview,
                # not agentview, per upstream tool_hang_image.yaml shape_meta)
                sideview_img = None
                for cam_key in ('sideview_image', 'agentview_image', 'robot0_sideview_image'):
                    if cam_key in obs:
                        sideview_img = obs[cam_key][:]
                        break
                if sideview_img is None:
                    raise KeyError(
                        f"Could not find sideview camera in demo '{demo_key}'. "
                        "Tried: sideview_image, agentview_image, robot0_sideview_image. "
                        f"Available keys: {list(obs.keys())}"
                    )
                sideview_list.append(sideview_img.astype(np.uint8))

                # hand camera
                hand_img = None
                for cam_key in ('robot0_eye_in_hand_image', 'hand_camera_image'):
                    if cam_key in obs:
                        hand_img = obs[cam_key][:]
                        break
                if hand_img is None:
                    raise KeyError(
                        f"Could not find hand camera in demo '{demo_key}'. "
                        "Tried: robot0_eye_in_hand_image, hand_camera_image. "
                        f"Available keys: {list(obs.keys())}"
                    )
                hand_list.append(hand_img.astype(np.uint8))

    # --- concatenate across demos ---
    data_dict = {}
    data_dict['action'] = np.concatenate(actions_list, axis=0)       # (N, 7)
    data_dict['obs_robot'] = np.concatenate(obs_robot_list, axis=0)  # (N, 9)

    if has_object and obs_object_list:
        data_dict['obs_object'] = np.concatenate(obs_object_list, axis=0)  # (N, 44)
    else:
        data_dict['obs_object'] = None

    if load_images:
        data_dict['sideview_image'] = np.concatenate(sideview_list, axis=0)  # (N, H, W, 3)
        data_dict['hand_image'] = np.concatenate(hand_list, axis=0)          # (N, H, W, 3)

    # --- episode_ends: exclusive cumulative lengths ---
    lengths = [len(a) for a in actions_list]
    episode_ends = np.cumsum(lengths).astype(np.int64)

    return data_dict, episode_ends


# ---------------------------------------------------------------------------
# ToolHangLowdimDataset
# ---------------------------------------------------------------------------

class ToolHangLowdimDataset(Dataset):
    """ToolHang low-dimensional dataset (robomimic tool_hang/ph).

    DataSample fields:
      images               — (T, 7) normalized action sequence (denoising target)
      embedding_conditions — (T, obs_dim) normalized state observations
        obs_dim = 9 (robot only) or 53 (object + robot, object-first, if
        object key present) — matches upstream's obs_keys order
        ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

    Requires h5py: pip install h5py
    Download data from https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip
    or the robomimic dataset page.
    """

    collate_fn = staticmethod(collate_data_samples)

    def __init__(
        self,
        hdf5_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.02,
        seed=42,
        split='train',
        download=False,
    ):
        super().__init__()

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Load HDF5 (lowdim keys only — no images)
        data_dict, episode_ends = _load_hdf5_data(hdf5_path, load_images=False, download=download)

        # Build observation array — object-first, matching upstream's
        # obs_keys order ['object', 'robot0_eef_pos', 'robot0_eef_quat',
        # 'robot0_gripper_qpos'].
        if data_dict['obs_object'] is not None:
            obs = np.concatenate([data_dict['obs_object'], data_dict['obs_robot']], axis=-1)  # (N, 53)
        else:
            obs = data_dict['obs_robot']  # (N, 9)

        full_data = {'obs': obs, 'action': data_dict['action']}

        # Fit normalizers over the full dataset (before splitting), matching
        # upstream: identity normalizer for actions (already-normalized delta
        # actions, abs_action=False), single global scalar max-abs normalizer
        # for obs (not a per-dimension [-1,1] fit).
        self.normalizer = LinearNormalizer()
        self.normalizer.fit({'action': data_dict['action']}, mode='identity')
        self.normalizer.fit({'obs': obs}, mode='global_scalar')

        # Real held-out train/val split (previously `split` was accepted but
        # ignored, so val always aliased train).
        selected = _split_episodes(episode_ends, split=split, val_ratio=val_ratio, seed=seed)
        self._data, split_episode_ends = _build_split_data(full_data, episode_ends, selected)

        # Build sampler
        self.sampler = SequenceSampler(split_episode_ends, horizon, pad_before, pad_after)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        seq = self.sampler.sample_sequence(idx, self._data)

        obs_seq = seq['obs']      # (T, obs_dim)  float32 numpy
        act_seq = seq['action']   # (T, 7)        float32 numpy

        obs_norm = self.normalizer['obs'].normalize(obs_seq)
        act_norm = self.normalizer['action'].normalize(act_seq)

        obs_tensor = torch.from_numpy(obs_norm)  # (T, obs_dim)
        act_tensor = torch.from_numpy(act_norm)  # (T, 7)

        return DataSample(
            images=act_tensor,
            embedding_conditions=obs_tensor,
        )


# ---------------------------------------------------------------------------
# ToolHangImageDataset
# ---------------------------------------------------------------------------

class ToolHangImageDataset(Dataset):
    """ToolHang image dataset — two camera views (sideview + robot0_eye_in_hand)
    plus robot proprioception, matching upstream's tool_hang_image.yaml shape_meta
    (which keeps low-dim robot state alongside the RGB views rather than
    dropping it).

    DataSample fields:
      images               — (T, 7) normalized action sequence (denoising target)
      spatial_conditions   — (T, 2, 3, 240, 240) float32 — sideview and hand
                              views stacked along a view dimension (kept
                              separate, not channel-concatenated, so a vision
                              encoder can encode each view independently)
      embedding_conditions — (T, 9) float32 normalized robot proprioception
                              (eef_pos 3 + eef_quat 4 + gripper 2)

    Requires h5py and potentially significant RAM for the full image dataset.
    The ~78GB image HDF5 will be loaded entirely into memory — ensure sufficient RAM.
    """

    collate_fn = staticmethod(collate_data_samples)

    def __init__(
        self,
        hdf5_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.02,
        seed=42,
        split='train',
        download=False,
    ):
        super().__init__()

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Load HDF5 including images — store uint8 to save memory
        data_dict, episode_ends = _load_hdf5_data(hdf5_path, load_images=True, download=download)

        full_data = {
            'action': data_dict['action'],               # (N, 7)  float32
            'obs_robot': data_dict['obs_robot'],          # (N, 9)  float32
            'sideview_image': data_dict['sideview_image'],  # (N, H, W, 3) uint8
            'hand_image': data_dict['hand_image'],        # (N, H, W, 3) uint8
        }

        # Fit normalizers over the full dataset (before splitting): identity
        # for actions, global scalar for robot proprioception (images are
        # normalized in __getitem__ via a fixed range normalizer).
        self.normalizer = LinearNormalizer()
        self.normalizer.fit({'action': data_dict['action']}, mode='identity')
        self.normalizer.fit({'obs_robot': data_dict['obs_robot']}, mode='global_scalar')
        self.image_normalizer = get_image_range_normalizer()  # scale=2, offset=-1

        # Real held-out train/val split.
        selected = _split_episodes(episode_ends, split=split, val_ratio=val_ratio, seed=seed)
        self._data, split_episode_ends = _build_split_data(full_data, episode_ends, selected)

        # Build sampler
        self.sampler = SequenceSampler(split_episode_ends, horizon, pad_before, pad_after)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        seq = self.sampler.sample_sequence(idx, self._data)

        # --- actions ---
        act_seq = seq['action']   # (T, 7) float32
        act_norm = self.normalizer['action'].normalize(act_seq)
        act_tensor = torch.from_numpy(act_norm)  # (T, 7)

        # --- robot proprioception ---
        obs_norm = self.normalizer['obs_robot'].normalize(seq['obs_robot'])
        obs_tensor = torch.from_numpy(obs_norm)  # (T, 9)

        # --- images: uint8 (T, H, W, C) -> float32 (T, C, H, W) in [-1, 1] ---
        sideview = seq['sideview_image'].astype(np.float32) / 255.0  # (T, H, W, 3)
        hand = seq['hand_image'].astype(np.float32) / 255.0          # (T, H, W, 3)

        sideview = np.transpose(sideview, (0, 3, 1, 2))  # (T, 3, H, W)
        hand = np.transpose(hand, (0, 3, 1, 2))          # (T, 3, H, W)

        sideview_t = torch.from_numpy(sideview) * 2.0 - 1.0  # (T, 3, H, W)
        hand_t = torch.from_numpy(hand) * 2.0 - 1.0          # (T, 3, H, W)

        # stack views along a new view axis -> (T, 2, 3, H, W); each view is
        # encoded independently downstream instead of being channel-merged.
        img_stacked = torch.stack([sideview_t, hand_t], dim=1)

        return DataSample(
            images=act_tensor,
            spatial_conditions=img_stacked,
            embedding_conditions=obs_tensor,
        )


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_tool_hang(data_dir='data', variant='lowdim'):
    """Download the ToolHang HDF5 dataset from diffusion_policy servers.

    Note this zip bundles robomimic's *entire* task suite (lift/can/square/
    transport/tool_hang, each ph/mh) under a top-level robomimic/ folder, not
    just ToolHang — that's why it's ~1.9GB (lowdim) / ~78GB (image) even
    though we only use the tool_hang/ph subset.

    Parameters
    ----------
    data_dir : str
        Root directory under which the data will be extracted.
    variant : str
        'lowdim' or 'image'.
    """
    import urllib.request, zipfile, pathlib
    out_dir = pathlib.Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f'https://diffusion-policy.cs.columbia.edu/data/training/robomimic_{variant}.zip'
    zip_path = out_dir / f'robomimic_{variant}.zip'
    print(f'Downloading ToolHang {variant} from {url} ...')
    urllib.request.urlretrieve(url, zip_path)
    # The zip already contains a top-level robomimic/ folder (verified against
    # its actual contents), so it's extracted into data_dir directly —
    # extracting into data_dir/robomimic/ would double-nest it.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    print('Done.')
