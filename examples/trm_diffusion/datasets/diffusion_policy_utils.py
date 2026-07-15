"""
Shared utilities for diffusion policy datasets.
Follows real-stanford/diffusion_policy patterns.
"""

import os
from pathlib import Path

import numpy as np
import torch
import zarr


# ---------------------------------------------------------------------------
# SingleFieldLinearNormalizer
# ---------------------------------------------------------------------------

class SingleFieldLinearNormalizer:
    """Per-dimension min-max linear normalizer mapping data to [-1, 1]."""

    def __init__(self, scale, offset):
        # scale and offset are float32 numpy arrays of shape (D,)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.offset = np.asarray(offset, dtype=np.float32)

    @classmethod
    def fit(cls, data):
        """Fit normalizer from array of shape (..., D).

        Parameters
        ----------
        data : np.ndarray
            Array whose last dimension is the feature dimension.

        Returns
        -------
        SingleFieldLinearNormalizer
        """
        data = np.asarray(data, dtype=np.float32)
        # Flatten all leading dims so shape becomes (N, D)
        D = data.shape[-1]
        flat = data.reshape(-1, D)

        lo = flat.min(axis=0)   # (D,)
        hi = flat.max(axis=0)   # (D,)
        data_range = hi - lo    # (D,)

        scale = np.empty(D, dtype=np.float32)
        offset = np.empty(D, dtype=np.float32)

        for i in range(D):
            if data_range[i] < 1e-7:
                # Degenerate dimension: center but don't scale
                scale[i] = 1.0
                offset[i] = -(lo[i] + hi[i]) / 2.0
            else:
                scale[i] = 2.0 / data_range[i]
                offset[i] = -1.0 - scale[i] * lo[i]

        return cls(scale, offset)

    @classmethod
    def fit_identity(cls, data):
        """Identity normalizer (scale=1, offset=0) sized to data's feature dim.

        Matches upstream's ``get_identity_normalizer_from_stat`` — used e.g.
        for actions that are already normalized (delta/relative actions).
        """
        data = np.asarray(data, dtype=np.float32)
        D = data.shape[-1]
        return cls(np.ones(D, dtype=np.float32), np.zeros(D, dtype=np.float32))

    @classmethod
    def fit_global_scalar(cls, data, output_max=1.0, output_min=-1.0):
        """Single symmetric max-abs scale shared across all feature dims.

        Matches upstream's ``normalizer_from_stat`` used for robomimic
        low-dim observations: one scalar scale/offset broadcast over every
        dimension, rather than a per-dimension [-1, 1] fit.
        """
        data = np.asarray(data, dtype=np.float32)
        D = data.shape[-1]
        flat = data.reshape(-1, D)
        input_max = flat.max()
        input_min = flat.min()
        input_abs_max = max(abs(input_max), abs(input_min), 1e-7)
        output_abs_max = max(abs(output_max), abs(output_min))
        scale_val = output_abs_max / input_abs_max
        scale = np.full(D, scale_val, dtype=np.float32)
        offset = np.zeros(D, dtype=np.float32)
        return cls(scale, offset)

    def normalize(self, x):
        """Map x -> x * scale + offset.

        Works on both np.ndarray and torch.Tensor.
        """
        if isinstance(x, torch.Tensor):
            scale = torch.from_numpy(self.scale).to(x.device, x.dtype)
            offset = torch.from_numpy(self.offset).to(x.device, x.dtype)
        else:
            scale = self.scale
            offset = self.offset
        return x * scale + offset

    def unnormalize(self, x):
        """Map x -> (x - offset) / scale.

        Works on both np.ndarray and torch.Tensor.
        """
        if isinstance(x, torch.Tensor):
            scale = torch.from_numpy(self.scale).to(x.device, x.dtype)
            offset = torch.from_numpy(self.offset).to(x.device, x.dtype)
        else:
            scale = self.scale
            offset = self.offset
        return (x - offset) / scale

    def to_dict(self):
        """Serialize to plain dict of numpy arrays."""
        return {
            'scale': self.scale.copy(),
            'offset': self.offset.copy(),
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dict produced by to_dict()."""
        return cls(
            scale=np.asarray(d['scale'], dtype=np.float32),
            offset=np.asarray(d['offset'], dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# get_image_range_normalizer
# ---------------------------------------------------------------------------

def get_image_range_normalizer():
    """Return a normalizer mapping [0, 1] -> [-1, 1].

    scale=2.0, offset=-1.0 stored as shape-(1,) arrays for broadcasting.
    """
    scale = np.array([2.0], dtype=np.float32)
    offset = np.array([-1.0], dtype=np.float32)
    return SingleFieldLinearNormalizer(scale, offset)


# ---------------------------------------------------------------------------
# LinearNormalizer
# ---------------------------------------------------------------------------

class LinearNormalizer:
    """Container mapping string keys to SingleFieldLinearNormalizer instances."""

    def __init__(self):
        self._normalizers = {}

    def fit(self, data, last_n_dims=1, mode='limits'):
        """Fit per-key normalizers from a dict of numpy arrays.

        Parameters
        ----------
        data : dict
            Mapping from string key to np.ndarray.
        last_n_dims : int
            Number of trailing dimensions to treat as the feature dimension.
            They are flattened together before fitting.
        mode : str
            One of 'limits' (per-dimension min-max to [-1,1]), 'identity'
            (scale=1, offset=0 — data assumed already normalized), or
            'global_scalar' (single symmetric max-abs scale shared across
            all feature dimensions, matching upstream's robomimic low-dim
            normalizer).
        """
        fit_fns = {
            'limits': SingleFieldLinearNormalizer.fit,
            'identity': SingleFieldLinearNormalizer.fit_identity,
            'global_scalar': SingleFieldLinearNormalizer.fit_global_scalar,
        }
        if mode not in fit_fns:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {list(fit_fns)}.")
        fit_fn = fit_fns[mode]

        for key, arr in data.items():
            arr = np.asarray(arr, dtype=np.float32)
            if last_n_dims > 1:
                # Flatten the last last_n_dims dimensions into one
                leading_shape = arr.shape[:-last_n_dims]
                feature_size = int(np.prod(arr.shape[-last_n_dims:]))
                arr = arr.reshape(leading_shape + (feature_size,))
            self._normalizers[key] = fit_fn(arr)

    def normalize(self, data):
        """Apply per-key normalizer to each array in data dict."""
        out = {}
        for key, arr in data.items():
            if key in self._normalizers:
                out[key] = self._normalizers[key].normalize(arr)
            else:
                out[key] = arr
        return out

    def unnormalize(self, data):
        """Apply per-key inverse normalizer to each array in data dict."""
        out = {}
        for key, arr in data.items():
            if key in self._normalizers:
                out[key] = self._normalizers[key].unnormalize(arr)
            else:
                out[key] = arr
        return out

    def __getitem__(self, key):
        return self._normalizers[key]

    def __setitem__(self, key, normalizer):
        self._normalizers[key] = normalizer

    def get(self, key, default=None):
        return self._normalizers.get(key, default)

    def save(self, path):
        """Save all normalizers to a .npz file.

        Stores 'keys' as an object array of strings, and for each key k
        stores 'k_scale' and 'k_offset' arrays.
        """
        arrays = {}
        keys = list(self._normalizers.keys())
        arrays['keys'] = np.array(keys, dtype=object)
        for k in keys:
            d = self._normalizers[k].to_dict()
            arrays[k + '_scale'] = d['scale']
            arrays[k + '_offset'] = d['offset']
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Load a LinearNormalizer from a .npz file."""
        data = np.load(path, allow_pickle=True)
        norm = cls()
        keys = data['keys'].tolist()
        for k in keys:
            scale = data[k + '_scale']
            offset = data[k + '_offset']
            norm._normalizers[k] = SingleFieldLinearNormalizer(scale, offset)
        return norm


# ---------------------------------------------------------------------------
# SequenceSampler
# ---------------------------------------------------------------------------

class SequenceSampler:
    """Episode-boundary-aware sliding window sampler.

    Parameters
    ----------
    episode_ends : np.ndarray
        1-D array of cumulative EXCLUSIVE end indices.
        Episode i spans buffer indices [episode_ends[i-1], episode_ends[i]).
        Episode 0 spans [0, episode_ends[0]).
    sequence_length : int
        Length of each sampled window.
    pad_before : int
        Maximum repeat-first-frame padding steps at the start of an episode.
    pad_after : int
        Maximum repeat-last-frame padding steps at the end of an episode.
    """

    def __init__(self, episode_ends, sequence_length, pad_before=0, pad_after=0):
        episode_ends = np.asarray(episode_ends, dtype=np.int64)
        self.episode_ends = episode_ends
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Build index list: each entry is (ep_start_in_buffer, window_start, ep_length)
        # window_start is relative to the episode start; can be negative (pad before)
        # or cause the window to extend past the episode (pad after).
        indices = []
        ep_start = 0
        for ep_idx, ep_end in enumerate(episode_ends):
            L = int(ep_end) - int(ep_start)
            # window_start ranges from -pad_before to L - sequence_length + pad_after
            # (matches upstream diffusion_policy's SequenceSampler max_start bound)
            max_start = L - sequence_length + pad_after
            for window_start in range(-pad_before, max_start + 1):
                indices.append((int(ep_start), window_start, L))
            ep_start = int(ep_end)

        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx, data):
        """Return a dict of arrays each of shape (sequence_length, *feature_shape).

        Parameters
        ----------
        idx : int
            Index into self.indices.
        data : dict
            Mapping from string key to np.ndarray of shape (N, *feature_shape).

        Returns
        -------
        dict
            Sampled sequences with repeat-edge padding applied.
        """
        ep_start, window_start, L = self.indices[idx]
        seq_len = self.sequence_length

        # Number of padding frames at each end
        n_pad_before = max(0, -window_start)
        n_pad_after = max(0, window_start + seq_len - L)

        # Slice of real data within the episode
        real_start_in_ep = max(0, window_start)
        real_end_in_ep = min(L, window_start + seq_len)

        buf_real_start = ep_start + real_start_in_ep
        buf_real_end = ep_start + real_end_in_ep

        out = {}
        for key, arr in data.items():
            real_slice = arr[buf_real_start:buf_real_end]   # (n_real, *feat)

            parts = []
            if n_pad_before > 0:
                first_frame = real_slice[:1]                # (1, *feat)
                pad = np.repeat(first_frame, n_pad_before, axis=0)
                parts.append(pad)
            parts.append(real_slice)
            if n_pad_after > 0:
                last_frame = real_slice[-1:]                # (1, *feat)
                pad = np.repeat(last_frame, n_pad_after, axis=0)
                parts.append(pad)

            seq = np.concatenate(parts, axis=0)             # (seq_len, *feat)
            out[key] = seq

        return out


# ---------------------------------------------------------------------------
# load_zarr_data
# ---------------------------------------------------------------------------

def load_zarr_data(path):
    """Load a diffusion_policy zarr store.

    Parameters
    ----------
    path : str
        Path to the zarr store directory or zip file.

    Returns
    -------
    data_dict : dict
        Mapping from key name to np.ndarray for all arrays under 'data/'.
    episode_ends : np.ndarray
        Shape (E,) cumulative EXCLUSIVE end indices.
        Episode i spans [episode_ends[i-1], episode_ends[i]).
        Episode 0 spans [0, episode_ends[0]).
    """
    store = zarr.open(path, 'r')

    data_dict = {}
    for key in store['data']:
        data_dict[key] = store['data'][key][:]

    episode_ends = store['meta']['episode_ends'][:]
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    return data_dict, episode_ends
