"""
datasets/topodiff_dataset.py — TopoDiff topology-optimization dataset
(Mazé & Ahmed, "Diffusion Models Beat GANs on Topology Optimization",
AAAI 2023, arXiv:2208.09591; data from
https://www.dropbox.com/sh/psaybuwcuh6b3ef/AAAtmDxF0RkooNrY9XqXjMVAa).

Task: given a rectangular 64x64 design domain with fixed boundary
conditions, applied point load(s), and a target volume fraction, generate
the stiffest (minimum-compliance) binary material layout achieving that
volume fraction — a 2D structural topology optimization problem, with SIMP
(Solid Isotropic Material with Penalization) ground-truth solutions.

Directory layout (after extracting dataset_1_diff.zip into `data_dir`):
    training_data/            — 30000 samples, IDs 0..29999. Each sample:
      gt_topo_<id>.png         — (64, 64) uint8 grayscale SIMP topology.
                                  Material is DARK (<127), void is BRIGHT
                                  (>=127) — see topo_to_tab in
                                  eval/topodiff_analysis.py.
      cons_pf_array_<id>.npy   — (64, 64, 3) float64: [VF (constant
                                  broadcast), von Mises stress field, strain
                                  energy density field].
      cons_load_array_<id>.npy — (64, 64, 2) float64: [load x-component
                                  field, load y-component field].
    training_data_summary.npy  — (30000,) object array of per-sample dicts:
                                  BC_conf, load_nodes, x_loads, y_loads, VF,
                                  load_coord (see eval/topodiff_eval.py).
    test_data_level_{1,2}/     — 1800 / 1000 samples. IDs are a contiguous
                                  but NOT zero-based range (level_1 starts at
                                  200, per direct inspection of the actual
                                  download — always derived from the files
                                  present, never hardcoded). Only the three
                                  cons_*_array_<id>.npy files (plus
                                  cons_bc_array_<id>.npy, unused here — see
                                  below) — NO gt_topo image. This is the
                                  paper's own official train/test split
                                  (level 1 = in-distribution BCs used to
                                  test "the full guided model"; level 2 =
                                  out-of-distribution BCs).
    test_data_level_{1,2}_summary.npy, _compliance.npy — sibling files:
                                  same per-sample summary dict shape as
                                  training, plus a separately precomputed
                                  reference SIMP compliance array (index k
                                  <-> file id (min_id + k), verified by
                                  direct inspection against cons_load_array's
                                  own encoded load position — NOT via the
                                  paper's own order_list/re_order_tab
                                  lexicographic reordering, which reconstructs
                                  a different (their-sampler-specific) file
                                  order that doesn't apply to summary/
                                  compliance indexing itself).

Conditioning channels: `condition_mode` selects what `spatial_conditions`
contains (always 5 channels either way, so the thinker/condition-encoder
config is identical for both):

  "full" (default) — channel-concat of cons_pf_array (3ch: VF, von Mises
    stress, strain energy density) + cons_load_array (2ch: load x, load y),
    matching the paper's own main diffusion model's conditioning exactly
    (see topodiff/image_datasets_diffusion_model.py: `concat([constraints_pf,
    loads], axis=2)`), no additional normalization beyond the raw float32
    values already baked into the .npy files at dataset-generation time.

  "hard" — [VF, load_x, load_y, BC_x, BC_y]: drops the von Mises/strain-
    energy fields (which already hand the model a solved FEA hint) in favor
    of the two raw boundary-condition node lists (BC_conf_x/BC_conf_y from
    the summary .npy — 1-indexed FEA node numbers on a 65x65 node grid, ';'-
    separated), rasterized to a 64x64 spatial field via
    rasterize_bc_field(). This is deliberately a harder task: the model has
    to infer how BC + loads propagate into stress/strain itself rather than
    being handed the solved fields directly.

    Rasterization convention (validated against test_data_level_1's real
    cons_bc_array ground truth, not just derived by hand — see
    conversation/commit history): decode each 1-indexed node id into its
    (X, Y) position on the 65x65 FEA node grid via
    X, Y = (node_id - 1) // 65, (node_id - 1) % 65 (matches
    eval/topodiff_analysis.py's create_files' own node numbering), scatter
    a 1.0 into a (65, 65) array at [Y, X], then apply
    eval.topodiff_analysis.resize() — the vendored paper code's own
    4-corner-average node-grid -> 64x64-element-grid downsampler (present
    in that file but otherwise unused elsewhere in this codebase) — exactly
    the operation needed here, since a constrained FEA *node* is naturally
    shared by up to 4 neighboring pixels/elements, unlike a point *load*
    (which lands on a single crisp pixel — cons_load_array's own
    convention, confirmed by checking its nonzero pixel against load_coord
    directly; resize() is the WRONG tool for loads specifically, which is
    why they're rasterized offline into cons_load_array with a different,
    single-pixel convention instead).

    Validated over 100 test_data_level_1 samples: 0 false-positive pixels,
    ~98.3% of true nonzero BC pixels reproduced exactly (3893/3960 for the
    x-channel). The ~1.7% miss rate is concentrated at domain CORNER nodes,
    where the source dataset's own BC_conf/BC_conf_x/BC_conf_y encoding is
    internally inconsistent about corner double-constraint (a genuine
    idiosyncrasy in the original data, not a bug in this rasterization —
    e.g. id=204's cons_bc_array marks corner node 4225 in its x-channel even
    though BC_conf lists it only as type=2/y-constrained). Given this
    channel exists specifically to make the task harder rather than to
    reproduce a metric, that residual is acceptable rather than something to
    chase further; not part of the eval, only of training-time conditioning.

cons_bc_array (the paper's own pre-rendered BC field, only present in the
test_data_level_* dirs — never in training_data, hence "hard" mode
rasterizes from the summary instead of reading it directly, so the SAME
code path works for both training and test) is otherwise NOT read directly.

Per this project's convention (unconditional stage-1 painter, all real
conditioning applied by the stage-2 thinker's ControlNet steering — see
Maze/Steiner/Polygon), `spatial_conditions` is NOT read by the default
topodiff_unet_painter experiment; it's the thinker's condition-encoder input
in topodiff_thinker_controlnet.

`images` (the diffusion target) is populated only for split="training_data"
(the only split with actual SIMP topology images); it's None for the
test_data_level_* splits, which exist purely to drive TopodiffEvalCallback's
periodic CE/VFE/LV/FM computation against the paper's own precomputed
reference compliance — see configs/eval_callbacks/topodiff*.yaml. Do not mix
splits within one DataLoader batch (collate_data_samples requires every
sample in a batch to agree on which fields are populated).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.data_sample import DataSample, collate_data_samples
from eval.topodiff_analysis import resize as _fea_node_grid_resize

IMAGE_SIZE = 64
_ID_RE = re.compile(r"_(\d+)\.npy$")
_CONDITION_MODES = ("full", "hard")


def rasterize_bc_field(node_str: str) -> np.ndarray:
    """';'-separated 1-indexed FEA node ids (65x65 node grid) -> (64, 64)
    float32 field, via a 1.0-per-node scatter onto the 65x65 node grid
    followed by the vendored paper code's own 4-corner-average downsampler
    (eval.topodiff_analysis.resize). See this module's docstring
    ("hard" condition_mode) for the validation this convention is based on."""
    arr = np.zeros((65, 65), dtype=np.float64)
    for tok in node_str.split(";"):
        if not tok:
            continue
        node0 = int(tok) - 1  # 0-indexed
        X, Y = node0 // 65, node0 % 65
        arr[Y, X] = 1.0
    return _fea_node_grid_resize(arr).astype(np.float32)


def _discover_ids(split_dir: str) -> list[int]:
    ids = sorted(int(_ID_RE.search(f).group(1)) for f in glob.glob(os.path.join(split_dir, "cons_load_array_*.npy")))
    if not ids:
        raise FileNotFoundError(f"No cons_load_array_*.npy files found under {split_dir}")
    expected = list(range(ids[0], ids[-1] + 1))
    if ids != expected:
        raise ValueError(f"{split_dir}: sample ids are not a contiguous range ({ids[0]}..{ids[-1]})")
    return ids


class TopodiffDataset(Dataset):
    """
    Args:
        data_dir: path to the extracted dataset_1_diff directory (containing
                  training_data/, test_data_level_1/, test_data_level_2/ and
                  their sibling _summary.npy / _compliance.npy files).
        split: "training_data" | "test_data_level_1" | "test_data_level_2".
        ids: optional explicit subset of puzzle_ids (file ids) to include.
             Defaults to every id present in the split directory.
        id_start, id_end: alternative to `ids` — a half-open [id_start,
             id_end) range, resolved against the ids actually present in the
             split directory (either bound may be omitted). Used to carve a
             held-out val split out of "training_data" without duplicating
             files on disk or spelling out an explicit id list in a Hydra
             config (e.g. id_end=29000 for train, id_start=29000 for val).
             Ignored if `ids` is given.
        condition_mode: "full" (default, VF+von Mises+SED+load_x+load_y) or
             "hard" (VF+load_x+load_y+BC_x+BC_y) — see this module's
             docstring. Same channel count (5) either way.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "training_data",
        ids: Optional[list[int]] = None,
        id_start: Optional[int] = None,
        id_end: Optional[int] = None,
        condition_mode: str = "full",
    ):
        super().__init__()
        if split not in ("training_data", "test_data_level_1", "test_data_level_2"):
            raise ValueError(f"unknown split {split!r}")
        if condition_mode not in _CONDITION_MODES:
            raise ValueError(f"unknown condition_mode {condition_mode!r}, expected one of {_CONDITION_MODES}")
        self.data_dir = data_dir
        self.split = split
        self.condition_mode = condition_mode
        self.split_dir = os.path.join(data_dir, split)
        self.has_images = split == "training_data"

        all_ids = _discover_ids(self.split_dir)
        self._min_id = all_ids[0]
        if ids is None and (id_start is not None or id_end is not None):
            ids = list(range(id_start if id_start is not None else all_ids[0],
                              id_end if id_end is not None else all_ids[-1] + 1))
        self.ids: list[int] = list(ids) if ids is not None else all_ids
        _available = set(all_ids)
        missing = [i for i in self.ids if i not in _available]
        if missing:
            raise ValueError(f"{self.split_dir}: requested ids not present (e.g. {missing[:5]})")

        summary_path = os.path.join(data_dir, f"{split}_summary.npy")
        self._summary = np.load(summary_path, allow_pickle=True, encoding="latin1")

        compliance_path = os.path.join(data_dir, f"{split}_compliance.npy")
        self._compliance = np.load(compliance_path) if os.path.exists(compliance_path) else None

    def __len__(self) -> int:
        return len(self.ids)

    def _summary_index(self, puzzle_id: int) -> int:
        return int(puzzle_id) - self._min_id

    def summary_for(self, puzzle_id: int) -> dict:
        """Raw per-sample dict (BC_conf, load_nodes, x_loads, y_loads, VF,
        load_coord) — the exact shape eval/topodiff_eval.evaluate_topodiff
        expects, unmodified from the source .npy."""
        return self._summary[self._summary_index(puzzle_id)]

    def compliance_for(self, puzzle_id: int) -> Optional[float]:
        """Paper's own precomputed reference SIMP compliance for this
        instance — only available for the test_data_level_* splits."""
        if self._compliance is None:
            return None
        return float(self._compliance[self._summary_index(puzzle_id)])

    def _load_condition(self, puzzle_id: int) -> np.ndarray:
        load = np.load(os.path.join(self.split_dir, f"cons_load_array_{puzzle_id}.npy"))  # (64, 64, 2)

        if self.condition_mode == "full":
            pf = np.load(os.path.join(self.split_dir, f"cons_pf_array_{puzzle_id}.npy"))  # (64, 64, 3)
            cond = np.concatenate([pf, load], axis=-1).astype(np.float32)  # (64, 64, 5): VF, vM, SED, load_x, load_y
        else:  # "hard"
            s = self.summary_for(puzzle_id)
            vf = np.full((IMAGE_SIZE, IMAGE_SIZE, 1), s["VF"], dtype=np.float32)
            bc_x = rasterize_bc_field(s["BC_conf_x"])[..., None]
            bc_y = rasterize_bc_field(s["BC_conf_y"])[..., None]
            cond = np.concatenate([vf, load, bc_x, bc_y], axis=-1).astype(np.float32)  # (64,64,5): VF, load_x, load_y, BC_x, BC_y

        return np.transpose(cond, (2, 0, 1))  # (5, 64, 64)

    def __getitem__(self, idx: int) -> DataSample:
        puzzle_id = self.ids[idx]
        spatial_conditions = torch.from_numpy(self._load_condition(puzzle_id))

        images = None
        if self.has_images:
            img = np.array(Image.open(os.path.join(self.split_dir, f"gt_topo_{puzzle_id}.png")))
            img = img.astype(np.float32) / 127.5 - 1.0  # {0,255} -> {-1,+1}; matches the painter's pixel_range="[-1,1]"
            images = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)

        return DataSample(
            images=images,
            spatial_conditions=spatial_conditions,
            puzzle_id=torch.tensor(puzzle_id, dtype=torch.long),
        )

    collate_fn = staticmethod(collate_data_samples)
