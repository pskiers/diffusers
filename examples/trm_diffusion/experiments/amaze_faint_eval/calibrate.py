"""Sweep the faint detector's B-R tint threshold.

For each shape, take up to N boards per size and count, per threshold, how many
boards get extra/missing cells when scoring
  * the JPEG ground-truth image infer_bagel saved next to the attempts (`*_gt.jpg`)
    -> should be perfect at any usable threshold (noise floor), and
  * the generated attempt001 -> shows how many ghost cells each threshold catches.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from faint_metrics import (BAGEL_RUNS, FaintAmazeMetrics, attempt_id, faint_blue_mask,  # noqa: E402
                           load_native, load_shape_index, solid_blue_mask)

# Usage: calibrate.py [boards_per_shape] [shape ...]
N_BOARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 105
SHAPES = sys.argv[2:] or list(BAGEL_RUNS)
TINTS = [3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
M = FaintAmazeMetrics()


def cells(mask, cell_ids, ignored):
    return M._cells_covered(M._morph_open(mask), cell_ids) - ignored


for shape in SHAPES:
    gen_dir = BAGEL_RUNS[shape]
    # Filename prefixes don't reliably encode size (circle is always "10x10"), so take
    # a seeded random sample of boards; uuids are random, so sizes mix evenly.
    files = sorted(gen_dir.glob("*_attempt001.jpg"))
    files = [files[i] for i in np.random.default_rng(0).permutation(len(files))[:N_BOARDS]]
    index = load_shape_index(shape, wanted={attempt_id(f) for f in files})
    per_size = defaultdict(list)
    for f in files:
        pid = attempt_id(f)
        if pid in index:
            per_size[index[pid][0]].append((pid, f))

    # stats[src][tint or 'solid'] = [boards_with_extra, boards_with_missing, total_extra_cells]
    stats = {src: defaultdict(lambda: [0, 0, 0]) for src in ("gt", "gen")}
    n = 0
    for s, items in sorted(per_size.items()):
        for pid, f in items:
            md = index[pid][1]
            import json
            meta = json.loads(md["metadata"]) if isinstance(md["metadata"], str) else md["metadata"]
            path_ids = [int(c) for c in meta["path_cell_ids"]]
            cell_ids = M.decode_cell_map_ids(md["cell_map"])
            size = (cell_ids.shape[1], cell_ids.shape[0])
            ignored = M._endpoint_cells(cell_ids, (path_ids[0], path_ids[-1]))
            gt = set(path_ids) - ignored
            srcs = {"gt": f.with_name(f.name.replace("_attempt001.jpg", "_gt.jpg")), "gen": f}
            for src, p in srcs.items():
                arr = M._to_pixel_array(load_native(p), size)
                masks = {"solid": solid_blue_mask(arr)}
                masks.update({t: faint_blue_mask(arr, t) for t in TINTS})
                for key, mk in masks.items():
                    pred = cells(mk, cell_ids, ignored)
                    st = stats[src][key]
                    st[0] += bool(pred - gt)
                    st[1] += bool(gt - pred)
                    st[2] += len(pred - gt)
            n += 1

    print(f"\n===== {shape}: {n} boards (sizes {sorted(per_size)}) =====")
    print(f"{'detector':>10} | {'GT.jpg: brd w/ extra':>20} {'brd w/ missing':>14} | "
          f"{'GEN: brd w/ extra':>17} {'brd w/ missing':>14} {'extra cells':>11}")
    for key in ["solid"] + TINTS:
        g, q = stats["gt"][key], stats["gen"][key]
        label = "solid(orig)" if key == "solid" else f"tint>={key}"
        print(f"{label:>10} | {g[0]:>20} {g[1]:>14} | {q[0]:>17} {q[1]:>14} {q[2]:>11}")
