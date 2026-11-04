"""TEMPORARY (Claude): discrete square mazes for a token-per-cell TRM (no image tokens, no cell/token misalignment).

Encoding on a GRID x GRID token grid (the maze in the top-left n x n corner, the rest padding):
  token id = 0 for padding, else 1 + open + 16 * marker,  open = 4 bits (up 1, right 2, down 4, left 8),
  marker 0 none / 1 start / 2 goal          -> 49 ids
  label    = 1 on the solution, 0 other maze cells, -1 padding
Walls come from the rendered maze through the same code as every analysis (mcl.build_maze_sample + check_loops.tree), so
training mazes (generated here) and test mazes (the amaze test parquets) are discretised identically.

  gen_train     run the vendored amaze generator (node) for square mazes with the seeds / algorithms of the amaze training
                pool (scripts/gen_amaze.py: seed 1000000 + size_index * 100000 + i, algorithm TRAIN_ALGORITHMS[i % 10]);
                the first 1071 per size are the square mazes of the PaTh / plain TRM training set
  convert_test  the square test parquets data/amaze/maze/square/n<size>_test.parquet

python experiments/discrete_maze.py gen_train --per_size 4284 --out data/discrete/train.npz --nproc 16
python experiments/discrete_maze.py convert_test --sizes 5,7,8,9,10,11,13,16 --out_dir data/discrete
"""
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import maze_corruption_lib as mcl
from check_loops import tree

GRID = 16
TRAIN_SIZES = [5, 7, 8, 9, 11, 13, 16]
ALGOS = ["recursiveBacktrack", "simplifiedPrims", "truePrims", "wilson", "aldousBroder", "huntAndKill",
         "kruskal", "binaryTree", "sidewinder", "ellers"]
GEN = Path(__file__).resolve().parent.parent / "third_party" / "amaze" / "mazes-generator"
DIRS = {(-1, 0): 1, (0, 1): 2, (1, 0): 4, (0, -1): 8}


def encode(nb, n, start, goal, path_cells, grid=GRID):
    """nb: {(r, c): [neighbour cells with an open passage]} -> ids (grid*grid,), labels (grid*grid,)."""
    ids = np.zeros(grid * grid, np.int64)
    lab = np.full(grid * grid, -1, np.int64)
    for (r, c), nbs in nb.items():
        o = sum(DIRS[(r2 - r, c2 - c)] for r2, c2 in nbs)
        m = 1 if (r, c) == start else 2 if (r, c) == goal else 0
        ids[r * grid + c] = 1 + o + 16 * m
        lab[r * grid + c] = int((r, c) in path_cells)
    return ids, lab


def from_sample(ms):
    """MazeSample -> (nb, n, start, goal, path cells)."""
    nb, _ = tree(ms)
    rc = [ms.id2rc[p] for p in ms.path]
    return nb, ms.width, rc[0], rc[-1], set(rc)


def _parse_one(args):
    meta_path, maze_png, sol_png, cmap_png = args
    meta = json.loads(Path(meta_path).read_text())
    md = {"metadata": meta, "sol_img": Image.open(sol_png), "m_original_img": Image.open(maze_png), "cell_map": Image.open(cmap_png)}
    ms = mcl.build_maze_sample(md)
    if ms is None:
        return None
    nb, n, s, g, path = from_sample(ms)
    if sum(len(x) for x in nb.values()) // 2 != len(nb) - 1:
        return None                                                     # not read as a perfect maze
    ids, lab = encode(nb, n, s, g, path)
    return ids, lab, n, int(meta["maze_config"]["seed"]), meta["maze_config"]["algorithm"]


def gen_train(a):
    entries = []
    for si, size in enumerate(TRAIN_SIZES):
        base = 1_000_000 + si * 100_000                                    # square = shape index 0
        for i in range(a.per_size):
            entries.append({"shape": "square", "algorithm": ALGOS[i % len(ALGOS)], "exitConfig": "hardest",
                            "seed": base + i, "filename": f"square_{size}_{i:06d}.png", "width": size, "height": size})
    work = Path(tempfile.mkdtemp(prefix="discrete_maze_", dir=a.tmp))
    chunks = [entries[k::a.nproc] for k in range(a.nproc)]
    procs = []
    for k, ch in enumerate(chunks):
        w = work / f"w{k}"; w.mkdir()
        (w / "cfg.json").write_text(json.dumps({"mazes": ch}))
        procs.append(subprocess.Popen(["node", str(GEN / "batch-maze-generator.js"), "config", "cfg.json"], cwd=w,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    jobs = []
    for k in range(a.nproc):
        w = work / f"w{k}"
        for e in chunks[k]:
            stem = e["filename"][:-4]
            jobs.append((w / "generated_metadata" / f"{stem}.json", w / "generated_mazes" / f"{stem}.png",
                         w / "generated_solutions" / f"{stem}_solution.png", w / "generated_metadata" / f"{stem}_cell_map.png"))
    jobs = [j for j in jobs if all(Path(x).exists() for x in j)]
    with ProcessPoolExecutor(a.nproc) as ex:
        out = [r for r in ex.map(_parse_one, jobs, chunksize=32) if r is not None]
    ids, lab, n, seed, alg = zip(*out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, ids=np.stack(ids), labels=np.stack(lab), n=np.array(n), seed=np.array(seed), algorithm=np.array(alg))
    print(f"{len(out)} of {len(entries)} mazes -> {a.out}; per size " +
          ", ".join(f"{s}: {int((np.array(n) == s).sum())}" for s in TRAIN_SIZES), flush=True)
    if not a.keep_images:
        subprocess.run(["rm", "-rf", str(work)])


def convert_test(a):
    from datasets.amaze_dataset import AmazeDataset
    for size in [int(s) for s in a.sizes.split(",")]:
        ds = AmazeDataset(dataset_path=f"data/amaze/maze/square/n{size}_test.parquet", split="test", image_size=144,
                          condition_field="m_original_img", target_field="sol_img", num_channels=3, include_metadata=True)
        ids, lab, idx = [], [], []
        for i in range(len(ds)):
            ms = mcl.build_maze_sample(ds[i].metadata)
            if ms is None or ms.width != size or size > GRID:
                continue
            x, y = encode(*from_sample(ms))
            ids.append(x); lab.append(y); idx.append(i)
        out = Path(a.out_dir) / f"n{size}_test.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, ids=np.stack(ids), labels=np.stack(lab), n=np.full(len(ids), size), index=np.array(idx))
        print(f"n{size}: {len(ids)} of {len(ds)} test mazes -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen_train"); g.add_argument("--per_size", type=int, default=1071); g.add_argument("--out", required=True)
    g.add_argument("--nproc", type=int, default=os.cpu_count()); g.add_argument("--tmp", default=None)
    g.add_argument("--keep_images", action="store_true")
    c = sub.add_parser("convert_test"); c.add_argument("--sizes", default="5,7,8,9,10,11,13,16"); c.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    gen_train(a) if a.cmd == "gen_train" else convert_test(a)
