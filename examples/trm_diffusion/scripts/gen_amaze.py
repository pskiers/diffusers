from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


TRM_ROOT = Path(__file__).resolve().parent.parent
MAZE_GEN = TRM_ROOT / "third_party" / "amaze" / "mazes-generator"
QUEEN_GEN = TRM_ROOT / "third_party" / "amaze" / "queen-generator"

# Output root — always data/amaze
OUT_ROOT = Path(os.environ.get("AMAZE_OUT_ROOT", str(TRM_ROOT / "data" / "amaze")))

# ── Paper test spec ──────────────────────────────────────────────────────────
MAZE_GEOMETRIES = ["square", "hexagon", "triangle", "circle"]
MAZE_SCALES = [3, 5, 7, 9, 11, 13, 16]          # 7 scales, 3×3 … 16×16 (circle: layers)
MAZE_TEST_PER_SCALE = int(os.environ.get("MAZE_TEST_PER_SCALE", "100"))  # DFS-only → 100 keeps 700/geom, 2800 total
QUEEN_SCALES = [4, 5, 6, 7, 8, 9, 10]            # 7 scales
QUEEN_TEST_PER_SCALE = int(os.environ.get("QUEEN_TEST_PER_SCALE", "50")) # 7 × 50 = 350

# ── Train sizes (env-overridable, mirror the legacy shell script) ────────────
MAZE_TRAIN = int(os.environ.get("MAZE_TRAIN", "30000"))
QUEEN_TRAIN = int(os.environ.get("QUEEN_TRAIN", "30000"))
QUEEN_CELL_SIZE = os.environ.get("QUEEN_CELL_SIZE", "64")
QUEEN_RADIUS = os.environ.get("QUEEN_RADIUS", "16")

_UNIVERSAL = ["recursiveBacktrack", "simplifiedPrims", "truePrims", "wilson", "aldousBroder", "huntAndKill"]
TRAIN_ALGOS = {
    "square": _UNIVERSAL + ["kruskal", "binaryTree", "sidewinder", "ellers"],
    "hexagon": _UNIVERSAL,
    "triangle": _UNIVERSAL,
    "circle": _UNIVERSAL,
}
TEST_ALGORITHM = "recursiveBacktrack"

# ── Seed bases (disjoint ranges so train never overlaps test) ────────────────
MAZE_TRAIN_SEED = 1_000_000
MAZE_TEST_SEED = 7_000_000       # per-combo: MAZE_TEST_SEED + scale*10_000 + i
QUEEN_TRAIN_SEED = 5_000_000
QUEEN_TEST_SEED = 8_000_000      # per-scale: QUEEN_TEST_SEED + scale


# ── Path helpers ─────────────────────────────────────────────────────────────
def test_maze_dir() -> Path:
    return OUT_ROOT / "test_maze"


def test_maze_combo_file(geometry: str, scale: int) -> Path:
    return test_maze_dir() / f"{geometry}_{scale}.parquet"


def test_maze_all_file() -> Path:
    return test_maze_dir() / "all_test.parquet"


def test_queens_dir() -> Path:
    return OUT_ROOT / "test_queens"


def test_queens_scale_file(scale: int) -> Path:
    return test_queens_dir() / f"n{scale}.parquet"


def test_queens_all_file() -> Path:
    return test_queens_dir() / "all_test.parquet"


def train_maze_dir(geometry: str, scale: int) -> Path:
    return OUT_ROOT / f"train_maze_{geometry}_n{scale}"


def train_queens_dir(scale: int) -> Path:
    return OUT_ROOT / f"train_queens_n{scale}"


# ── Subprocess helper ────────────────────────────────────────────────────────
def _run(cmd, cwd=None) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    print(">>", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed ({res.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
        )
    return res


def _ensure_node_deps() -> None:
    """One-time `npm install` for the maze generator (needs internet)."""
    if (MAZE_GEN / "node_modules").is_dir():
        return
    print(f"Installing node deps in {MAZE_GEN} (needs internet — run on a login node)…", flush=True)
    _run(["npm", "install"], cwd=MAZE_GEN)


# ── Maze generation ──────────────────────────────────────────────────────────
def _maze_entries(geometry: str, scale: int, count: int, algorithms: list[str], seed_base: int) -> list[dict]:
    entries = []
    for i in range(count):
        entry = {
            "shape": geometry,
            "algorithm": algorithms[i % len(algorithms)],
            "exitConfig": "hardest",
            "seed": seed_base + i,
            "filename": f"{geometry}_{scale}_{i:06d}.png",
        }
        if geometry == "circle":
            entry["layers"] = scale          # circle scale = number of layers
        else:
            entry["width"] = scale
            entry["height"] = scale          # paper-spec square dims (n×n)
        entries.append(entry)
    return entries


def _gen_maze_split(entries: list[dict], out_dir: Path, out_name: str, train_ratio: float) -> None:
    """Render ``entries`` with the node generator, convert to parquet, and move
    the requested split file (train_ratio=1.0 → *_train, 0.0 → *_test) to
    ``out_dir/out_name``. Cleans up the (large) intermediate render dir."""
    _ensure_node_deps()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="amaze_maze_"))
    try:
        (work / "cfg.json").write_text(json.dumps({"mazes": entries}))
        _run(["node", MAZE_GEN / "batch-maze-generator.js", "config", work / "cfg.json"], cwd=work)
        _run([
            sys.executable, MAZE_GEN / "process_maze_into_parquet.py",
            "--maze-dir", work / "generated_mazes",
            "--no-markers-dir", work / "generated_mazes_no_markers",
            "--solution-dir", work / "generated_solutions",
            "--metadata-dir", work / "generated_metadata",
            "--output", work / "maze_dataset.parquet",
            "--train-ratio", str(train_ratio),
            "--seed", "42",
        ])
        produced = work / ("maze_dataset_train.parquet" if train_ratio >= 1.0 else "maze_dataset_test.parquet")
        if not produced.exists():
            raise RuntimeError(f"Expected {produced.name} not produced by process_maze_into_parquet.py")
        shutil.move(str(produced), str(out_dir / out_name))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ensure_maze_test_combo(geometry: str, scale: int) -> None:
    """Generate one paper-spec test combo file (skip if already present)."""
    target = test_maze_combo_file(geometry, scale)
    if target.exists():
        print(f"   test_maze/{target.name} already exists — skip")
        return
    entries = _maze_entries(
        geometry, scale, MAZE_TEST_PER_SCALE, [TEST_ALGORITHM],
        seed_base=MAZE_TEST_SEED + scale * 10_000,
    )
    _gen_maze_split(entries, target.parent, target.name, train_ratio=0.0)


def gen_maze_train(geometry: str, scale: int) -> None:
    """Generate a single-geometry train set (skip if already present)."""
    target = train_maze_dir(geometry, scale) / "maze_dataset_train.parquet"
    if target.exists():
        print(f">> {target.parent.name}/maze_dataset_train.parquet already exists — skip")
        return
    algos = TRAIN_ALGOS.get(geometry, _UNIVERSAL)
    print(f">> generating {MAZE_TRAIN} train mazes: {geometry} {scale}×{scale} (algos={len(algos)})")
    entries = _maze_entries(geometry, scale, MAZE_TRAIN, algos, seed_base=MAZE_TRAIN_SEED)
    _gen_maze_split(entries, target.parent, "maze_dataset_train.parquet", train_ratio=1.0)


# ── Queen generation ─────────────────────────────────────────────────────────
def _gen_queens_pool(scale: int, count: int, seed: int) -> tuple[Path, Path, Path]:
    """Render ``count`` queen puzzles at n=scale and convert to a single parquet
    (everything into *_train via --test-ratio 0). Returns the parquet path in a
    temp dir the caller must clean up."""
    raw = Path(tempfile.mkdtemp(prefix="amaze_queen_raw_"))
    pq = Path(tempfile.mkdtemp(prefix="amaze_queen_pq_"))
    try:
        _run([
            sys.executable, QUEEN_GEN / "generate_queens_puzzle.py",
            "--n", scale, "--count", count, "--outdir", raw, "--seed", seed,
            "--cell-size", QUEEN_CELL_SIZE, "--queen-radius", QUEEN_RADIUS, "--image-format", "png",
        ])
        _run([
            sys.executable, QUEEN_GEN / "convert_queen_to_parquet.py",
            "--queen-outdir", raw, "--dataset-outdir", pq, "--test-ratio", "0", "--seed", "42",
        ])
        produced = pq / "maze_dataset_train.parquet"
        if not produced.exists():
            raise RuntimeError("convert_queen_to_parquet.py did not produce maze_dataset_train.parquet")
        return produced, raw, pq
    except Exception:
        shutil.rmtree(raw, ignore_errors=True)
        shutil.rmtree(pq, ignore_errors=True)
        raise


def ensure_queens_test_scale(scale: int) -> None:
    """Generate one paper-spec queen test scale file (skip if already present)."""
    target = test_queens_scale_file(scale)
    if target.exists():
        print(f"   test_queens/{target.name} already exists — skip")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    produced, raw, pq = _gen_queens_pool(scale, QUEEN_TEST_PER_SCALE, QUEEN_TEST_SEED + scale)
    try:
        shutil.move(str(produced), str(target))
    finally:
        shutil.rmtree(raw, ignore_errors=True)
        shutil.rmtree(pq, ignore_errors=True)


def gen_queens_train(scale: int) -> None:
    """Generate a queen train set at n=scale (skip if already present)."""
    target = train_queens_dir(scale) / "maze_dataset_train.parquet"
    if target.exists():
        print(f">> {target.parent.name}/maze_dataset_train.parquet already exists — skip")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f">> generating {QUEEN_TRAIN} train queen puzzles: n={scale}")
    produced, raw, pq = _gen_queens_pool(scale, QUEEN_TRAIN, QUEEN_TRAIN_SEED)
    try:
        shutil.move(str(produced), str(target))
    finally:
        shutil.rmtree(raw, ignore_errors=True)
        shutil.rmtree(pq, ignore_errors=True)


# ── Merge + val copy ─────────────────────────────────────────────────────────
def _merge_parquets(parts: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(p) for p in parts if p.exists()]
    if not frames:
        raise RuntimeError(f"No parquet parts to merge into {out_path}")
    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(out_path, index=False, compression="snappy")
    print(f">> merged {len(frames)} parts ({len(merged)} rows) → {out_path}")


def ensure_maze_test_all() -> Path:
    """Ensure the full diversified maze test set + merged all_test file exist."""
    for geometry in MAZE_GEOMETRIES:
        for scale in MAZE_SCALES:
            ensure_maze_test_combo(geometry, scale)
    all_pq = test_maze_all_file()
    if not all_pq.exists():
        parts = [test_maze_combo_file(g, s)
                 for g in MAZE_GEOMETRIES for s in MAZE_SCALES]
        _merge_parquets(parts, all_pq)
    return all_pq


def ensure_queens_test_all() -> Path:
    """Ensure the full diversified queen test set + merged all_test file exist."""
    for scale in QUEEN_SCALES:
        ensure_queens_test_scale(scale)
    all_pq = test_queens_all_file()
    if not all_pq.exists():
        parts = [test_queens_scale_file(s) for s in QUEEN_SCALES]
        _merge_parquets(parts, all_pq)
    return all_pq


def copy_val(train_dir: Path, test_all_parquet: Path) -> None:
    """Copy the merged diversified test set into the train dir as the val split."""
    dst = train_dir / "maze_dataset_val.parquet"
    if dst.exists():
        print(f">> {train_dir.name}/maze_dataset_val.parquet already exists — skip")
        return
    shutil.copyfile(test_all_parquet, dst)
    print(f">> val ← copy of {test_all_parquet} → {dst}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_kv(tokens: list[str]) -> dict[str, str]:
    kv = {}
    for tok in tokens:
        if "=" not in tok:
            raise SystemExit(f"Expected key=value, got '{tok}'")
        k, v = tok.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=True, description="Amaze dataset generator")
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("task", help="maze | queens (amaze == maze)")
    parser.add_argument("kv", nargs="*", help="n=<int> type=<geom>")
    args = parser.parse_args(argv)

    task = "maze" if args.task in ("maze", "amaze") else args.task
    if task not in ("maze", "queens"):
        raise SystemExit(f"Unknown task '{args.task}' (use maze|queens)")
    kv = _parse_kv(args.kv)
    n = int(kv["n"]) if "n" in kv else None
    geom = kv.get("type")

    if task == "maze" and geom is not None and geom not in MAZE_GEOMETRIES:
        raise SystemExit(f"Unknown maze type '{geom}' (use {'|'.join(MAZE_GEOMETRIES)})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.mode == "test":
        # Full paper-spec test set (n=/type= are ignored — the test set is the
        # complete diversified sweep).
        if task == "maze":
            all_pq = ensure_maze_test_all()
        else:
            all_pq = ensure_queens_test_all()
        print(f"Done → {all_pq.parent}")
        return

    # mode == "train": single-geometry/scale train + val (= copy of the test set, generated once if absent).
    if task == "maze":
        if geom is None:
            raise SystemExit("maze train requires type=<geometry>")
        scale = n if n is not None else 8
        gen_maze_train(geom, scale)
        all_pq = ensure_maze_test_all()
        copy_val(train_maze_dir(geom, scale), all_pq)
        print(f"Done → {train_maze_dir(geom, scale)}")
    else:
        scale = n if n is not None else 7
        gen_queens_train(scale)
        all_pq = ensure_queens_test_all()
        copy_val(train_queens_dir(scale), all_pq)
        print(f"Done → {train_queens_dir(scale)}")


if __name__ == "__main__":
    main(sys.argv[1:])
