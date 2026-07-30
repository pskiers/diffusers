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


def _nproc(env_var: str) -> int:
    """Worker-process count: ``env_var`` override, else the SLURM CPU allocation,
    else the machine's cpu count (min 1)."""
    return max(1, int(os.environ.get(env_var)
                      or os.environ.get("SLURM_CPUS_PER_TASK")
                      or (os.cpu_count() or 1)))


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
# Downsize the train split to the model input resolution (data.image_size) after
# generation so per-step data loading isn't bound on the large source PNGs (the
# maze generator renders ~102 px/cell -> ~830-1300px images). 0 disables.
TRAIN_IMAGE_SIZE = int(os.environ.get("TRAIN_IMAGE_SIZE", "144"))
# Parallel worker processes, split the batch across processes
QUEEN_NPROC = _nproc("QUEEN_NPROC")
MAZE_NPROC = _nproc("MAZE_NPROC")

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


def _run_parallel(cmds: list) -> None:
    """Launch every command concurrently, wait for all, raise on any failure.
    Each item is either an argv list, or an ``(argv, cwd)`` tuple."""
    procs = []
    for item in cmds:
        cmd, cwd = item if isinstance(item, tuple) else (item, None)
        argv = [str(c) for c in cmd]
        procs.append((argv, subprocess.Popen(
            argv, cwd=(str(cwd) if cwd is not None else None),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    errors = []
    for argv, proc in procs:
        _out, err = proc.communicate()
        if proc.returncode != 0:
            errors.append(f"Command failed ({proc.returncode}): {' '.join(argv)}\n--- stderr ---\n{err}")
    if errors:
        raise RuntimeError("\n\n".join(errors))


def _split_count(count: int, nproc: int) -> list[int]:
    """Split ``count`` into as-even-as-possible positive chunks (<= nproc of them)."""
    nproc = max(1, min(nproc, count))
    base, rem = divmod(count, nproc)
    return [base + (1 if i < rem else 0) for i in range(nproc)]


def _split_list(items: list, nproc: int) -> list[list]:
    """Split a list into <= nproc as-even-as-possible non-empty sublists."""
    chunks, start = [], 0
    for size in _split_count(len(items), nproc):
        chunks.append(items[start:start + size])
        start += size
    return chunks


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
    ``out_dir/out_name``. Work is split across MAZE_NPROC parallel node workers
    (each rendering a chunk in its own dir), then the parquets are merged. Cleans
    up the (large) intermediate render dirs."""
    _ensure_node_deps()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="amaze_maze_"))
    try:
        chunks = _split_list(entries, MAZE_NPROC)
        want = "maze_dataset_train.parquet" if train_ratio >= 1.0 else "maze_dataset_test.parquet"
        print(f">> maze: {len(entries)} mazes across {len(chunks)} worker(s)", flush=True)

        node_cmds, proc_cmds, parts = [], [], []
        for i, chunk in enumerate(chunks):
            wk = work / f"w{i}"
            wk.mkdir()
            (wk / "cfg.json").write_text(json.dumps({"mazes": chunk}))
            # node writes generated_* relative to its cwd → give each worker its own.
            node_cmds.append((["node", MAZE_GEN / "batch-maze-generator.js", "config", wk / "cfg.json"], wk))
            proc_cmds.append([
                sys.executable, MAZE_GEN / "process_maze_into_parquet.py",
                "--maze-dir", wk / "generated_mazes",
                "--no-markers-dir", wk / "generated_mazes_no_markers",
                "--solution-dir", wk / "generated_solutions",
                "--metadata-dir", wk / "generated_metadata",
                "--output", wk / "maze_dataset.parquet",
                "--train-ratio", str(train_ratio),
                "--seed", "42",
            ])
            parts.append(wk / want)

        _run_parallel(node_cmds)     # the slow, CPU-bound render step — now on all cores
        _run_parallel(proc_cmds)
        for part in parts:
            if not part.exists():
                raise RuntimeError(f"Expected {part.name} not produced by process_maze_into_parquet.py")

        if len(parts) == 1:
            shutil.move(str(parts[0]), str(out_dir / out_name))
        else:
            _merge_parquets(parts, out_dir / out_name)
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
def _gen_queens_pool(scale: int, count: int, seed: int) -> tuple[Path, Path]:
    """Render ``count`` queen puzzles at n=scale and convert to a single parquet.
    The work is split across QUEEN_NPROC parallel worker processes (each with a
    distinct seed), then the per-worker parquets are merged. Returns
    (parquet_path, work_dir); the caller must rmtree work_dir."""
    work = Path(tempfile.mkdtemp(prefix="amaze_queen_"))
    try:
        chunks = _split_count(count, QUEEN_NPROC)
        print(f">> queens n={scale}: {count} puzzles across {len(chunks)} worker(s)", flush=True)

        gen_cmds, conv_cmds, parts = [], [], []
        for i, c in enumerate(chunks):
            raw_i = work / f"raw_{i}"
            pq_i = work / f"pq_{i}"
            raw_i.mkdir()
            pq_i.mkdir()
            gen_cmds.append([
                sys.executable, QUEEN_GEN / "generate_queens_puzzle.py",
                "--n", scale, "--count", c, "--outdir", raw_i, "--seed", seed + i,
                "--cell-size", QUEEN_CELL_SIZE, "--queen-radius", QUEEN_RADIUS, "--image-format", "png",
            ])
            conv_cmds.append([
                sys.executable, QUEEN_GEN / "convert_queen_to_parquet.py",
                "--queen-outdir", raw_i, "--dataset-outdir", pq_i, "--test-ratio", "0", "--seed", "42",
            ])
            parts.append(pq_i / "maze_dataset_train.parquet")

        _run_parallel(gen_cmds)     # the slow, CPU-bound step — now on all cores
        _run_parallel(conv_cmds)
        for part in parts:
            if not part.exists():
                raise RuntimeError("convert_queen_to_parquet.py did not produce maze_dataset_train.parquet")

        produced = work / "maze_dataset_train.parquet"
        if len(parts) == 1:
            shutil.move(str(parts[0]), str(produced))
        else:
            _merge_parquets(parts, produced)
        return produced, work
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


def ensure_queens_test_scale(scale: int) -> None:
    """Generate one paper-spec queen test scale file (skip if already present)."""
    target = test_queens_scale_file(scale)
    if target.exists():
        print(f"   test_queens/{target.name} already exists — skip")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    produced, work = _gen_queens_pool(scale, QUEEN_TEST_PER_SCALE, QUEEN_TEST_SEED + scale)
    try:
        shutil.move(str(produced), str(target))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def gen_queens_train(scale: int) -> None:
    """Generate a queen train set at n=scale (skip if already present)."""
    target = train_queens_dir(scale) / "maze_dataset_train.parquet"
    if target.exists():
        print(f">> {target.parent.name}/maze_dataset_train.parquet already exists — skip")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f">> generating {QUEEN_TRAIN} train queen puzzles: n={scale}")
    produced, work = _gen_queens_pool(scale, QUEEN_TRAIN, QUEEN_TRAIN_SEED)
    try:
        shutil.move(str(produced), str(target))
    finally:
        shutil.rmtree(work, ignore_errors=True)


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


# Train split reads only these two image columns (resized); the rest are eval-only.
_RESIZE_COLS = ("m_original_img", "sol_img")
_DROP_COLS = ("original_img", "mask_img", "cell_map")


def _resize_train_parquet(train_dir: Path) -> None:
    """Downsize the train split's images to TRAIN_IMAGE_SIZE in place (idempotent).

    The maze generator renders at ~102 px/cell (~830-1300px images); AmazeDataset
    would otherwise BICUBIC-antialias-resize each to the model input size on the CPU
    every epoch, making training data-loading-bound (hexagon ~8x slower than queens
    on identical model params). Precomputing that resize once -- byte-identical to
    AmazeDataset's transform -- turns per-step loading into a near no-op and shrinks
    the parquet. Only the train split is touched; val/test stay at native resolution
    for eval scoring, and the original is kept as *.orig.parquet. torch/PIL are
    imported lazily so plain test generation stays torch-free.
    """
    if TRAIN_IMAGE_SIZE <= 0:
        return
    train_pq = train_dir / "maze_dataset_train.parquet"
    if not train_pq.is_file():
        return

    import base64
    import io

    from PIL import Image
    from torchvision import transforms

    def _decode(raw):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
        if isinstance(raw, str):
            s = raw.split(",", 1)[1] if raw.startswith("data:") else raw
            return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        raise TypeError(f"Unsupported image cell type: {type(raw)}")

    def _to_png(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    df = pd.read_parquet(train_pq)
    present = [c for c in _RESIZE_COLS if c in df.columns]
    if not present:
        return
    probe = _decode(df.iloc[0][present[0]])
    if probe is not None and max(probe.size) <= TRAIN_IMAGE_SIZE:
        print(f">> train split already <= {TRAIN_IMAGE_SIZE}px — skip resize")
        return

    resize = transforms.Resize(
        (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    for col in present:
        out = []
        for v in df[col]:
            im = _decode(v)
            out.append(None if im is None else _to_png(resize(im)))
        df[col] = out
    dropped = [c for c in _DROP_COLS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)

    p = str(train_pq)
    backup = p[: -len(".parquet")] + ".orig.parquet"
    if not os.path.exists(backup):
        os.replace(p, backup)
    tmp = p + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    print(f">> resized train → {TRAIN_IMAGE_SIZE}px, dropped {dropped or '[]'}, backup {os.path.basename(backup)}")


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
        _resize_train_parquet(train_maze_dir(geom, scale))
        print(f"Done → {train_maze_dir(geom, scale)}")
    else:
        scale = n if n is not None else 7
        gen_queens_train(scale)
        all_pq = ensure_queens_test_all()
        copy_val(train_queens_dir(scale), all_pq)
        _resize_train_parquet(train_queens_dir(scale))
        print(f"Done → {train_queens_dir(scale)}")


if __name__ == "__main__":
    main(sys.argv[1:])
