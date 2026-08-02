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
# Parallel worker processes, split the batch across processes
QUEEN_NPROC = _nproc("QUEEN_NPROC")
MAZE_NPROC = _nproc("MAZE_NPROC")

TRAIN_IMAGE_SIZE = int(os.environ.get("TRAIN_IMAGE_SIZE", "144"))

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
# Test files (native resolution, one parquet each):
#   test_queens/all_test.parquet, test_queens/n{n}_test.parquet
#   test_maze/all_test.parquet
#   test_maze/{shape}/all_{shape}_test.parquet
#   test_maze/{shape}/n{n}_{shape}_test.parquet
# Train leaves (train.parquet + validate.parquet; image_size in the leaf name):
#   train_queens/{all|n{n}}_train_size{S}/
#   train_maze/all_train_size{S}/                       (all shapes, all sizes)
#   train_maze/{shape}/{all_{shape}|n{n}_{shape}}_train_size{S}/
def test_maze_dir() -> Path:
    return OUT_ROOT / "test_maze"


def test_maze_all_file() -> Path:
    return test_maze_dir() / "all_test.parquet"


def test_maze_shape_dir(shape: str) -> Path:
    return test_maze_dir() / shape


def test_maze_shape_all_file(shape: str) -> Path:
    return test_maze_shape_dir(shape) / f"all_{shape}_test.parquet"


def test_maze_combo_file(shape: str, scale: int) -> Path:
    return test_maze_shape_dir(shape) / f"n{scale}_{shape}_test.parquet"


def test_queens_dir() -> Path:
    return OUT_ROOT / "test_queens"


def test_queens_all_file() -> Path:
    return test_queens_dir() / "all_test.parquet"


def test_queens_scale_file(scale: int) -> Path:
    return test_queens_dir() / f"n{scale}_test.parquet"


def _scope_tag(size) -> str:
    return "all" if str(size).upper() == "ALL" else f"n{size}"


def train_queens_dir(size, image_size: int = TRAIN_IMAGE_SIZE) -> Path:
    return OUT_ROOT / "train_queens" / f"{_scope_tag(size)}_train_size{image_size}"


def train_maze_dir(shape: str, size, image_size: int = TRAIN_IMAGE_SIZE) -> Path:
    root = OUT_ROOT / "train_maze"
    if str(shape).upper() == "ALL":
        return root / f"all_train_size{image_size}"
    tag = _scope_tag(size)
    leaf = f"all_{shape}_train_size{image_size}" if tag == "all" else f"{tag}_{shape}_train_size{image_size}"
    return root / shape / leaf


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


def ensure_maze_test_combo(shape: str, scale: int) -> None:
    """Generate one paper-spec test combo file (skip if already present)."""
    target = test_maze_combo_file(shape, scale)
    if target.exists():
        print(f"   {target.parent.name}/{target.name} already exists — skip")
        return
    entries = _maze_entries(
        shape, scale, MAZE_TEST_PER_SCALE, [TEST_ALGORITHM],
        seed_base=MAZE_TEST_SEED + scale * 10_000,
    )
    _gen_maze_split(entries, target.parent, target.name, train_ratio=0.0)


def ensure_maze_test_shape_all(shape: str) -> Path:
    """Ensure every scale combo for ``shape`` exists, then merge them into
    test_maze/{shape}/all_{shape}_test.parquet (skip existing)."""
    for scale in MAZE_SCALES:
        ensure_maze_test_combo(shape, scale)
    all_pq = test_maze_shape_all_file(shape)
    if not all_pq.exists():
        _merge_parquets([test_maze_combo_file(shape, s) for s in MAZE_SCALES], all_pq)
    return all_pq


def _maze_train_entries(shapes: list[str], scales: list[int]) -> list[dict]:
    """Build MAZE_TRAIN maze entries distributed uniformly across the shape×scale
    combos. Each combo gets a disjoint, position-stable seed range so the same
    (shape, scale) yields identical puzzles whether generated alone or mixed."""
    combos = [(g, s) for g in shapes for s in scales]
    per_combo = _split_count(MAZE_TRAIN, len(combos))
    entries: list[dict] = []
    for (g, s), cnt in zip(combos, per_combo):
        if cnt <= 0:
            continue
        gi, si = MAZE_GEOMETRIES.index(g), MAZE_SCALES.index(s)
        seed_base = MAZE_TRAIN_SEED + (gi * len(MAZE_SCALES) + si) * 100_000
        entries += _maze_entries(g, s, cnt, TRAIN_ALGOS.get(g, _UNIVERSAL), seed_base)
    return entries


def gen_maze_train(shape: str, size, image_size: int = TRAIN_IMAGE_SIZE) -> None:
    """Generate a maze train leaf (train.parquet, skip if present).

    shape=all         → all shapes, all sizes.
    shape, size=all   → one shape, all sizes.
    shape, size=n     → one shape, one size.
    """
    target_dir = train_maze_dir(shape, size, image_size)
    target = target_dir / "train.parquet"
    if target.exists():
        print(f">> {target_dir.name}/train.parquet already exists — skip")
        return

    if str(shape).upper() == "ALL":
        shapes, scales, desc = MAZE_GEOMETRIES, MAZE_SCALES, "all shapes × all sizes"
    elif str(size).upper() == "ALL":
        shapes, scales, desc = [shape], MAZE_SCALES, f"{shape} × all sizes"
    else:
        shapes, scales, desc = [shape], [int(size)], f"{shape} {size}×{size}"

    print(f">> generating {MAZE_TRAIN} train mazes: {desc}")
    entries = _maze_train_entries(shapes, scales)
    _gen_maze_split(entries, target_dir, "train.parquet", train_ratio=1.0)


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


def _gen_queens_mixed_pool(count: int, seed_base: int) -> tuple[Path, Path]:
    """Render ``count`` queen puzzles spread uniformly across QUEEN_SCALES and
    merge them into a single mixed-size parquet. Each puzzle keeps its own ``n``
    in the ``sample_json`` metadata, so the train set is a random mix of sizes.
    Returns (parquet_path, work_dir); the caller must rmtree work_dir."""
    work = Path(tempfile.mkdtemp(prefix="amaze_queen_mixed_"))
    try:
        per_scale = _split_count(count, len(QUEEN_SCALES))
        parts = []
        for scale, c in zip(QUEEN_SCALES, per_scale):
            if c <= 0:
                continue
            # Disjoint seed ranges per scale (>> per-worker offsets inside the pool).
            sub_produced, sub_work = _gen_queens_pool(scale, c, seed_base + scale * 100_000)
            dst = work / f"pool_n{scale}.parquet"
            shutil.move(str(sub_produced), str(dst))
            shutil.rmtree(sub_work, ignore_errors=True)
            parts.append(dst)
        produced = work / "maze_dataset_train.parquet"
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


def gen_queens_train(size, image_size: int = TRAIN_IMAGE_SIZE) -> None:
    """Generate a queen train leaf (train.parquet, skip if present).

    ``size`` is an int (single ``n=size`` board) or the string ``"ALL"`` (30k
    puzzles of mixed sizes uniformly sampled across QUEEN_SCALES)."""
    target_dir = train_queens_dir(size, image_size)
    target = target_dir / "train.parquet"
    if target.exists():
        print(f">> {target_dir.name}/train.parquet already exists — skip")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    if str(size).upper() == "ALL":
        print(f">> generating {QUEEN_TRAIN} train queen puzzles: mixed scales {QUEEN_SCALES}")
        produced, work = _gen_queens_mixed_pool(QUEEN_TRAIN, QUEEN_TRAIN_SEED)
    else:
        print(f">> generating {QUEEN_TRAIN} train queen puzzles: n={size}")
        produced, work = _gen_queens_pool(int(size), QUEEN_TRAIN, QUEEN_TRAIN_SEED)
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
    """Ensure the full maze test tree: per-shape combos, per-shape all_{shape}_test,
    and the top-level all_test (all shapes × all sizes). Skips existing files."""
    shape_all = [ensure_maze_test_shape_all(shape) for shape in MAZE_GEOMETRIES]
    all_pq = test_maze_all_file()
    if not all_pq.exists():
        _merge_parquets(shape_all, all_pq)
    return all_pq


def ensure_queens_test_all() -> Path:
    """Ensure the full queen test set: per-scale n{n}_test + merged all_test."""
    for scale in QUEEN_SCALES:
        ensure_queens_test_scale(scale)
    all_pq = test_queens_all_file()
    if not all_pq.exists():
        _merge_parquets([test_queens_scale_file(s) for s in QUEEN_SCALES], all_pq)
    return all_pq


def copy_val(train_dir: Path, test_parquet: Path) -> None:
    """Copy the matching test set into the train leaf as validate.parquet."""
    dst = train_dir / "validate.parquet"
    if dst.exists():
        print(f">> {train_dir.name}/validate.parquet already exists — skip")
        return
    shutil.copyfile(test_parquet, dst)
    print(f">> validate ← copy of {test_parquet.name} → {dst}")


# Train split reads only these two image columns (resized); the rest are eval-only.
_RESIZE_COLS = ("m_original_img", "sol_img")
_DROP_COLS = ("original_img", "mask_img", "cell_map")


def _resize_train_parquet(train_dir: Path, image_size: int | None = None) -> None:
    """Downsize the train split's images to ``image_size`` (default TRAIN_IMAGE_SIZE) in place (idempotent).

    The maze generator renders at ~102 px/cell (~830-1300px images); AmazeDataset
    would otherwise BICUBIC-antialias-resize each to the model input size on the CPU
    every epoch, making training data-loading-bound (hexagon ~8x slower than queens
    on identical model params). Precomputing that resize once -- byte-identical to
    AmazeDataset's transform -- turns per-step loading into a near no-op and shrinks
    the parquet. Only the train split is touched; val/test stay at native resolution
    for eval scoring, and the original is kept as *.orig.parquet. torch/PIL are
    imported lazily so plain test generation stays torch-free.
    """
    if image_size is None:
        image_size = TRAIN_IMAGE_SIZE
    if image_size <= 0:
        return
    train_pq = train_dir / "train.parquet"
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
    if probe is not None and max(probe.size) <= image_size:
        print(f">> train split already <= {image_size}px — skip resize")
        return

    resize = transforms.Resize(
        (image_size, image_size),
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
    print(f">> resized train → {image_size}px, dropped {dropped or '[]'}, backup {os.path.basename(backup)}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def _norm_size(value):
    """Return 'all' or an int for a --size token; None stays None."""
    if value is None:
        return None
    if str(value).upper() == "ALL":
        return "all"
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"--size must be 'all' or an integer, got '{value}'")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=True, description="Amaze dataset generator")
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("task", help="maze | queens (amaze == maze)")
    parser.add_argument("--shape", default=None,
                        help="maze train: all|" + "|".join(MAZE_GEOMETRIES))
    parser.add_argument("--size", default=None, help="train: all | <int>")
    parser.add_argument("--image-size", type=int, default=TRAIN_IMAGE_SIZE, dest="image_size")
    args = parser.parse_args(argv)

    task = "maze" if args.task in ("maze", "amaze") else args.task
    if task not in ("maze", "queens"):
        raise SystemExit(f"Unknown task '{args.task}' (use maze|queens)")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── TEST: always (re)generate the full tree, skipping what already exists. ──
    if args.mode == "test":
        all_pq = ensure_maze_test_all() if task == "maze" else ensure_queens_test_all()
        print(f"Done → {all_pq.parent}")
        return

    # ── TRAIN: only the requested --shape/--size combo. ───────────────────────
    size = _norm_size(args.size)
    if size is None:
        raise SystemExit("train requires --size <all|int>")

    if task == "maze":
        shape = args.shape
        if shape is None:
            raise SystemExit("maze train requires --shape <all|" + "|".join(MAZE_GEOMETRIES) + ">")
        if str(shape).upper() == "ALL":
            shape = "all"
            if size != "all":
                raise SystemExit("--shape all is only valid with --size all")
        elif shape not in MAZE_GEOMETRIES:
            raise SystemExit(f"Unknown --shape '{shape}' (use all|{'|'.join(MAZE_GEOMETRIES)})")
        if isinstance(size, int) and size not in MAZE_SCALES:
            raise SystemExit(f"--size {size} not in {MAZE_SCALES}")

        gen_maze_train(shape, size, args.image_size)
        if shape == "all":
            test_pq = ensure_maze_test_all()
        elif size == "all":
            test_pq = ensure_maze_test_shape_all(shape)
        else:
            ensure_maze_test_combo(shape, size)
            test_pq = test_maze_combo_file(shape, size)
        leaf = train_maze_dir(shape, size, args.image_size)
        copy_val(leaf, test_pq)
        _resize_train_parquet(leaf, args.image_size)
        print(f"Done → {leaf}")
    else:
        if args.shape is not None:
            print("note: --shape is ignored for queens")
        if isinstance(size, int) and size not in QUEEN_SCALES:
            raise SystemExit(f"--size {size} not in {QUEEN_SCALES}")

        gen_queens_train(size, args.image_size)
        if size == "all":
            test_pq = ensure_queens_test_all()
        else:
            ensure_queens_test_scale(size)
            test_pq = test_queens_scale_file(size)
        leaf = train_queens_dir(size, args.image_size)
        copy_val(leaf, test_pq)
        _resize_train_parquet(leaf, args.image_size)
        print(f"Done → {leaf}")


if __name__ == "__main__":
    main(sys.argv[1:])
