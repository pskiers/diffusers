"""
Generate mazes using the AMAZE algorithm.
Data structure:
|- queens/
    |--- maze_dataset_train.parquet
    |--- maze_dataset_test.parquet
    |--- n{size}_test.parquet    # for each size n
|- maze/
    |--- maze_dataset_train.parquet
    |--- maze_dataset_test.parquet
    |--- circle/
        |--- all_test.parquet
        |--- n{size}_test.parquet
    |--- square/
        |--- all_test.parquet
        |--- n{size}_test.parquet
    |--- hexagon/
        |--- all_test.parquet
        |--- n{size}_test.parquet
    |--- triangle
        |--- all_test.parquet
        |--- n{size}_test.parquet
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from matplotlib.pylab import size
import pandas as pd
from PIL import Image
import typing

# Cap native thread pools before importing pandas.
for _thr_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS", "RAYON_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thr_var, "1")


_RESIZE_COLS = ("m_original_img", "sol_img")
_DROP_COLS = ("original_img", "mask_img", "cell_map")


def _resize_train_parquet(train_dir: Path, image_size: int | None = None) -> None:
    # Precompute the train-split resize once (byte-identical to AmazeDataset's transform)
    # so training isn't data-loading-bound; keep the native-res original as *.orig.parquet.
    if image_size is None:
        image_size = TRAIN_IMAGE_SIZE
    if image_size <= 0:
        return
    train_pq = train_dir / "train.parquet"
    if not train_pq.is_file():
        return

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


_FT_IMAGE_COLS = ("original_img", "m_original_img", "sol_img", "mask_img", "cell_map")



# --- New implementation
def dataset_generator_factory(puzzle_type: typing.Literal["maze", "queens"], size, shape):
    if puzzle_type not in ("maze", "queens"):
        raise SystemExit(f"Unknown task '{puzzle_type}' (use maze|queens)")
    if puzzle_type == "maze":
        return MazeDatasetGenerator(size, shape)
    if puzzle_type == "queens":
        return QueensDatasetGenerator(size)


class AmazeDatasetGenerator:
    def __init__(self, size, shape=None):
        self.trm_root = Path(__file__).resolve().parent.parent
        self.out_root = Path(os.environ.get("AMAZE_OUT_ROOT", str(self.trm_root / "data" / "amaze")))
        self.maze_gen = self.trm_root / "third_party" / "amaze" / "mazes-generator"
        self.queen_gen = self.trm_root / "third_party" / "amaze" / "queen-generator"

        self.shape = shape
        self.size = size

    def generate(self):
        raise NotImplementedError("Subclasses must implement the generate method.")

    def _merge_parquets(self, parts: list[Path], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames = [pd.read_parquet(p) for p in parts if p.exists()]
        if not frames:
            raise RuntimeError(f"No parquet parts to merge into {out_path}")
        merged = pd.concat(frames, ignore_index=True)
        merged.to_parquet(out_path, index=False, compression="snappy")
        print(f">> merged {len(frames)} parts ({len(merged)} rows) → {out_path}")

    def _nproc(self, env_var: str) -> int:
        return max(1,
            int(os.environ.get(env_var) or os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))
        )

    def _norm_size(self, value):
        if value is None:
            return None
        if str(value).upper() == "ALL":
            return "all"
        try:
            return int(value)
        except ValueError:
            raise SystemExit(f"--size must be 'all' or an integer, got '{value}'")

    def _scope_tag(self, size) -> str:
        return "all" if str(size).upper() == "ALL" else f"n{size}"

    # PARALLEL HELPERS
    def _split_count(self, count: int, nproc: int) -> list[int]:
        nproc = max(1, min(nproc, count))
        base, rem = divmod(count, nproc)
        return [base + (1 if i < rem else 0) for i in range(nproc)]

    def _split_list(self, items: list, nproc: int) -> list[list]:
        chunks, start = [], 0
        for size in self._split_count(len(items), nproc):
            chunks.append(items[start:start + size])
            start += size
        return chunks

    def _run_parallel(self, cmds: list) -> None:
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
                rc = proc.returncode
                note = ""
                if rc < 0:
                    try:
                        note = f" [killed by {signal.Signals(-rc).name}]"
                    except Exception:
                        note = f" [killed by signal {-rc}]"
                errors.append(f"Command failed ({rc}){note}: {' '.join(argv)}\n--- stderr ---\n{err}")
        if errors:
            raise RuntimeError("\n\n".join(errors))


class QueensDatasetGenerator(AmazeDatasetGenerator):
    """Queens dataset generator. Data structure:
    |- queens
    |--- maze_dataset_train.parquet
    |--- maze_dataset_test.parquet
    |--- n{size}_test.parquet    # for each size n
    """
    def __init__(self, size):
        super().__init__(size, None)
        self.scales = [4, 5, 6, 7, 8, 9, 10]
        if isinstance(size, int) and size not in self.scales:
            raise SystemExit(f"--size {size} not in {self.scales}")

        self.train_num_samples = int(os.environ.get("QUEEN_TRAIN", "30000"))
        self.test_per_scale_num_samples = {
            scale: int(os.environ.get(f"QUEEN_TEST_PER_SCALE_{scale}", "50"))
            for scale in self.scales
        }

        self.queen_cell_size = os.environ.get("QUEEN_CELL_SIZE", "64")
        self.queen_radius = os.environ.get("QUEEN_RADIUS", "16")
        self.queen_nproc = self._nproc("QUEEN_NPROC")

        self.train_seed = 5_000_000
        self.test_seed = 8_000_000

    def generate(self):
        data_paths = self._get_data_paths()
        data_paths["train"].parent.mkdir(parents=True, exist_ok=True)

        # Test per size
        for scale in self.scales:
            count = self.test_per_scale_num_samples[scale]
            n_scale_board_file_name = data_paths["scale_tests"][scale]

            if Path(n_scale_board_file_name).exists():
                print(f"File {n_scale_board_file_name.name} already exists — skip")
                continue

            produced, work = self._gen_queens_pool(scale, count, self.test_seed + scale)
            try:
                shutil.move(str(produced), str(n_scale_board_file_name))
            finally:
                shutil.rmtree(work, ignore_errors=True)

        # Test all sizes
        all_sizes_test_parquet = data_paths["test"]
        if not all_sizes_test_parquet.exists():
            self._merge_parquets([data_paths["scale_tests"][s] for s in self.scales], all_sizes_test_parquet)

        # Train (all sizes)
        self.gen_queens_train(data_paths)
        print(f"Done, everything in {data_paths['train'].parent} directory")


    def gen_queens_train(self, data_paths) -> None:
        maze_train_path: Path = data_paths["train"]
        if maze_train_path.exists():
            print(f">> {maze_train_path.name} already exists — skip")
            return

        # Train is always for all sizes
        print(f">> generating {str(maze_train_path.resolve())} train queen puzzles: mixed scales {self.scales}")

        temp_dir = Path(tempfile.mkdtemp(prefix="amaze_queen_mixed_"))
        try:
            print(f">> queen mixed per-scale counts: {data_paths['scale_tests']}", flush=True)
            parts = []
            for scale in self.scales:
                c = data_paths["scale_tests"].get(scale, 0)
                if c <= 0:
                    continue

                sub_produced, sub_work = self._gen_queens_pool(scale, c, self.train_seed + scale * 100_000)
                dst = temp_dir / f"pool_n{scale}.parquet"

                shutil.move(str(sub_produced), str(dst))
                shutil.rmtree(sub_work, ignore_errors=True)
                parts.append(dst)

            produced = temp_dir / "maze_dataset_train.parquet"
            self._merge_parquets(parts, produced)
            shutil.move(str(produced), str(maze_train_path))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _gen_queens_pool(self, scale: int, count: int, seed: int) -> tuple[Path, Path]:
        """Use vendored scripts to generate queens boards and convert them to parquet format."""
        work = Path(tempfile.mkdtemp(prefix="amaze_queen_"))
        try:
            chunks = self._split_count(count, self.queen_nproc)
            print(f">> queens n={scale}: {count} puzzles across {len(chunks)} worker(s)", flush=True)

            gen_cmds, conv_cmds, parts = [], [], []
            for i, c in enumerate(chunks):
                raw_i = work / f"raw_{i}"
                pq_i = work / f"pq_{i}"
                raw_i.mkdir()
                pq_i.mkdir()
                gen_cmds.append([
                    sys.executable, self.queen_gen / "generate_queens_puzzle.py",
                    "--n", scale, "--count", c, "--outdir", raw_i, "--seed", seed + i,
                    "--cell-size", self.queen_cell_size, "--queen-radius", self.queen_radius, "--image-format", "png",
                ])
                conv_cmds.append([
                    sys.executable, self.queen_gen / "convert_queen_to_parquet.py",
                    "--queen-outdir", raw_i, "--dataset-outdir", pq_i, "--test-ratio", "0", "--seed", "42",
                ])
                parts.append(pq_i / "maze_dataset_train.parquet")

            self._run_parallel(gen_cmds)
            self._run_parallel(conv_cmds)
            for part in parts:
                if not part.exists():
                    raise RuntimeError("convert_queen_to_parquet.py did not produce maze_dataset_train.parquet")

            produced = work / "maze_dataset_train.parquet"
            if len(parts) == 1:
                shutil.move(str(parts[0]), str(produced))
            else:
                self._merge_parquets(parts, produced)

            # Ids must be unique: the scorer looks up ground truth BY id.
            self._reindex_queen_ids(produced, scale)
            return produced, work

        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise

    def _reindex_queen_ids(self, path: Path, scale: int) -> None:
        """Give every queens puzzle in a merged pool a unique id.
        Because of multiple workers, those ids are duplicated.
        """
        df = pd.read_parquet(path)
        df["id"] = [f"level_{scale}_{i:04d}" for i in range(len(df))]
        df.to_parquet(path, index=False, compression="snappy")
        print(f">> re-indexed {len(df)} queens ids in {path.name}", flush=True)

    def _get_data_paths(self):
        queens_dir = self.out_root / "queens"
        queen_train_file = queens_dir / "maze_dataset_train.parquet"
        queen_test_file = queens_dir / "maze_dataset_test.parquet"
        return {
            "train": queen_train_file,
            "test": queen_test_file,
            "scale_tests": {scale: queens_dir / f"n{scale}_test.parquet" for scale in self.scales},
        }



class MazeDatasetGenerator(AmazeDatasetGenerator):
    def __init__(self, size, shape):
        super().__init__(size, shape)
        self.maze_geometries = ["square", "hexagon", "triangle", "circle"]
        self.maze_scales = [int(x) for x in os.environ.get("MAZE_SCALES", "5,7,8,9,11,13,16").split(",") if x.strip()]
        self.maze_ood_scales = [int(x) for x in os.environ.get("MAZE_OOD_SCALES", "10").split(",") if x.strip()]
        self.maze_test_per_scale = int(os.environ.get("MAZE_TEST_PER_SCALE", "100"))
        self.train_image_size = int(os.environ.get("TRAIN_IMAGE_SIZE", "144"))
        self._universal = ["recursiveBacktrack", "simplifiedPrims", "truePrims", "wilson", "aldousBroder", "huntAndKill"]
        self.train_algos = {
            "square": self._universal + ["kruskal", "binaryTree", "sidewinder", "ellers"],
            "hexagon": self._universal,
            "triangle": self._universal,
            "circle": self._universal,
        }
        self.test_algorithm = "recursiveBacktrack"
        self.maze_train = int(os.environ.get("MAZE_TRAIN", "30000"))
        self.maze_nproc = self._nproc("MAZE_NPROC")

        self.train_seed = 1_000_000
        self.test_seed = 7_000_000

    def generate(self):
        data_paths = self._get_data_paths()
        data_paths["train"].parent.mkdir(parents=True, exist_ok=True)

    def _maze_train_entries(shapes: list[str], scales: list[int]) -> list[dict]:
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

    def gen_maze_train(self, shape: str, size) -> None:
        target_dir = self.train_maze_dir(shape, size)
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

    def _get_data_paths(self):
        maze_dir = self.out_root / "maze"
        queen_train_file = maze_dir / "maze_dataset_train.parquet"
        queen_test_file = maze_dir / "maze_dataset_test.parquet"

        base_dir = {"train": queen_train_file, "test": queen_test_file}
        for shape in self.maze_geometries:
            shape_dir = {"all": maze_dir / f"all_{shape}_test.parquet"}

            for scale in self.maze_scales:
                shape_dir[str(scale)] = maze_dir / f"n{scale}_test.parquet"

            base_dir[str(shape)] = shape_dir    # type: ignore

        return base_dir

    # ---- CHECK EXISTANCE AND GENERATE (if doesnt exist) ---
    def _ensure_maze_test_all(self) -> Path:
        shape_all = [self._ensure_maze_test_shape_all(shape) for shape in MAZE_GEOMETRIES]
        all_mazes_all_shapes_parquet = self._test_all_file()   # the path from helpers
        if not all_mazes_all_shapes_parquet.exists():         # merge all per shape/size files
            _merge_parquets(shape_all, all_mazes_all_shapes_parquet)
        return all_mazes_all_shapes_parquet


    def _ensure_maze_test_shape_all(self, shape: str) -> Path:
        for scale in MAZE_SCALES + MAZE_OOD_SCALES:
            self._ensure_maze_test_combo(shape, scale)
        all_mazes_per_shape_parquet = self._test_shape_all_file(shape)
        if not all_mazes_per_shape_parquet.exists():
            all_combos = [self._test_combo_file(shape, s) for s in MAZE_SCALES]
            _merge_parquets(all_combos, all_mazes_per_shape_parquet)
        return all_mazes_per_shape_parquet


    def _ensure_maze_test_combo(self, shape: str, scale: int) -> None:
        target = self._test_combo_file(shape, scale)
        if target.exists():
            print(f"{target.parent.name}/{target.name} already exists — skip")
            return
        entries = _maze_entries(
            shape, scale, MAZE_TEST_PER_SCALE, [TEST_ALGORITHM],
            seed_base=MAZE_TEST_SEED + scale * 10_000,
        )
        _gen_maze_split(entries, target.parent, target.name, train_ratio=0.0)

    def _ensure_node_deps() -> None:
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
        if (MAZE_GEN / "node_modules").is_dir():
            return
        print(f"Installing node deps in {MAZE_GEN}", flush=True)
        _run(["npm", "install"], cwd=MAZE_GEN)

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
                entry["layers"] = scale
            else:
                entry["width"] = scale
                entry["height"] = scale
            entries.append(entry)
        return entries

    def _gen_maze_split(entries: list[dict], out_dir: Path, out_name: str, train_ratio: float) -> None:
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

            _run_parallel(node_cmds)
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

    def _maze_train_entries(shapes: list[str], scales: list[int]) -> list[dict]:
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


    def gen_maze_train(shape: str, size) -> None:
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



def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=True, description="Amaze dataset generator")
    parser.add_argument("task", help="maze | queens")
    parser.add_argument("--shape", default=None, type=str)
    parser.add_argument("--size", default=None, help="train: all | <int>")
    args = parser.parse_args(argv)
    task = args.task

    if task not in ("maze", "queens"):
        raise SystemExit(f"Unknown task '{args.task}' (use maze|queens)")

    generator = dataset_generator_factory(task, args.size, args.shape)
    generator.generate()

    # Calculate sizes of the parquet files
    # @TODO

    # Test every test dataset on ground truth using AmazeMetrics
    # @TODO


if __name__ == "__main__":
    main(sys.argv[1:])



    #    self.test_per_scale_num_samples = int(os.environ.get("QUEEN_TEST_PER_SCALE", "50"))
    #     self.

    #     self.queen_train_scale_caps = {
    #         int(k): int(v)
    #         for tok in os.environ.get("QUEEN_TRAIN_SCALE_CAPS", "4:60,5:3040").split(",")
    #         if tok.strip()
    #         for k, v in [tok.split(":")]
    #     }