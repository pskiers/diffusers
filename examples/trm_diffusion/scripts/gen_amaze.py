"""
Generate the AMAZE datasets (mazes with the maze generator, queens boards with the
queen generator) and post-process them into the parquet layout the trainers expect.

Everything is driven by configs/data/amaze_generation.yaml — which puzzle sizes to
build, how many samples per size, where to write them, which columns to keep.

    python scripts/gen_amaze.py --config configs/data/amaze_generation.yaml
    python scripts/gen_amaze.py --task maze --stage test
    python scripts/gen_amaze.py -o output_root=/scratch/amaze -o overwrite=true

Data structure:
|- queens/
    |--- maze_dataset_train_144.parquet         # scaled, slim columns (one per scaled_image_sizes)
    |--- maze_dataset_train_original.parquet    # native resolution, slim columns
    |--- maze_dataset_test.parquet              # native resolution, ALL columns
    |--- n{size}_test.parquet                   # for each size n
|- maze/
    |--- maze_dataset_train_144.parquet
    |--- maze_dataset_train_original.parquet
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

Out-of-distribution sizes are written as n{size}_test.parquet only; they are never
merged into all_test.parquet / maze_dataset_test.parquet.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# Cap native thread pools before importing pandas/pyarrow.
for _thr_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS", "RAYON_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thr_var, "1")

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402


TRM_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = TRM_ROOT / "configs" / "data" / "amaze_generation.yaml"

TASKS = ("maze", "queens")

IMAGE_COLUMNS = ("original_img", "m_original_img", "sol_img", "mask_img", "cell_map")
SCALED_COLUMNS = ("m_original_img", "sol_img")


class AmazeDatasetGenerator:
    """Shared plumbing: parallel subprocesses, parquet merge/slim/resize, checks."""

    def __init__(self, cfg: DictConfig, task: str):
        if task not in TASKS:
            raise SystemExit(f"Unknown task '{task}' (use {'|'.join(TASKS)})")
        self.cfg = cfg
        self.task = task
        self.task_cfg = cfg[task]

        self.maze_gen = TRM_ROOT / "third_party" / "amaze" / "mazes-generator"
        self.queen_gen = TRM_ROOT / "third_party" / "amaze" / "queen-generator"

        root = Path(str(cfg.output_root))
        self.out_root = root if root.is_absolute() else TRM_ROOT / root
        self.out_dir = self.out_root / task

        self.overwrite = bool(cfg.overwrite)
        self.chunk_rows = int(cfg.chunk_rows)
        self.train_columns = [str(c) for c in cfg.train_columns]
        self.scaled_sizes = [int(s) for s in cfg.scaled_image_sizes]
        self.nproc = max(1, int(cfg.nproc) if cfg.nproc else
                         int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1)))
        self.verify_results: list[tuple[str, int, float, float]] = []

    # IMAGE CELLS
    @staticmethod
    def decode_image(raw) -> Image.Image | None:
        """Parquet image cell -> RGB PIL image (None for a missing cell)."""
        if raw is None:
            return None
        if isinstance(raw, float):          # pandas NaN
            return None
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
        if isinstance(raw, str):
            payload = raw.split(",", 1)[1] if raw.startswith("data:") else raw
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        raise TypeError(f"Unsupported image cell type: {type(raw)}")

    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """RGB PIL image -> base64 PNG string (the encoding every loader expects)."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @classmethod
    def as_base64(cls, raw) -> str | None:
        """Normalise a cell to base64 PNG WITHOUT re-encoding one that already is.

        Both vendored converters store base64 strings, so this is a pass-through for
        every parquet this script reads; it only does work for a pool some other
        producer left as raw PNG bytes.
        """
        if raw is None or isinstance(raw, float):
            return None
        if isinstance(raw, str):
            return raw
        return cls.encode_image(cls.decode_image(raw))   # type: ignore[arg-type]

    @staticmethod
    def image_dimensions(raw) -> tuple[int, int] | None:
        """(width, height) of an image cell, read from the PNG header alone.

        No pixel decode and no full base64 pass — 64 characters carry the IHDR — so
        measuring a whole dataset costs parquet I/O rather than CPU.
        """
        if raw is None or isinstance(raw, float):
            return None
        if isinstance(raw, str):
            payload = raw.split(",", 1)[1] if raw.startswith("data:") else raw
            head = base64.b64decode(payload[:64], validate=False)
        elif isinstance(raw, (bytes, bytearray)):
            head = bytes(raw[:48])
        else:
            head = b""
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
        image = AmazeDatasetGenerator.decode_image(raw)     # not a PNG — fall back
        return image.size if image is not None else None

    def measure_dimensions(self, path: Path, max_row_groups: int = 16,
                           per_group: int = 24) -> dict[str, list[str]]:
        """Distinct WxH per image column, sampled across the file's row groups.

        Sampling is spread over the whole file, so a mixed pool (many shapes and
        sizes concatenated) reports every resolution it holds without being read end
        to end.
        """
        parquet = pq.ParquetFile(path)
        columns = [c for c in IMAGE_COLUMNS if c in parquet.schema_arrow.names]
        if not columns or parquet.metadata.num_rows == 0:
            return {}
        groups = parquet.num_row_groups
        take = min(max_row_groups, groups)
        picks = sorted({round(i * (groups - 1) / max(1, take - 1)) for i in range(take)})

        found: dict[str, set[str]] = {column: set() for column in columns}
        for group in picks:
            table = parquet.read_row_group(group, columns=columns)
            # Spread the sample inside the group too: a small file is one single row
            # group holding every shape and size, so its first cells are all alike.
            rows = table.num_rows
            wanted = min(per_group, rows)
            offsets = sorted({round(i * (rows - 1) / max(1, wanted - 1)) for i in range(wanted)})
            for column in columns:
                cells = table.column(column)
                for offset in offsets:
                    size = self.image_dimensions(cells[offset].as_py())
                    if size is not None:
                        found[column].add(f"{size[0]}x{size[1]}")
        return {column: sorted(sizes, key=lambda wh: int(wh.split("x")[0]))
                for column, sizes in found.items() if sizes}

    @classmethod
    def build_metadata(cls, row) -> dict:
        """Parquet row -> the metadata dict AmazeDataset hands to AmazeMetrics."""
        return {
            "id": row.get("id"),
            "metadata": row.get("metadata"),
            "sample_json": row.get("sample_json"),
            "original_img": cls.decode_image(row.get("original_img")),
            "m_original_img": cls.decode_image(row.get("m_original_img")),
            "sol_img": cls.decode_image(row.get("sol_img")),
            "mask_img": cls.decode_image(row.get("mask_img")),
            "cell_map": cls.decode_image(row.get("cell_map")),
        }

    # PATHS
    def train_path(self, tag: str | int) -> Path:
        return self.out_dir / f"maze_dataset_train_{tag}.parquet"

    def test_path(self) -> Path:
        return self.out_dir / "maze_dataset_test.parquet"

    def _pool_path(self) -> Path:
        """Native-resolution, all-columns train pool. Hidden, and deleted on success —
        it only survives a crash so a re-run can resume without regenerating."""
        return self.out_dir / ".train_pool.parquet"

    def test_leaf_paths(self) -> list[Path]:
        raise NotImplementedError

    def merged_test_parts(self) -> dict[Path, list[Path]]:
        """merged parquet -> the per-size parquets it is a concatenation of."""
        raise NotImplementedError

    # SIZES
    def _test_sizes(self) -> dict[int, int]:
        """In-distribution and OOD test sizes together (both get an n{size} file).

        A size asked for 0 rows is dropped, not built empty: 0 is how the config
        says "skip this size", and the vendored converter fails on an empty batch.
        """
        sizes = {int(k): int(v) for k, v in (self.task_cfg.test.in_distribution or {}).items()}
        sizes.update({int(k): int(v) for k, v in (self.task_cfg.test.get("ood") or {}).items()})
        return {size: count for size, count in sorted(sizes.items()) if count > 0}

    def _merged_test_sizes(self) -> list[int]:
        return sorted(int(k) for k, v in (self.task_cfg.test.in_distribution or {}).items()
                      if int(v) > 0)

    def _train_sizes(self) -> dict[int, int]:
        sizes = {int(k): int(v) for k, v in (self.task_cfg.train.samples or {}).items()}
        return {size: count for size, count in sorted(sizes.items()) if count > 0}

    # STAGES
    def generate(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.generate_test()
        self.generate_train()

    def generate_test(self) -> None:
        raise NotImplementedError

    def generate_train(self) -> None:
        """Build every requested train parquet, generating as little as possible.

        A new entry in scaled_image_sizes is derived from the existing
        maze_dataset_train_original.parquet — the generators only run when that
        file is missing too.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        wanted: dict[str, int | None] = {"original": None}
        wanted.update({str(size): size for size in self.scaled_sizes})

        pending = {tag: size for tag, size in wanted.items()
                   if not self._skip_existing(self.train_path(tag))}
        if not pending:
            return

        original = self.train_path("original")
        pool = self._pool_path()
        if "original" not in pending and original.is_file():
            print(f">> deriving {sorted(pending)} from {original.name}", flush=True)
            self._write_train_outputs(original, pending)
            return

        if pool.is_file() and not self.overwrite:
            print(f">> reusing train pool {pool.name} from an earlier run", flush=True)
        else:
            pool.unlink(missing_ok=True)
            self._generate_train_pool(pool)
        self._write_train_outputs(pool, pending)
        pool.unlink(missing_ok=True)

    def _generate_train_pool(self, dest: Path) -> None:
        raise NotImplementedError

    @staticmethod
    def score_against_ground_truth(task: str, path: str,
                                   chunk_rows: int) -> tuple[str, int, float, float]:
        """Score every row of a test parquet against its OWN sol_img.

        Runs in a worker process, hence a staticmethod: none of the generator has to
        travel with it. A correct scorer must return pass=1.0 for the ground truth
        itself, so a mean pass below 1.0 means the parquet and the scorer disagree —
        a stale file, or a metric that is too strict.
        """
        import numpy as np
        import torch

        if str(TRM_ROOT) not in sys.path:
            sys.path.insert(0, str(TRM_ROOT))
        from eval.amaze_eval import AmazeMetrics

        scorer = AmazeMetrics(task=task)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_rows):
            for _, row in batch.to_pandas().iterrows():
                metadata = AmazeDatasetGenerator.build_metadata(row)
                solution = metadata["sol_img"]
                if solution is None:
                    raise RuntimeError(f"{path}: row {row.get('id')} has no sol_img")
                arr = np.asarray(solution, dtype="float32") / 255.0
                generated = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
                scorer.compute_and_accumulate_metrics(generated, [metadata])

        metrics = scorer.return_metrics()
        return (path,
                int(metrics["generated_samples"]),
                float(metrics["mean_pass"]),
                float(metrics.get("mean_gt_cell_coverage", 0.0)))

    def verify(self) -> list[tuple[str, int, float, float]]:
        """GT-vs-GT check on every per-size test parquet.

        Only the leaves are scored: all_test / maze_dataset_test are exact
        concatenations of them, so re-scoring those would triple the work. Their
        row counts are checked structurally in report().
        """
        paths = [p for p in self.test_leaf_paths() if p.is_file()]
        if not paths:
            print(f"!! {self.task}: no test parquets to verify")
            return []
        self.verify_results = []

        workers = max(1, min(self.nproc, 8, len(paths)))   # each worker imports torch
        print(f">> verifying {len(paths)} {self.task} test parquet(s) "
              f"on {workers} worker(s) — scoring every row", flush=True)

        results: list[tuple[str, int, float, float]] = []
        if workers == 1:
            for path in paths:
                results.append(self.score_against_ground_truth(
                    self.task, str(path), self.chunk_rows))
                self._print_verify(results[-1])
            self.verify_results = results
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self.score_against_ground_truth,
                                       self.task, str(p), self.chunk_rows)
                           for p in paths]
                for future in futures:
                    results.append(future.result())
                    self._print_verify(results[-1])
        self.verify_results = results
        return results

    def _print_verify(self, result: tuple[str, int, float, float]) -> None:
        path, rows, mean_pass, coverage = result
        min_pass = float(self.cfg.verify.min_pass)
        status = "ok  " if mean_pass >= min_pass else "FAIL"
        print(f"   [{status}] {self._rel(Path(path))}  "
              f"rows={rows}  GT pass={mean_pass:.3f}  coverage={coverage:.3f}", flush=True)

    REPORT_NAME = "dataset_report.json"

    def report(self) -> None:
        """Print, and write to <out_dir>/dataset_report.json, what was produced:
        rows, on-disk size and the measured pixel dimensions of every image column,
        plus the GT pass metrics when verify ran in the same invocation."""
        print(f"\n=== {self.task} ===")
        produced = [self.train_path("original")]
        produced += [self.train_path(size) for size in self.scaled_sizes]
        produced += [self.test_path()]
        produced += sorted(self.merged_test_parts())
        produced += self.test_leaf_paths()

        files: list[dict] = []
        seen: set[Path] = set()
        for path in produced:
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file():
                print(f"   {'MISSING':>10}  {self._rel(path)}")
                files.append({"path": self._rel(path), "present": False})
                continue
            rows = pq.ParquetFile(path).metadata.num_rows
            megabytes = path.stat().st_size / 2**20
            dimensions = self.measure_dimensions(path)
            distinct = sorted({wh for sizes in dimensions.values() for wh in sizes},
                              key=lambda wh: int(wh.split("x")[0]))
            label = (", ".join(distinct) if len(distinct) <= 2
                     else f"{distinct[0]}..{distinct[-1]} ({len(distinct)} sizes)")
            print(f"   {rows:>8,} rows  {megabytes:>8.1f} MB  {label:>24}  {self._rel(path)}")
            files.append({
                "path": self._rel(path),
                "present": True,
                "rows": rows,
                "size_mb": round(megabytes, 3),
                "image_size": dimensions,
            })

        stale: list[dict] = []
        for merged, parts in self.merged_test_parts().items():
            if not merged.is_file():
                continue
            expected = sum(pq.ParquetFile(p).metadata.num_rows for p in parts if p.is_file())
            actual = pq.ParquetFile(merged).metadata.num_rows
            if actual != expected:
                print(f"   !! {merged.name}: {actual} rows but its parts hold {expected} "
                      f"— stale merge, delete it and re-run")
                stale.append({"path": self._rel(merged), "rows": actual,
                              "rows_in_parts": expected})

        document = {
            "task": self.task,
            "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "output_root": str(self.out_root),
            "scaled_image_sizes": self.scaled_sizes,
            "train_columns": self.train_columns,
            "image_size_note": ("distinct WxH seen in a bounded sample of rows per file, "
                                "not a full scan — a leaf n<size> file is uniform, a "
                                "merged or train file is not"),
            "files": files,
            "stale_merges": stale,
            "verify": [
                {"path": self._rel(Path(path)), "rows": rows,
                 "mean_pass": round(mean_pass, 4),
                 "mean_gt_cell_coverage": round(coverage, 4),
                 "min_pass": float(self.cfg.verify.min_pass),
                 "ok": mean_pass >= float(self.cfg.verify.min_pass)}
                for path, rows, mean_pass, coverage in self.verify_results
            ],
        }
        destination = self.out_dir / self.REPORT_NAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2) + "\n")
        note = "" if self.verify_results else "  (no verify in this run — pass metrics empty)"
        print(f"   -> {self._rel(destination)}{note}")

    def link_ft(self) -> None:
        """Point the BAGEL fine-tune layout at the native-resolution parquets."""
        train_src, test_src = self.train_path("original"), self.test_path()
        if not (train_src.is_file() and test_src.is_file()):
            print(f">> ft links for {self.task} skipped — train/test parquet missing")
            return
        ft_dir = self.out_root / "ft" / self.task
        ft_dir.mkdir(parents=True, exist_ok=True)
        for name, src in (("maze_dataset_train.parquet", train_src),
                          ("maze_dataset_test.parquet", test_src)):
            link = ft_dir / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(src, ft_dir))
            print(f">> ft link {self._rel(link)} -> {src.name}")

    # ---- parquet helpers

    def _rel(self, path: Path) -> str:
        """Path relative to the output root for logging; absolute for temp work dirs."""
        try:
            return str(path.relative_to(self.out_root))
        except ValueError:
            return str(path)

    def _skip_existing(self, path: Path) -> bool:
        """True if `path` is already there and may be kept; clears it when overwriting."""
        if not path.exists():
            return False
        if self.overwrite:
            path.unlink()
            return False
        print(f">> {self._rel(path)} already exists — skip")
        return True

    def _merge_parquets(self, parts: list[Path], out_path: Path) -> None:
        """Concatenate parquets by streaming row groups, so a 30k-row native-resolution
        pool never has to fit in RAM."""
        parts = [p for p in parts if p.exists()]
        if not parts:
            raise RuntimeError(f"No parquet parts to merge into {out_path}")

        fields: dict[str, pa.Field] = {}
        for part in parts:
            for field in pq.ParquetFile(part).schema_arrow:
                fields.setdefault(field.name, field)
        schema = pa.schema(list(fields.values()))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        rows = 0
        writer = pq.ParquetWriter(tmp, schema, compression="snappy")
        try:
            for part in parts:
                for batch in pq.ParquetFile(part).iter_batches(batch_size=self.chunk_rows):
                    frame = self._conform(batch.to_pandas(), schema.names)
                    writer.write_table(pa.Table.from_pandas(frame, schema=schema,
                                                            preserve_index=False))
                    rows += len(frame)
        finally:
            writer.close()
        os.replace(tmp, out_path)
        print(f">> merged {len(parts)} parts ({rows} rows) → "
              f"{self._rel(out_path)}", flush=True)

    @staticmethod
    def _conform(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.Series([None] * len(frame), dtype=object)
        return frame[columns]

    def _write_train_outputs(self, source: Path, pending: dict[str, int | None]) -> None:
        """One streaming pass over `source`, writing the original and every scaled
        train parquet at once: keep `train_columns`, resize the two model-facing
        columns for each scaled size.
        """
        from torchvision import transforms   # only needed here; keeps generation torch-free

        available = pq.ParquetFile(source).schema_arrow.names
        keep = [c for c in self.train_columns if c in available]
        dropped = [c for c in available if c not in keep]
        missing = [c for c in self.train_columns if c not in available]
        if missing:
            print(f"!! {self.task}: train_columns {missing} absent from {source.name}")
        if not keep:
            raise RuntimeError(f"{source}: none of train_columns {self.train_columns} present")

        resizers = {
            tag: transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            for tag, size in pending.items() if size is not None
        }
        scaled = [c for c in SCALED_COLUMNS if c in keep]
        image_columns = [c for c in IMAGE_COLUMNS if c in keep]
        tmps = {tag: self.train_path(tag).with_name(self.train_path(tag).name + ".tmp")
                for tag in pending}
        writers: dict[str, pq.ParquetWriter] = {}
        schema: pa.Schema | None = None
        rows = 0

        print(f">> writing {sorted(pending)} from {source.name}: keep {keep}, "
              f"drop {dropped or '[]'}", flush=True)
        try:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=self.chunk_rows,
                                                             columns=keep):
                frame = batch.to_pandas()
                for column in image_columns:
                    frame[column] = [self.as_base64(v) for v in frame[column]]
                # Decoding costs a full PNG decompress per image, so only pay it for
                # the columns a resize actually needs. With no scaled size pending,
                # the unscaled copy is a straight pass-through of the source bytes.
                pixels = ({c: [self.decode_image(v) for v in frame[c]] for c in scaled}
                          if resizers else {})

                for tag in pending:
                    out = frame
                    if tag in resizers:
                        out = frame.copy()
                        for column in scaled:
                            out[column] = [None if im is None
                                           else self.encode_image(resizers[tag](im))
                                           for im in pixels[column]]
                    table = pa.Table.from_pandas(out, preserve_index=False)
                    if schema is None:
                        schema = table.schema
                    elif table.schema != schema:
                        table = table.cast(schema)
                    if tag not in writers:
                        writers[tag] = pq.ParquetWriter(tmps[tag], schema, compression="snappy")
                    writers[tag].write_table(table)
                rows += len(frame)
        finally:
            for writer in writers.values():
                writer.close()

        for tag in pending:
            os.replace(tmps[tag], self.train_path(tag))
            size = pending[tag]
            note = "native resolution" if size is None else f"{size}x{size}"
            print(f">> train {self.train_path(tag).name}: {rows} rows, {note}", flush=True)

    # ---- subprocess helpers

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
            out, err = proc.communicate()
            if proc.returncode != 0:
                rc = proc.returncode
                note = ""
                if rc < 0:
                    try:
                        note = f" [killed by {signal.Signals(-rc).name}]"
                    except Exception:
                        note = f" [killed by signal {-rc}]"
                # The vendored converters report their failures on stdout, so both
                # streams have to be shown or the error says nothing.
                errors.append(f"Command failed ({rc}){note}: {' '.join(argv)}\n"
                              f"--- stdout (last 40 lines) ---\n"
                              f"{chr(10).join(out.splitlines()[-40:])}\n"
                              f"--- stderr ---\n{err}")
        if errors:
            raise RuntimeError("\n\n".join(errors))


class QueensDatasetGenerator(AmazeDatasetGenerator):
    """Queens boards, one parquet per board size plus a mixed-size train pool."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, "queens")
        self.cell_size = str(self.task_cfg.renderer.cell_size)
        self.queen_radius = str(self.task_cfg.renderer.queen_radius)
        self.train_seed = int(self.task_cfg.seeds.train)
        self.test_seed = int(self.task_cfg.seeds.test)

    def test_leaf_path(self, size: int) -> Path:
        return self.out_dir / f"n{size}_test.parquet"

    def test_leaf_paths(self) -> list[Path]:
        return [self.test_leaf_path(size) for size in self._test_sizes()]

    def merged_test_parts(self) -> dict[Path, list[Path]]:
        return {self.test_path(): [self.test_leaf_path(s) for s in self._merged_test_sizes()]}

    def generate_test(self) -> None:
        for size, count in self._test_sizes().items():
            target = self.test_leaf_path(size)
            if self._skip_existing(target):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Same seed base across sizes is safe: n is an input to generation, so the
            # same RNG stream at a different n yields a different board.
            produced, work = self._gen_pool(size, count, self.test_seed + size)
            try:
                shutil.move(str(produced), str(target))
            finally:
                shutil.rmtree(work, ignore_errors=True)

        if not self._skip_existing(self.test_path()):
            self._merge_parquets([self.test_leaf_path(s) for s in self._merged_test_sizes()],
                                 self.test_path())

    def _generate_train_pool(self, dest: Path) -> None:
        per_size = self._train_sizes()
        print(f">> generating {sum(per_size.values())} train queens boards: {per_size}",
              flush=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="amaze_queens_pool_"))
        try:
            parts = []
            for size, count in per_size.items():
                if count <= 0:
                    continue
                produced, work = self._gen_pool(size, count, self.train_seed + size * 100_000)
                part = temp_dir / f"pool_n{size}.parquet"
                shutil.move(str(produced), str(part))
                shutil.rmtree(work, ignore_errors=True)
                parts.append(part)
            self._merge_parquets(parts, dest)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _gen_pool(self, size: int, count: int, seed: int) -> tuple[Path, Path]:
        if count <= 0:
            raise SystemExit(f"queens n={size}: asked for {count} boards")
        """Generate `count` boards of side `size` with the vendored scripts and convert
        them to one parquet. Returns (parquet, work dir) — the caller owns both."""
        work = Path(tempfile.mkdtemp(prefix="amaze_queens_"))
        try:
            chunks = self._split_count(count, self.nproc)
            print(f">> queens n={size}: {count} boards across {len(chunks)} worker(s)",
                  flush=True)

            gen_cmds, conv_cmds, parts = [], [], []
            for i, chunk in enumerate(chunks):
                raw_dir, pq_dir = work / f"raw_{i}", work / f"pq_{i}"
                raw_dir.mkdir()
                pq_dir.mkdir()
                gen_cmds.append([
                    sys.executable, self.queen_gen / "generate_queens_puzzle.py",
                    "--n", size, "--count", chunk, "--outdir", raw_dir, "--seed", seed + i,
                    "--cell-size", self.cell_size, "--queen-radius", self.queen_radius,
                    "--image-format", "png",
                ])
                conv_cmds.append([
                    sys.executable, self.queen_gen / "convert_queen_to_parquet.py",
                    "--queen-outdir", raw_dir, "--dataset-outdir", pq_dir,
                    "--test-ratio", "0", "--seed", "42",
                ])
                parts.append(pq_dir / "maze_dataset_train.parquet")

            self._run_parallel(gen_cmds)
            self._run_parallel(conv_cmds)
            for part in parts:
                if not part.exists():
                    raise RuntimeError("convert_queen_to_parquet.py did not produce "
                                       f"{part.name}")

            produced = work / "maze_dataset_train.parquet"
            if len(parts) == 1:
                shutil.move(str(parts[0]), str(produced))
            else:
                self._merge_parquets(parts, produced)

            # Workers each restart the level_{n}_{i} numbering, so ids repeat. The
            # scorer looks GT up BY id, so make them unique before anything merges.
            self._reindex_ids(produced, size)
            return produced, work
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise

    def _reindex_ids(self, path: Path, size: int) -> None:
        tmp = path.with_name(path.name + ".reindex")
        schema, writer, index = None, None, 0
        try:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=self.chunk_rows):
                frame = batch.to_pandas()
                frame["id"] = [f"level_{size}_{i:06d}" for i in range(index, index + len(frame))]
                index += len(frame)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if schema is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(tmp, schema, compression="snappy")
                else:
                    table = table.cast(schema)
                writer.write_table(table)   # type: ignore[union-attr]
        finally:
            if writer is not None:
                writer.close()
        os.replace(tmp, path)
        print(f">> re-indexed {index} queens ids in {path.name}", flush=True)


class MazeDatasetGenerator(AmazeDatasetGenerator):
    """Mazes for every shape x size, plus per-shape and global merged test sets."""

    # Algorithms are fixed properties of the vendored generator, not a knob: these
    # are the ones every shape supports, plus the four that only square grids do.
    UNIVERSAL_ALGORITHMS = ["recursiveBacktrack", "simplifiedPrims", "truePrims",
                            "wilson", "aldousBroder", "huntAndKill"]
    TRAIN_ALGORITHMS = {
        "square": UNIVERSAL_ALGORITHMS + ["kruskal", "binaryTree", "sidewinder", "ellers"],
        "hexagon": UNIVERSAL_ALGORITHMS,
        "triangle": UNIVERSAL_ALGORITHMS,
        "circle": UNIVERSAL_ALGORITHMS,
    }
    TEST_ALGORITHM = "recursiveBacktrack"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, "maze")
        self.shapes = [str(s) for s in self.task_cfg.shapes]
        unknown = [s for s in self.shapes if s not in self.TRAIN_ALGORITHMS]
        if unknown:
            raise SystemExit(f"Unknown maze shape(s) {unknown} "
                             f"(use {'|'.join(self.TRAIN_ALGORITHMS)})")
        self.exit_config = str(self.task_cfg.exit_config)
        self.train_seed = int(self.task_cfg.seeds.train)
        self.test_seed = int(self.task_cfg.seeds.test)

    def shape_dir(self, shape: str) -> Path:
        return self.out_dir / shape

    def test_leaf_path(self, shape: str, size: int) -> Path:
        return self.shape_dir(shape) / f"n{size}_test.parquet"

    def shape_test_path(self, shape: str) -> Path:
        return self.shape_dir(shape) / "all_test.parquet"

    def test_leaf_paths(self) -> list[Path]:
        return [self.test_leaf_path(shape, size)
                for shape in self.shapes for size in self._test_sizes()]

    def merged_test_parts(self) -> dict[Path, list[Path]]:
        merged = {self.shape_test_path(shape): [self.test_leaf_path(shape, size)
                                                for size in self._merged_test_sizes()]
                  for shape in self.shapes}
        merged[self.test_path()] = [self.shape_test_path(shape) for shape in self.shapes]
        return merged

    def generate_test(self) -> None:
        for shape in self.shapes:
            for size, count in self._test_sizes().items():
                target = self.test_leaf_path(shape, size)
                if self._skip_existing(target):
                    continue
                entries = self._entries(shape, size, count, [self.TEST_ALGORITHM],
                                        self.test_seed + size * 10_000)
                self._gen_split(entries, target, train_ratio=0.0)

            shape_all = self.shape_test_path(shape)
            if not self._skip_existing(shape_all):
                self._merge_parquets([self.test_leaf_path(shape, size)
                                      for size in self._merged_test_sizes()], shape_all)

        if not self._skip_existing(self.test_path()):
            self._merge_parquets([self.shape_test_path(s) for s in self.shapes],
                                 self.test_path())

    def _generate_train_pool(self, dest: Path) -> None:
        per_size = self._train_sizes()
        sizes = list(per_size)
        entries: list[dict] = []
        for shape_index, shape in enumerate(self.shapes):
            for size_index, size in enumerate(sizes):
                count = per_size[size]
                if count <= 0:
                    continue
                seed_base = self.train_seed + (shape_index * len(sizes) + size_index) * 100_000
                entries += self._entries(shape, size, count,
                                         self.TRAIN_ALGORITHMS[shape], seed_base)
        print(f">> generating {len(entries)} train mazes: {len(self.shapes)} shape(s) "
              f"x {len(sizes)} size(s), {per_size} per shape", flush=True)
        self._gen_split(entries, dest, train_ratio=1.0)

    def _entries(self, shape: str, size: int, count: int,
                 algorithms: list[str], seed_base: int) -> list[dict]:
        entries = []
        for i in range(count):
            entry = {
                "shape": shape,
                "algorithm": algorithms[i % len(algorithms)],
                "exitConfig": self.exit_config,
                "seed": seed_base + i,
                "filename": f"{shape}_{size}_{i:06d}.png",
            }
            if shape == "circle":
                entry["layers"] = size
            else:
                entry["width"] = size
                entry["height"] = size
            entries.append(entry)
        return entries

    def _gen_split(self, entries: list[dict], target: Path, train_ratio: float) -> None:
        if not entries:
            raise SystemExit(f"No mazes to generate for {self._rel(target)} — every size in "
                             f"the maze config asks for 0 rows")
        self._ensure_node_deps()
        target.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="amaze_maze_"))
        try:
            chunks = self._split_list(entries, self.nproc)
            want = ("maze_dataset_train.parquet" if train_ratio >= 1.0
                    else "maze_dataset_test.parquet")
            print(f">> maze: {len(entries)} mazes across {len(chunks)} worker(s) "
                  f"→ {target.name}", flush=True)

            node_cmds, proc_cmds, parts = [], [], []
            for i, chunk in enumerate(chunks):
                worker = work / f"w{i}"
                worker.mkdir()
                (worker / "cfg.json").write_text(json.dumps({"mazes": chunk}))
                node_cmds.append((["node", self.maze_gen / "batch-maze-generator.js",
                                   "config", worker / "cfg.json"], worker))
                proc_cmds.append([
                    sys.executable, self.maze_gen / "process_maze_into_parquet.py",
                    "--maze-dir", worker / "generated_mazes",
                    "--no-markers-dir", worker / "generated_mazes_no_markers",
                    "--solution-dir", worker / "generated_solutions",
                    "--metadata-dir", worker / "generated_metadata",
                    "--output", worker / "maze_dataset.parquet",
                    "--train-ratio", str(train_ratio),
                    "--seed", "42",
                ])
                parts.append(worker / want)

            self._run_parallel(node_cmds)
            self._run_parallel(proc_cmds)
            for part in parts:
                if not part.exists():
                    raise RuntimeError(f"process_maze_into_parquet.py did not produce "
                                       f"{part.name}")

            if len(parts) == 1:
                shutil.move(str(parts[0]), str(target))
            else:
                self._merge_parquets(parts, target)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _ensure_node_deps(self) -> None:
        if (self.maze_gen / "node_modules").is_dir():
            return
        print(f"Installing node deps in {self.maze_gen} (needs internet — "
              f"run on a login node)…", flush=True)
        res = subprocess.run(["npm", "install"], cwd=self.maze_gen,
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"npm install failed ({res.returncode})\n"
                               f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")


def build_generator(cfg: DictConfig, task: str) -> AmazeDatasetGenerator:
    if task == "maze":
        return MazeDatasetGenerator(cfg)
    if task == "queens":
        return QueensDatasetGenerator(cfg)
    raise SystemExit(f"Unknown task '{task}' (use {'|'.join(TASKS)})")


def load_config(path: Path, overrides: list[str]) -> DictConfig:
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg   # type: ignore[return-value]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Amaze dataset generator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"generation settings yaml (default: {DEFAULT_CONFIG})")
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--stage", nargs="+", metavar="STAGE",
                        choices=["all", "test", "train", "verify", "report"],
                        default=["all"],
                        help="one or more of: all test train verify report "
                             "(e.g. --stage test verify report)")
    parser.add_argument("--output-root", default=None,
                        help="override output_root from the config")
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config key, e.g. -o overwrite=true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    if args.output_root:
        cfg.output_root = args.output_root

    tasks = list(TASKS) if args.task == "both" else [args.task]
    tasks = [t for t in tasks if cfg[t].get("enabled", True)]
    if not tasks:
        raise SystemExit("Nothing to do — every requested task is disabled in the config")

    generators = [build_generator(cfg, task) for task in tasks]
    print(f">> config {args.config}")
    stages = set(args.stage)
    everything = "all" in stages
    print(f">> tasks {tasks}, stages {sorted(stages)}, output {generators[0].out_root}",
          flush=True)

    for generator in generators:
        if everything or "test" in stages:
            generator.out_dir.mkdir(parents=True, exist_ok=True)
            generator.generate_test()
        if everything or "train" in stages:
            generator.generate_train()
        if everything and cfg.get("ft_links", False):
            generator.link_ft()

    # An explicit --stage runs that step even if the config switched it off.
    failures: list[tuple[str, float]] = []
    if "verify" in stages or (everything and cfg.verify.get("enabled", True)):
        min_pass = float(cfg.verify.min_pass)
        for generator in generators:
            for path, _rows, mean_pass, _cov in generator.verify():
                if mean_pass < min_pass:
                    failures.append((path, mean_pass))

    if "report" in stages or (everything and cfg.get("report", True)):
        for generator in generators:
            generator.report()

    if failures:
        print(f"\n!! {len(failures)} test parquet(s) scored below "
              f"min_pass={cfg.verify.min_pass} on their own ground truth:")
        for path, mean_pass in failures:
            print(f"   {mean_pass:.3f}  {path}")
        print("   Either the parquet is stale (regenerate it with overwrite=true) "
              "or the scorer is too strict.")
        if cfg.verify.get("fail_on_low", False):
            raise SystemExit(1)

    print(f"\nDone → {generators[0].out_root}")


if __name__ == "__main__":
    main(sys.argv[1:])
