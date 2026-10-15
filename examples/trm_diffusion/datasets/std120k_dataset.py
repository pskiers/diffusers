"""
Std120KDataset — the Std-120K text-to-image fine-tuning set, sample for sample
identical to the fine-tuning data of Looped-DiT (arXiv 2609.40305). It is also
MiniT2I's own fine-tuning set, i.e. what our frozen MiniT2I-B/16 painter was
last trained on.

Std-120K mixes three Hugging Face datasets (sampling weights from their
configs/b32_finetune.yml, = MiniT2I's finetune.yml mix_weights):

  blip3o_60k  BLIP3o/BLIP3o-60k                      11 tars                    0.06
  dalle3      OpenDatasets/dalle-3-dataset           67 parquet files           0.016
  sharegpt4o  FreedomIntelligence/ShareGPT-4o-Image  text-to-image part:        0.04
                                                     10 tars + captions JSON

Faithfulness
------------
third_party/looped_dit vendors their real prep (tools/prepare_finetune_data.py)
and loader (looped_dit/data.py). For every raw sample this module does the same
thing: the same per-source sample construction (DALL-E's caption fallback and
"\\n" suffix, ShareGPT's >256-T5-token caption filter, ...), webdataset's own
tar grouping and "pil" decoding, their image/caption selection, then
Resize(512, bicubic) + CenterCrop(512) — and it skips exactly the samples their
warn_and_continue handlers skip. Images are returned scaled to [-1, 1] exactly
as their train.py does it, captions tokenized exactly as their loader does it.
tests/test_std120k_dataset.py runs both pipelines on the same real raw files and
checks every sample is equal (pixels, token ids, attention mask).

Storage
-------
The resized 512x512 image is stored instead of the ~1.5 MB 1024^2 original:
their transform is idempotent on its own output, so this is exact, and lossless
WebP keeps it bit-exact at ~256 KB per image (~32 GB in total vs ~230 GB raw).
Each source is one images.bin (WebP blobs back to back) + one index.parquet
(offset, length, caption, key): a handful of files in total, and one pread per
sample.

Building
--------
A source whose <root>/<source>/DONE is missing is built on first use. Raw files
are downloaded one at a time into <root>/_raw (never the home-directory HF
cache) and deleted once processed, so the build needs about one raw file
(<= ~18 GB) of scratch space on top of the output. Each processed raw file
becomes a part in <root>/<source>/_parts, so an interrupted build resumes where
it stopped; the parts are merged at the end. Prebuild explicitly (recommended
before multi-GPU runs, where every rank would otherwise wait on the build lock):

    python -m datasets.std120k_dataset --root data/std120k

Raw files are processed one at a time, whereas their pipeline streams a whole
source's tars in one go — the same unless a sample's key spans two tars, which
the build checks for and reports in DONE (boundary_key_collisions).

Sampling
--------
Their loader draws each sample from a source picked with probability
proportional to its weight, walking that source in a fresh random order per
pass. Std120KMixtureSampler reproduces that; train_trm.py picks it up through
make_train_sampler().
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import logging
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from datasets.data_sample import DataSample, collate_data_samples
from third_party.looped_dit.tools.prepare_finetune_data import BLIP3O_TARS

logger = logging.getLogger(__name__)

LOOPED_DIT_COMMIT = "65a7705ad2954fa127721bd1131ea8f865688848"  # see third_party/looped_dit/README.md
FORMAT_VERSION = 1
WEBP_KWARGS = dict(format="WEBP", lossless=True, quality=100, method=4, exact=True)
TEXT_ENCODER = "google/flan-t5-large"
PROMPT_LENGTH = 256
SOURCE_WEIGHTS = {"blip3o_60k": 0.06, "dalle3": 0.016, "sharegpt4o": 0.04}


# ── Raw samples, as their loader would read them from the prepared shards ────


def _identity(src):
    return src


def _tar_samples(path: Path) -> Iterator[dict]:
    """Undecoded samples of one tar, read and grouped by webdataset with the same
    options as their loader (looped_dit/data.py: WebDatasetMix.source)."""
    import webdataset as wds

    yield from wds.WebDataset(
        [str(path)],
        handler=wds.warn_and_continue,
        empty_check=False,
        shardshuffle=False,
        nodesplitter=_identity,
        workersplitter=_identity,
    )


def _blip3o_samples(path: Path, aux) -> Iterator[dict]:
    # prepare_finetune_data.py:blip3o_60k symlinks the tars unchanged.
    yield from _tar_samples(path)


def _dalle3_samples(path: Path, aux) -> Iterator[dict]:
    # prepare_finetune_data.py:dalle3, for one parquet file. Their ShardWriter
    # stores "txt" as UTF-8 and their loader reads those bytes back, so the
    # caption is handed on as bytes; the image bytes pass through untouched.
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            image = row.get("image")
            image = image.get("bytes") if isinstance(image, dict) else image
            if not image:
                continue
            caption = row.get("caption") or row.get("synthetic_caption") or ""
            yield {"__key__": row.get("image_hash") or "sample", "jpg": image, "txt": (caption.strip() + "\n").encode("utf-8")}


def _sharegpt4o_captions(json_path: Path, max_tokens: int = PROMPT_LENGTH) -> dict[str, str]:
    # prepare_finetune_data.py:sharegpt4o — captions longer than 256 T5 tokens are dropped.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    captions = {}
    for row in json.loads(Path(json_path).read_text(encoding="utf-8")):
        caption = str(row.get("input_prompt") or "")
        if tokenizer(caption, truncation=False, return_tensors="pt", verbose=False).input_ids.shape[1] <= max_tokens:
            captions[str(Path(row["output_image"]).with_suffix(""))] = caption + "\n"
    return captions


def _sharegpt4o_samples(path: Path, captions: dict[str, str]) -> Iterator[dict]:
    # prepare_finetune_data.py:sharegpt4o, for one text_to_image tar.
    for sample in _tar_samples(path):
        caption = captions.get(sample["__key__"])
        if caption is None or "png" not in sample:
            continue
        yield {"__key__": sample["__key__"].replace("/", "_"), "png": sample["png"], "txt": caption.encode("utf-8")}


@dataclass(frozen=True)
class _SourceSpec:
    repo: str
    data_files: Callable[[list[str]], list[str]]  # repo file list -> files to process, in order
    raw_samples: Callable[[Path, object], Iterator[dict]]
    aux_file: Optional[str] = None  # downloaded first, turned into `aux` by load_aux
    load_aux: Optional[Callable[[Path], object]] = None


SOURCES = {
    "blip3o_60k": _SourceSpec(
        repo="BLIP3o/BLIP3o-60k",
        data_files=lambda files: sorted(f for f in files if f in BLIP3O_TARS),
        raw_samples=_blip3o_samples,
    ),
    "dalle3": _SourceSpec(
        repo="OpenDatasets/dalle-3-dataset",
        data_files=lambda files: sorted(f for f in files if f.endswith(".parquet")),
        raw_samples=_dalle3_samples,
    ),
    "sharegpt4o": _SourceSpec(
        repo="FreedomIntelligence/ShareGPT-4o-Image",
        data_files=lambda files: sorted(
            f for f in files if Path(f).name.startswith("text_to_image_part_") and f.endswith(".tar")
        ),
        raw_samples=_sharegpt4o_samples,
        aux_file="text_to_image.json",
        load_aux=_sharegpt4o_captions,
    ),
}


# ── Raw file access ───────────────────────────────────────────────────────────


class HubRawFiles:
    """Downloads raw files from the Hub into `raw_dir` one at a time (never the
    HF home cache) and deletes each once it has been processed."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def list_files(self, repo: str) -> list[str]:
        from huggingface_hub import HfApi

        return HfApi().list_repo_files(repo, repo_type="dataset")

    def fetch(self, repo: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        local_dir = self.raw_dir / repo.replace("/", "__")
        return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=local_dir))

    def release(self, path: Path) -> None:
        Path(path).unlink(missing_ok=True)

    def finish(self, repo: str) -> None:
        shutil.rmtree(self.raw_dir / repo.replace("/", "__"), ignore_errors=True)


class LocalRawFiles:
    """Raw files already on disk, laid out as in each repo under
    <root>/<repo with / replaced by __>/ (e.g. tests' fixtures). Never deleted."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def list_files(self, repo: str) -> list[str]:
        base = self.root / repo.replace("/", "__")
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())

    def fetch(self, repo: str, filename: str) -> Path:
        return self.root / repo.replace("/", "__") / filename

    def release(self, path: Path) -> None:
        pass

    def finish(self, repo: str) -> None:
        pass


# ── Per-sample processing (build workers) ─────────────────────────────────────

_WORKER: dict = {}


def _to_pixels_and_caption(transform, sample: dict):
    # looped_dit/data.py: WebDatasetMix.to_sample, minus the tokenization
    # (done at load time instead, so the caption is stored as text).
    image = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
    if not isinstance(image, Image.Image):
        raise ValueError(f"sample {sample.get('__key__')} has no image")
    caption = sample.get("txt", "")
    caption = caption.decode("utf-8", errors="replace") if isinstance(caption, bytes) else str(caption)
    return transform(image.convert("RGB")), caption


def _init_worker(image_size: int) -> None:
    import webdataset as wds
    from PIL import ImageFile
    from torchvision import transforms

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # looped_dit/data.py sets this at import
    transform = transforms.Compose(  # looped_dit/data.py: WebDatasetMix.__init__
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ]
    )
    # The same webdataset stages (and failure handler) as their loader's
    # .decode("pil", ...).map(self.to_sample, ...).
    _WORKER["decode"] = wds.decode("pil", handler=wds.warn_and_continue)
    _WORKER["map"] = wds.map(functools.partial(_to_pixels_and_caption, transform), handler=wds.warn_and_continue)


def _process(raw: dict) -> Optional[tuple[bytes, str, str]]:
    out = list(_WORKER["map"](_WORKER["decode"](iter([raw]))))
    if not out:
        return None  # skipped, exactly as their warn_and_continue would
    pixels, caption = out[0]
    buf = io.BytesIO()
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(buf, **WEBP_KWARGS)
    return buf.getvalue(), caption, str(raw.get("__key__", ""))


# ── Building ──────────────────────────────────────────────────────────────────

_INDEX_SCHEMA = pa.schema(
    [("offset", pa.int64()), ("length", pa.int64()), ("caption", pa.string()), ("key", pa.string())]
)


def _build_part(raw_iter: Iterator[dict], bin_path: Path, pool, desc: str) -> dict:
    """Process one raw file into <part>.bin + <part>.parquet; returns its metadata."""
    offsets, lengths, captions, keys = [], [], [], []
    stats = {"n_raw": 0, "first_key": None, "last_key": None}

    def tracked():
        for raw in raw_iter:
            stats["n_raw"] += 1
            stats["first_key"] = stats["first_key"] or raw.get("__key__")
            stats["last_key"] = raw.get("__key__")
            yield raw

    tmp_bin = bin_path.with_suffix(".bin.tmp")
    with open(tmp_bin, "wb") as f:
        for res in tqdm(pool.imap(_process, tracked(), chunksize=4), desc=desc, unit="img", leave=False):
            if res is None:
                continue
            blob, caption, key = res
            offsets.append(f.tell())
            lengths.append(len(blob))
            captions.append(caption)
            keys.append(key)
            f.write(blob)
    table = pa.table({"offset": offsets, "length": lengths, "caption": captions, "key": keys}, schema=_INDEX_SCHEMA)
    pq.write_table(table, bin_path.with_suffix(".parquet.tmp"))
    os.replace(bin_path.with_suffix(".parquet.tmp"), bin_path.with_suffix(".parquet"))
    os.replace(tmp_bin, bin_path)  # last: a part counts as done once its .bin exists
    return {"n_samples": len(offsets), "n_skipped": stats["n_raw"] - len(offsets),
            "first_key": stats["first_key"], "last_key": stats["last_key"]}


def _boundary_key_collisions(parts: list[dict]) -> list[str]:
    """A sample key ending one tar and starting another would, in their
    one-stream reader, be grouped into a single sample; we process tars
    separately. Any collision is reported (none is expected)."""
    firsts = {p["first_key"]: p["file"] for p in parts if p["first_key"] is not None}
    return [
        f"{p['file']} ends with key {p['last_key']!r}, which starts {firsts[p['last_key']]}"
        for p in parts
        if p["last_key"] in firsts and firsts[p["last_key"]] != p["file"]
    ]


def build_source(
    root: Path,
    name: str,
    image_size: int = 512,
    raw_files=None,
    workers: Optional[int] = None,
) -> None:
    """Build <root>/<name>/{images.bin, index.parquet, DONE} (see module docstring)."""
    spec = SOURCES[name]
    root = Path(root)
    raw_files = raw_files or HubRawFiles(root / "_raw")
    src_dir = root / name
    parts_dir = src_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or len(os.sched_getaffinity(0))
    logger.info(f"Std-120K: building {name} into {src_dir} with {workers} workers "
                f"({shutil.disk_usage(root).free / 1e9:.0f} GB free)")

    files = spec.data_files(raw_files.list_files(spec.repo))
    if not files:
        raise FileNotFoundError(f"no raw files found for {name} ({spec.repo})")
    aux, aux_path = None, None
    parts = []
    # fork, like DataLoader workers: the workers only run PIL/webdataset on the
    # CPU, and unlike spawn it never re-imports the caller's main script (with
    # spawn, a caller without one, e.g. stdin, leaves Pool respawning workers
    # forever instead of failing).
    pool = mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(image_size,))
    try:
        for i, filename in enumerate(files):
            bin_path = parts_dir / f"{i:05d}.bin"
            meta_path = parts_dir / f"{i:05d}.json"
            if not (bin_path.exists() and meta_path.exists()):
                if spec.aux_file is not None and aux is None:
                    aux_path = raw_files.fetch(spec.repo, spec.aux_file)
                    aux = spec.load_aux(aux_path)
                raw_path = raw_files.fetch(spec.repo, filename)
                meta = _build_part(spec.raw_samples(raw_path, aux), bin_path, pool, f"{name} {i + 1}/{len(files)}")
                meta["file"] = filename
                meta_path.write_text(json.dumps(meta))
                raw_files.release(raw_path)
            parts.append(json.loads(meta_path.read_text()))
    finally:
        pool.close()
        pool.join()
    if aux_path is not None:
        raw_files.release(aux_path)
    raw_files.finish(spec.repo)

    # Merge the parts into one images.bin + index.parquet.
    tables = []
    with open(src_dir / "images.bin.tmp", "wb") as out:
        for i in range(len(files)):
            base = out.tell()
            with open(parts_dir / f"{i:05d}.bin", "rb") as f:
                shutil.copyfileobj(f, out, length=64 << 20)
            t = pq.read_table(parts_dir / f"{i:05d}.parquet")
            tables.append(t.set_column(0, "offset", pc.add(t["offset"], base)))
    pq.write_table(pa.concat_tables(tables), src_dir / "index.parquet")
    os.replace(src_dir / "images.bin.tmp", src_dir / "images.bin")
    collisions = _boundary_key_collisions(parts)
    for c in collisions:
        logger.warning(f"Std-120K {name}: {c}")
    done = {
        "format_version": FORMAT_VERSION,
        "looped_dit_commit": LOOPED_DIT_COMMIT,
        "repo": spec.repo,
        "image_size": image_size,
        "codec": WEBP_KWARGS,
        "n_samples": sum(p["n_samples"] for p in parts),
        "n_skipped": sum(p["n_skipped"] for p in parts),
        "raw_files": [p["file"] for p in parts],
        "boundary_key_collisions": collisions,
    }
    (src_dir / "DONE").write_text(json.dumps(done, indent=2))
    shutil.rmtree(parts_dir)
    logger.info(f"Std-120K: {name} done — {done['n_samples']} samples, {done['n_skipped']} skipped")


def ensure_built(
    root: Path, sources: Sequence[str], image_size: int = 512, download: bool = True, raw_files=None,
    workers: Optional[int] = None,
) -> None:
    """Build every missing source, one process at a time (file lock)."""
    from filelock import FileLock

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".build.lock")):
        for name in sources:
            done = root / name / "DONE"
            if not done.exists():
                if not download:
                    raise FileNotFoundError(f"{done} missing and download=False")
                build_source(root, name, image_size, raw_files=raw_files, workers=workers)
            meta = json.loads(done.read_text())
            if meta["format_version"] != FORMAT_VERSION or meta["image_size"] != image_size:
                raise ValueError(
                    f"{root / name} was built with format {meta['format_version']} at {meta['image_size']}px; "
                    f"expected format {FORMAT_VERSION} at {image_size}px — delete it to rebuild"
                )


# ── Dataset ───────────────────────────────────────────────────────────────────


class Std120KDataset(Dataset):
    """
    Args:
        root: directory holding (or receiving) the built sources.
        sources: which Std-120K sources to include.
        weights: per-source sampling weights for make_train_sampler();
            default SOURCE_WEIGHTS (theirs).
        image_size: 512 (theirs); a different size is a different dataset.
        tokenizer, prompt_length: caption tokenization, as their loader's.
        max_samples: keep a fixed random subset of this many samples (seeded
            by subset_seed) — e.g. a small validation set. Std-120K has no
            held-out split, so such a subset overlaps the training data...
        holdout: ...unless the training set drops it: holdout=k removes exactly
            the samples that max_samples=k (same subset_seed) keeps. 0 = the
            full dataset, as Looped-DiT trains on.
        download: build missing sources (downloading their raw files).
        build_workers: processes used when building (default: all CPUs).

    Each sample:
        images            – (3, 512, 512) float32 in [-1, 1]
                              (looped_dit/train.py: pixel_values / 127.5 - 1).
        token_conditions  – (prompt_length,) long, caption token ids.
        token_mask        – (prompt_length,) long, 1 = real token.
    """

    collate_fn = staticmethod(collate_data_samples)

    def __init__(
        self,
        root: str = "data/std120k",
        sources: Sequence[str] = ("blip3o_60k", "dalle3", "sharegpt4o"),
        weights: Optional[dict] = None,
        image_size: int = 512,
        tokenizer: str = TEXT_ENCODER,
        prompt_length: int = PROMPT_LENGTH,
        max_samples: Optional[int] = None,
        subset_seed: int = 0,
        holdout: int = 0,
        download: bool = True,
        build_workers: Optional[int] = None,
        raw_files=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.sources = list(sources)
        self.weights = [float((weights or SOURCE_WEIGHTS)[s]) for s in self.sources]
        self.tokenizer_name = tokenizer
        self.prompt_length = prompt_length
        ensure_built(self.root, self.sources, image_size, download, raw_files, build_workers)

        offsets, lengths, captions, source_ids = [], [], [], []
        for i, name in enumerate(self.sources):
            t = pq.read_table(self.root / name / "index.parquet", columns=["offset", "length", "caption"])
            offsets.append(t["offset"].to_numpy())
            lengths.append(t["length"].to_numpy())
            captions.extend(t["caption"].to_pylist())
            source_ids.append(np.full(len(t), i, dtype=np.int64))
        self._offsets = np.concatenate(offsets)
        self._lengths = np.concatenate(lengths)
        self._captions = np.array(captions, dtype=object)
        self._source_ids = np.concatenate(source_ids)
        if holdout and max_samples is not None:
            raise ValueError("max_samples and holdout select complementary subsets; set one")
        if holdout or (max_samples is not None and max_samples < len(self._offsets)):
            perm = np.random.default_rng(subset_seed).permutation(len(self._offsets))
            keep = np.sort(perm[holdout:] if holdout else perm[:max_samples])  # sorted: sources stay contiguous
            self._offsets, self._lengths = self._offsets[keep], self._lengths[keep]
            self._captions, self._source_ids = self._captions[keep], self._source_ids[keep]

        self._bin_paths = [str(self.root / name / "images.bin") for name in self.sources]
        self._fds: dict = {}  # per process: opened lazily after DataLoader workers fork
        self._tokenizer = None

    def __len__(self) -> int:
        return len(self._offsets)

    def _fd(self, source_id: int) -> int:
        key = (os.getpid(), source_id)
        if key not in self._fds:
            self._fds[key] = os.open(self._bin_paths[source_id], os.O_RDONLY)
        return self._fds[key]

    def _tokenize(self, caption: str):
        if self._tokenizer is None:  # looped_dit/data.py: WebDatasetMix.__init__
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, model_max_length=self.prompt_length)
        return self._tokenizer(  # looped_dit/data.py: WebDatasetMix.to_sample
            caption, max_length=self.prompt_length, padding="max_length", truncation=True, return_tensors="pt"
        )

    def __getitem__(self, idx: int) -> DataSample:
        sid = int(self._source_ids[idx])
        blob = os.pread(self._fd(sid), int(self._lengths[idx]), int(self._offsets[idx]))
        with Image.open(io.BytesIO(blob)) as im:
            pixels = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)
        tokens = self._tokenize(self._captions[idx])
        return DataSample(
            images=pixels.float().mul_(1 / 127.5).add_(-1.0),  # looped_dit/train.py
            token_conditions=tokens["input_ids"][0].long(),
            token_mask=tokens["attention_mask"][0].long(),
        )

    def caption(self, idx: int) -> str:
        return str(self._captions[idx])

    def make_train_sampler(self, seed: int = 42) -> "Std120KMixtureSampler":
        bounds = np.searchsorted(self._source_ids, np.arange(len(self.sources) + 1))
        return Std120KMixtureSampler(list(zip(bounds[:-1], bounds[1:])), self.weights, len(self), seed)


class Std120KMixtureSampler(Sampler[int]):
    """looped_dit/data.py: WebDatasetMix.__iter__ — every draw picks a source
    with probability proportional to its weight and takes that source's next
    sample, each source walked in a fresh random order per pass over it.

    Deterministic in (seed, epoch), so every rank produces the same index
    stream and accelerate splits it into disjoint per-rank batches.
    """

    def __init__(self, ranges: list[tuple[int, int]], weights: Sequence[float], num_samples: int, seed: int = 42):
        self.ranges = [(int(a), int(b)) for a, b in ranges]
        w = np.array([wt if b > a else 0.0 for wt, (a, b) in zip(weights, self.ranges)], dtype=np.float64)
        self.p = w / w.sum()
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        orders: list = [None] * len(self.ranges)
        pos = [0] * len(self.ranges)
        for s in rng.choice(len(self.ranges), size=self.num_samples, p=self.p):
            a, b = self.ranges[s]
            if orders[s] is None or pos[s] == b - a:
                orders[s], pos[s] = a + rng.permutation(b - a), 0
            yield int(orders[s][pos[s]])
            pos[s] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build Std-120K (see module docstring).")
    parser.add_argument("--root", default="data/std120k")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_built(Path(args.root), args.sources, args.image_size, download=True, workers=args.workers)
    for name in args.sources:
        meta = json.loads((Path(args.root) / name / "DONE").read_text())
        print(f"{name}: {meta['n_samples']} samples, {meta['n_skipped']} skipped, "
              f"{(Path(args.root) / name / 'images.bin').stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
