"""Std120KDataset vs Looped-DiT's real pipeline (third_party/looped_dit), on the
same real raw files.

The fixture is the head of each source's real raw data (streamed from the Hub
once, cached in data/std120k_testfixture), plus edge cases each pipeline must
treat identically: a corrupt image, an image without a caption, a caption
without an image (BLIP3o); rows without an image or with missing captions
(DALL-E 3); an image without a JSON caption and a caption over 256 T5 tokens
(ShareGPT-4o). Both pipelines run on it — theirs through their real
prepare_finetune_data.py + WebDatasetMix, ours through build + Std120KDataset —
and every sample must match exactly: pixels as fed to the model, token ids,
attention mask.

Run from examples/trm_diffusion:  pytest tests/test_std120k_dataset.py -s
"""

import hashlib
import io
import json
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.std120k_dataset import SOURCE_WEIGHTS, SOURCES, LocalRawFiles, Std120KDataset  # noqa: E402

FIXTURE = ROOT / "data" / "std120k_testfixture"
N_HEAD = 16  # real samples taken from each source


def _repo_dir(name: str) -> Path:
    return FIXTURE / SOURCES[name].repo.replace("/", "__")


def _stream_tar_head(repo: str, filename: str, n_samples: int) -> list[tuple[str, bytes]]:
    """Members of the first n_samples samples of a remote tar, without downloading all of it."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import get_session

    resp = get_session().get(hf_hub_url(repo, filename, repo_type="dataset"), stream=True, timeout=60)
    resp.raise_for_status()
    resp.raw.decode_content = True
    members, keys = [], []
    with tarfile.open(fileobj=resp.raw, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key = m.name.rsplit(".", 1)[0]
            if key not in keys:
                if len(keys) == n_samples:
                    break
                keys.append(key)
            members.append((m.name, tf.extractfile(m).read()))
    resp.close()
    return members


def _write_tar(path: Path, members: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _build_fixture() -> dict:
    """Returns the number of samples each source must yield."""
    meta_path = FIXTURE / "expected.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    expected = {}

    # BLIP3o: real head of object_2.tar + three broken samples at the end.
    members = _stream_tar_head("BLIP3o/BLIP3o-60k", "object_2.tar", N_HEAD)
    an_image = members[0][1]
    members += [
        ("zzcorrupt.jpg", b"definitely not an image"), ("zzcorrupt.txt", b"a caption"),  # skipped
        ("zznotxt.jpg", an_image),                                                     # kept, caption ""
        ("zzonlytxt.txt", b"caption without an image"),                                # skipped
    ]
    _write_tar(_repo_dir("blip3o_60k") / "object_2.tar", members)
    expected["blip3o_60k"] = N_HEAD + 1

    # DALL-E 3: real first rows of the first parquet + rows exercising the prep's branches.
    from huggingface_hub import hf_hub_download

    src = hf_hub_download("OpenDatasets/dalle-3-dataset", "data/train-00000-of-00067.parquet",
                          repo_type="dataset", local_dir=FIXTURE / "_download")
    table = pq.read_table(src)
    rows = table.slice(0, N_HEAD).to_pylist()
    rows += [
        {**rows[0], "image": None},                                   # skipped: no image
        {**rows[1], "caption": None},                                 # kept: falls back to synthetic_caption
        {**rows[2], "caption": None, "synthetic_caption": None},      # kept: caption "\n"
    ]
    out = _repo_dir("dalle3") / "data" / "train-00000-of-00067.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), out)
    Path(src).unlink()
    expected["dalle3"] = N_HEAD + 2

    # ShareGPT-4o: real head of text_to_image_part_0.tar + the real captions JSON,
    # minus one image's entry and with one caption made longer than 256 tokens.
    members = _stream_tar_head("FreedomIntelligence/ShareGPT-4o-Image", "text_to_image_part_0.tar", N_HEAD)
    _write_tar(_repo_dir("sharegpt4o") / "text_to_image_part_0.tar", members)
    caption_json = hf_hub_download("FreedomIntelligence/ShareGPT-4o-Image", "text_to_image.json",
                                   repo_type="dataset", local_dir=FIXTURE / "_download")
    head_keys = [name.rsplit(".", 1)[0] for name, _ in members]
    by_key = {str(Path(r["output_image"]).with_suffix("")): r for r in json.loads(Path(caption_json).read_text())}
    assert all(k in by_key for k in head_keys)
    rows = [by_key[k] for k in head_keys[1:]]                     # head_keys[0]: no caption -> skipped
    rows[0] = {**rows[0], "input_prompt": "a very long caption " * 80}  # > 256 tokens -> filtered
    rows += [r for k, r in list(by_key.items())[:20] if k not in head_keys]  # captions with no image here
    (_repo_dir("sharegpt4o") / "text_to_image.json").write_text(json.dumps(rows))
    Path(caption_json).unlink()
    # Many real ShareGPT captions are themselves over 256 tokens, so count what the rule keeps.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
    kept = [r for r in rows[: len(head_keys) - 1] if len(tok(r["input_prompt"], verbose=False).input_ids) <= 256]
    assert len(kept) < N_HEAD - 1, "the synthetic over-length caption should have been filtered"
    expected["sharegpt4o"] = len(kept)

    meta_path.write_text(json.dumps(expected))
    return expected


def _sample_key(images: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in (images.float(), ids.long(), mask.long()):
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def expected_counts():
    return _build_fixture()


@pytest.fixture(scope="module")
def reference(tmp_path_factory, expected_counts):
    """Their pipeline: prepare_finetune_data.py, then one pass of WebDatasetMix per source."""
    import third_party.looped_dit.tools.prepare_finetune_data as prep
    from third_party.looped_dit.looped_dit.config import TrainConfig
    from third_party.looped_dit.looped_dit.data import WebDatasetMix

    out = tmp_path_factory.mktemp("looped_dit_finetune")
    real_download = prep.download
    prep.download = lambda repo, patterns, local_dir=None, cache_dir=None: FIXTURE / repo.replace("/", "__")
    try:
        for name in SOURCES:
            prep.SOURCES[name](out, None)
    finally:
        prep.download = real_download
    cfg = TrainConfig(
        dataset="webdataset",
        webdataset_sources={n: {"path": str(out / n), "weight": w} for n, w in SOURCE_WEIGHTS.items()},
        shuffle_buffer=1,
    )
    mix = WebDatasetMix(cfg)
    ref = {}
    for i, name in enumerate(mix.names):
        ref[name] = Counter(
            # looped_dit/train.py turns pixel_values into the model's input this way
            _sample_key(s["pixel_values"].float().mul_(1 / 127.5).add_(-1.0), s["input_ids"], s["attention_mask"])
            for s in mix.source(i, epoch=0)
        )
    return ref


@pytest.fixture(scope="module")
def built_root(tmp_path_factory, expected_counts):
    root = tmp_path_factory.mktemp("std120k")
    Std120KDataset(root=root, raw_files=LocalRawFiles(FIXTURE), build_workers=4)
    return root


@pytest.mark.parametrize("name", list(SOURCES))
def test_identical_to_looped_dit_pipeline(name, reference, built_root, expected_counts):
    ds = Std120KDataset(root=built_root, sources=(name,), download=False)
    ours = Counter(_sample_key(s.images, s.token_conditions, s.token_mask) for s in (ds[i] for i in range(len(ds))))
    assert sum(reference[name].values()) == expected_counts[name], "fixture edge cases not hit as designed"
    assert sum(ours.values()) == expected_counts[name]
    assert ours == reference[name]


def test_sample_format(built_root):
    ds = Std120KDataset(root=built_root, download=False)
    s = ds[0]
    assert s.images.shape == (3, 512, 512) and s.images.dtype == torch.float32
    assert -1.0 <= s.images.min() and s.images.max() <= 1.0
    assert s.token_conditions.shape == s.token_mask.shape == (256,)
    batch = Std120KDataset.collate_fn([ds[0], ds[1]])
    assert batch.images.shape == (2, 3, 512, 512) and batch.token_mask.shape == (2, 256)
    meta = json.loads((built_root / "blip3o_60k" / "DONE").read_text())
    assert meta["boundary_key_collisions"] == []
    files = [p for p in built_root.rglob("*") if p.is_file() and p.name != ".build.lock"]
    assert len(files) == 3 * len(SOURCES), files  # images.bin + index.parquet + DONE per source


def test_max_samples_subset(built_root):
    full = Std120KDataset(root=built_root, download=False)
    sub = Std120KDataset(root=built_root, download=False, max_samples=10, subset_seed=3)
    again = Std120KDataset(root=built_root, download=False, max_samples=10, subset_seed=3)
    assert len(sub) == 10
    captions = {full.caption(i) for i in range(len(full))}
    assert all(sub.caption(i) in captions for i in range(10))
    assert [sub.caption(i) for i in range(10)] == [again.caption(i) for i in range(10)]


def test_holdout_excludes_val_subset(built_root):
    full = Std120KDataset(root=built_root, download=False)
    val = Std120KDataset(root=built_root, download=False, max_samples=10, subset_seed=3)
    train = Std120KDataset(root=built_root, download=False, holdout=10, subset_seed=3)
    keys = lambda ds: {(int(s), int(o)) for s, o in zip(ds._source_ids, ds._offsets)}  # (source, offset): unique
    assert len(train) == len(full) - 10
    assert not keys(train) & keys(val) and keys(train) | keys(val) == keys(full)
    assert (np.diff(train._source_ids) >= 0).all()  # sources stay contiguous for the mixture sampler
    assert set(train.make_train_sampler(seed=0)) <= set(range(len(train)))


def test_mixture_sampler(built_root):
    ds = Std120KDataset(root=built_root, download=False)
    sampler = ds.make_train_sampler(seed=7)
    sampler.num_samples = 60_000
    draws = np.array(list(sampler))
    src = ds._source_ids[draws]
    freq = np.bincount(src, minlength=3) / len(draws)
    p = np.array(ds.weights) / sum(ds.weights)
    assert np.allclose(freq, p, atol=0.01), (freq, p)
    # Each source is walked without replacement within a pass.
    first_pass = draws[src == 0][: int((ds._source_ids == 0).sum())]
    assert len(set(first_pass.tolist())) == len(first_pass)
    # Same seed and epoch on every rank -> same stream.
    a, b = ds.make_train_sampler(seed=7), ds.make_train_sampler(seed=7)
    assert list(a) == list(b)  # epoch 0 on both
    assert list(a) == list(b)  # both advanced to epoch 1
    a.set_epoch(5), b.set_epoch(5)
    assert list(a) == list(b)


def test_loading_throughput(built_root):
    """Per-sample load cost (read + decode + scale + tokenize) = one DataLoader
    worker's throughput; a DataLoader with N workers scales ~linearly. Printed;
    the floor is loose — it only guards against an order-of-magnitude regression."""
    ds = Std120KDataset(root=built_root, download=False)
    ds[0]  # warm up the tokenizer
    idx = np.random.default_rng(0).integers(0, len(ds), 100)
    t0 = time.perf_counter()
    for i in idx:
        ds[int(i)]
    rate = len(idx) / (time.perf_counter() - t0)
    bytes_per_img = sum((built_root / s / "images.bin").stat().st_size for s in SOURCES) / len(ds)
    print(f"\nStd-120K loading: {rate:.0f} samples/s per DataLoader worker; "
          f"{bytes_per_img / 1e3:.0f} KB per stored image")
    assert rate > 50
