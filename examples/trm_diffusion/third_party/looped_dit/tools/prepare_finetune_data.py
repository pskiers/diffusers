"""Download the fine-tuning data and store every source as WebDataset shards of (image, txt) pairs.

    python tools/prepare_finetune_data.py --out data/finetune                        # Std-120K (all models)
    python tools/prepare_finetune_data.py --out data/finetune --sources fine_t2i     # Fine-T2I (B/16, L/16)

Std-120K is BLIP3o-60K + DALL-E 3 + the text-to-image part of ShareGPT-4o-Image.
BLIP3o-60K and Fine-T2I are already (jpg, txt) shards and are downloaded as is;
DALL-E 3 comes as parquet files; ShareGPT-4o-Image keeps its captions in a JSON
file: captions longer than 256 T5 tokens are dropped and the images keep their
original PNG encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import webdataset as wds
from huggingface_hub import snapshot_download

BLIP3O_TARS = [
    "dalle3.tar", "geneval_train.tar", "human_gestures.tar", "journeyDB.tar", "mscoco_human.tar", "object_1.tar",
    "object_2.tar", "occupation_1.tar", "occupation_2.tar", "text_1.tar", "text_2.tar",
]


def download(repo: str, patterns: list[str], local_dir: Path | None = None, cache_dir: str | None = None) -> Path:
    return Path(snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns,
                                  local_dir=local_dir, cache_dir=cache_dir))


def blip3o_60k(out: Path, cache_dir: str | None) -> None:
    snapshot = download("BLIP3o/BLIP3o-60k", ["*.tar", "**/*.tar"], cache_dir=cache_dir)
    (out / "blip3o_60k").mkdir(parents=True, exist_ok=True)
    for tar in snapshot.rglob("*.tar"):
        if tar.name in BLIP3O_TARS and not (out / "blip3o_60k" / tar.name).exists():
            (out / "blip3o_60k" / tar.name).symlink_to(tar.resolve())


def fine_t2i(out: Path, cache_dir: str | None) -> None:
    download("ma-xu/fine-t2i", ["*.tar", "**/*.tar"], local_dir=out / "fine_t2i")


def dalle3(out: Path, cache_dir: str | None) -> None:
    import pyarrow.parquet as pq

    snapshot = download("OpenDatasets/dalle-3-dataset", ["*.parquet", "**/*.parquet"], cache_dir=cache_dir)
    (out / "dalle3").mkdir(parents=True, exist_ok=True)
    count = 0
    with wds.ShardWriter(str(out / "dalle3" / "shard-%06d.tar"), maxcount=10000) as sink:
        for parquet in sorted(snapshot.rglob("*.parquet")):
            for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    image = row.get("image")
                    image = image.get("bytes") if isinstance(image, dict) else image
                    if not image:
                        continue
                    caption = row.get("caption") or row.get("synthetic_caption") or ""
                    sink.write({"__key__": f"{row.get('image_hash') or 'sample'}-{count:012d}", "jpg": image, "txt": caption.strip() + "\n"})
                    count += 1


def sharegpt4o(out: Path, cache_dir: str | None, tokenizer_name: str = "google/flan-t5-large", max_tokens: int = 256) -> None:
    from transformers import AutoTokenizer

    snapshot = download("FreedomIntelligence/ShareGPT-4o-Image", ["text_to_image.json", "text_to_image_part_*.tar"], cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    captions = {}
    for row in json.loads((snapshot / "text_to_image.json").read_text(encoding="utf-8")):
        caption = str(row.get("input_prompt") or "")
        if tokenizer(caption, truncation=False, return_tensors="pt", verbose=False).input_ids.shape[1] <= max_tokens:
            captions[str(Path(row["output_image"]).with_suffix(""))] = caption + "\n"
    shards = sorted(str(p) for p in snapshot.glob("text_to_image_part_*.tar"))
    (out / "sharegpt4o").mkdir(parents=True, exist_ok=True)
    with wds.ShardWriter(str(out / "sharegpt4o" / "shard-%06d.tar"), maxcount=10000) as sink:
        for sample in wds.WebDataset(shards, handler=wds.warn_and_continue, empty_check=False, shardshuffle=False):
            caption = captions.get(sample["__key__"])
            if caption is None or "png" not in sample:
                continue
            sink.write({"__key__": sample["__key__"].replace("/", "_"), "png": sample["png"], "txt": caption})


SOURCES = {"blip3o_60k": blip3o_60k, "dalle3": dalle3, "sharegpt4o": sharegpt4o, "fine_t2i": fine_t2i}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources", nargs="+", default=["blip3o_60k", "dalle3", "sharegpt4o"], choices=sorted(SOURCES))
    parser.add_argument("--cache-dir", help="Hugging Face cache for the raw downloads (Fine-T2I downloads into --out)")
    args = parser.parse_args()
    for name in args.sources:
        print(f"preparing {name}", flush=True)
        SOURCES[name](Path(args.out), args.cache_dir)


if __name__ == "__main__":
    main()
