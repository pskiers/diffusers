"""
eval/corebench_eval.py — T2I-CoReBench (Li et al., ICLR 2026) image generation
and score aggregation for text-to-image painters/thinkers; the judging itself
runs in a separate environment via eval/corebench_judge.py.

The 1,080 prompts (12 dimensions x 90) and their checklists are the official
data (third_party/t2i_corebench/data), loaded with Looped-DiT's own loader.
Images are generated as in Looped-DiT's looped_dit/eval/generate.py — 4 per
prompt, PNG, converted to 8-bit as their pipeline does — with one deliberate
difference: each image's noise is seeded by (seed, prompt, sample) only, not
also by the GPU rank, so scores don't depend on how many GPUs generated them.
The `official` protocol also gets the official sample.py's crop to the image's
non-black bounding box and its file layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from statistics import mean
from typing import Optional

import torch
from PIL import Image

from datasets.data_sample import DataSample
from models.sampling import CFGPredictor, SamplingPipeline
from third_party.looped_dit.looped_dit.eval.benchmarks import Prompt, image_path, load_prompts

ROOT = Path(__file__).resolve().parents[1]
COREBENCH_DATA = ROOT / "third_party" / "t2i_corebench" / "data"
COMPOSITION = ("C-MI", "C-MA", "C-MR", "C-TR")
REASONING = ("R-LR", "R-BR", "R-HR", "R-PR", "R-GR", "R-AR", "R-CR", "R-RR")
PROTOCOLS = ("looped_dit", "official")


def corebench_prompts(per_dimension: Optional[int] = None) -> list[Prompt]:
    """The benchmark's prompts; per_dimension keeps only the first k of each
    of the 12 dimensions (a cheaper, noisier subset — not comparable to
    full-benchmark scores)."""
    prompts = load_prompts("corebench", str(COREBENCH_DATA))
    if per_dimension is None:
        return prompts
    seen: dict[str, int] = {}
    kept = []
    for p in prompts:
        if seen.get(p.group, 0) < per_dimension:
            kept.append(p)
            seen[p.group] = seen.get(p.group, 0) + 1
    return kept


def write_prompt_files(prompts: list[Prompt], data_dir: Path) -> None:
    """Prompt files in the official data/<task>.json format, for judging a
    subset (eval/corebench_judge.py --data)."""
    import json

    data_dir.mkdir(parents=True, exist_ok=True)
    for task in COMPOSITION + REASONING:
        rows = {p.item_id: p.metadata for p in prompts if p.group == task}
        (data_dir / f"{task}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def output_path(protocol: str, image_dir: Path, prompt: Prompt, sample: int, model_name: str = "model") -> Path:
    """Where each protocol's judge expects sample `sample` of `prompt`."""
    if protocol == "looped_dit":
        return image_path(image_dir, prompt, sample)  # <task>/<item_id>/<sample:05d>.png
    if protocol == "official":
        return Path(image_dir) / model_name / prompt.group / f"{prompt.item_id}-{sample}.png"  # sample.py
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def to_pil(images: torch.Tensor) -> list[Image.Image]:
    """[-1, 1] images → 8-bit PIL, exactly as looped_dit/pipeline.py: to_pil."""
    images = (images.float().clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    return [Image.fromarray(image) for image in images.permute(0, 2, 3, 1).cpu().numpy()]


def save_png(image: Image.Image, path: Path, crop_to_content: bool = False) -> None:
    """Atomic write (looped_dit/eval/generate.py: save). crop_to_content: the
    official sample.py's `sample.crop(sample.getbbox())` — skipped for an
    all-black image, where getbbox() is None and theirs would crash."""
    if crop_to_content and image.getbbox() is not None:
        image = image.crop(image.getbbox())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


@torch.no_grad()
def generate_corebench_images(
    model,
    image_dir: Path,
    tokenizer,
    device: torch.device,
    protocol: str = "looped_dit",
    rank: int = 0,
    world_size: int = 1,
    samples_per_prompt: int = 4,
    batch_size: int = 16,
    steps: int = 100,
    cfg_scale: float = 6.0,
    seed: int = 0,
    model_name: str = "model",
    prompts: Optional[list[Prompt]] = None,
) -> int:
    """Generate this rank's share (prompts[rank::world_size]) of the images not
    on disk yet, so an interrupted eval resumes. Returns how many were made.

    `model` is any model SamplingPipeline can drive whose conditioning is
    token_conditions / token_mask (e.g. ThinkerFrozenPainterBase around a
    MiniT2IPainter); `tokenizer` is the text encoder's tokenizer."""
    prompts = (prompts if prompts is not None else corebench_prompts())[rank::world_size]
    todo = [
        (p, s) for p in prompts for s in range(samples_per_prompt)
        if not output_path(protocol, image_dir, p, s, model_name).exists()
    ]
    pipeline = SamplingPipeline(num_inference_steps=steps, batch_size=batch_size, predictor=CFGPredictor(cfg_scale))
    sigma = getattr(model.scheduler, "init_noise_sigma", 1.0)
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        tokens = tokenizer(  # looped_dit/pipeline.py: TextEncoder.tokenize
            [p.text for p, _ in chunk], max_length=tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        noise = torch.cat([
            torch.randn((1, *model.noise_shape), device=device,
                        generator=torch.Generator(device=device).manual_seed(seed + p.index * 1000 + s))
            for p, s in chunk
        ]) * sigma
        conditions = DataSample(token_conditions=tokens["input_ids"], token_mask=tokens["attention_mask"]).to(device)
        images = pipeline.sample_one_batch(model, conditions, device, initial_noise=noise)
        for image, (p, s) in zip(to_pil(images), chunk):
            save_png(image, output_path(protocol, image_dir, p, s, model_name), crop_to_content=protocol == "official")
    return len(todo)


def summarize_scores(scores: dict) -> dict[str, float]:
    """eval/corebench_judge.py's scores.json → flat metrics (0-100): the
    protocol's overall score, composition and reasoning means, each dimension."""
    dims = scores["dimensions"]
    metrics = {
        "overall": scores["overall"],
        "composition": mean(dims[t] for t in COMPOSITION),
        "reasoning": mean(dims[t] for t in REASONING),
        "mean_of_dimensions": mean(dims[t] for t in COMPOSITION + REASONING),
    }
    metrics.update({t: dims[t] for t in COMPOSITION + REASONING})
    return metrics
