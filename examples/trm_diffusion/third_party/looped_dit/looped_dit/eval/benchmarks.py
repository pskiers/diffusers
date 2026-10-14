"""Benchmark prompt sets and the layout of their generated images."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BENCHMARKS = ("geneval", "dpg", "prism", "corebench", "spatial_geneval", "tiif")

PRISM_TRACKS = ("affection", "composition", "entity", "imagination", "long_text", "style", "text_rendering")
COREBENCH_TASKS = ("C-MA", "C-MI", "C-MR", "C-TR", "R-AR", "R-BR", "R-CR", "R-GR", "R-HR", "R-LR", "R-PR", "R-RR")
TIIF_TYPES = (
    "2d_spatial_relation", "3d_spatial_relation", "action+2d", "action+3d", "action+color", "action+texture",
    "color+2d", "color+3d", "color+texture", "comparison", "comparison+2d", "comparison+3d", "comparison+color",
    "comparison+texture", "differentiation", "differentiation+2d", "differentiation+3d", "differentiation+color",
    "differentiation+texture", "negation", "negation+2d", "negation+3d", "negation+color", "negation+texture",
    "numeracy", "numeracy+2d", "numeracy+3d", "numeracy+color", "numeracy+texture", "real_world", "shape+2d",
    "shape+3d", "shape+color", "shape+texture", "style", "text", "texture+2d", "texture+3d", "texture+color",
)


@dataclass(frozen=True)
class Prompt:
    index: int
    item_id: str
    text: str
    group: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dpg_rows(source: str) -> list[dict]:
    """DPG-Bench question rows: a local .jsonl/.json/.parquet/.csv file or the
    Hugging Face dataset id (Jialuo21/DPG-Bench)."""
    path = Path(source)
    if path.exists():
        import pandas as pd

        if path.suffix == ".csv":
            rows = pd.read_csv(path).to_dict("records")
        elif path.suffix == ".parquet":
            rows = pd.read_parquet(path).to_dict("records")
        elif path.suffix == ".jsonl":
            rows = read_jsonl(path)
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        from datasets import load_dataset

        rows = list(load_dataset(source, split="test"))
    if rows and "questions" in rows[0]:  # one row per prompt, with a list of questions
        rows = [
            {
                "item_id": row["item_id"],
                "text": row["prompt"],
                "proposition_id": int(q["proposition_id"]),
                "dependency": str(q.get("dependency", "0")),
                "category_broad": q.get("category_broad", ""),
                "category_detailed": q.get("category_detailed", ""),
                "question_natural_language": q["question"],
            }
            for row in rows
            for q in row["questions"]
        ]
    return rows


def load_prompts(benchmark: str, data: str) -> list[Prompt]:
    path = Path(data)
    if benchmark == "geneval":
        return [Prompt(i, f"{i:05d}", row["prompt"], metadata=row) for i, row in enumerate(read_jsonl(path))]
    if benchmark == "dpg":
        prompts: dict[str, str] = {}
        for row in load_dpg_rows(data):
            prompts.setdefault(row["item_id"], row["text"])
        return [Prompt(i, item_id, text) for i, (item_id, text) in enumerate(prompts.items())]
    items: list[tuple[str, str, str, dict]] = []
    if benchmark == "prism":
        for track in PRISM_TRACKS:
            for j, row in enumerate(read_jsonl(path / f"{track}.jsonl")):
                items.append((f"{track}-{j:03d}", row["prompt"], track, row))
    elif benchmark == "corebench":
        for task in COREBENCH_TASKS:
            rows = json.loads((path / f"{task}.json").read_text(encoding="utf-8"))
            items += [(item_id, row["Prompt"], task, row) for item_id, row in rows.items()]
    elif benchmark == "spatial_geneval":
        items = [(str(row["id"]), row["prompt"], row["scene"], row) for row in read_jsonl(path)]
    elif benchmark == "tiif":  # the short prompts, with the yes/no questions of their evaluation file
        for kind in TIIF_TYPES:
            prompts = read_jsonl(path / "test_prompts" / f"{kind}_prompts.jsonl")
            questions = read_jsonl(path / "test_eval_prompts" / f"{kind}_eval_prompts.jsonl")
            if len(prompts) != len(questions):
                raise ValueError(f"TIIF-Bench {kind}: {len(prompts)} prompts but {len(questions)} question sets")
            items += [(f"{kind}-{j:04d}", p["short_description"], kind, q) for j, (p, q) in enumerate(zip(prompts, questions))]
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}; expected one of {BENCHMARKS}")
    return [Prompt(i, item_id, text, group, row) for i, (item_id, text, group, row) in enumerate(items)]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-.") or "item"


def image_path(image_dir: str | Path, prompt: Prompt, sample: int) -> Path:
    """Where sample `sample` of `prompt` is stored (PRISM, CoReBench, SpatialGenEval).

    GenEval uses its own layout (<id>/samples/<n>.png) and DPG-Bench stores the
    samples of a prompt as one 2x2 grid (<id>.png)."""
    return Path(image_dir) / _safe(prompt.group) / _safe(prompt.item_id) / f"{sample:05d}.png"
