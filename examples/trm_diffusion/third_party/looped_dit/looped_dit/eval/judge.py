"""PRISM-Bench and T2I-CoReBench scoring with a local VLM judge served by vLLM.

    python -m looped_dit.eval.judge --benchmark prism --data eval_assets/prism/captions/en \
        --prism-script eval_assets/prism/eval_qwen25.py --image-dir OUT/prism_images --out-dir OUT/prism_results

PRISM-Bench follows the official Qwen2.5-VL-72B protocol: the rubric templates are
read verbatim from the benchmark's own evaluation/eval_qwen25.py, the image comes
first with the official pixel window, greedy decoding, repetition penalty 1.05.
T2I-CoReBench asks the judge each checklist question as a yes/no question.
Judgments are cached in results.jsonl, so an interrupted run resumes.
Needs `vllm` and `qwen-vl-utils`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .benchmarks import Prompt, image_path, load_prompts

COREBENCH_SYSTEM = (
    "You are an AI quality auditor for text-to-image generation. Answer the yes/no question using only clear "
    "visual evidence in the image. The original prompt is context, not evidence. Ambiguity, missing details, and "
    "partially visible evidence count as no. For conditional questions, answer yes only when both the condition "
    "and the main clause are visibly true. Return exactly one word: yes or no."
)
# Qwen's chat template inserts this system message when none is given, which is
# what the official PRISM script does.
PRISM_SYSTEM = "You are a helpful assistant."
PRISM_PIXELS = (("min_pixels", 256 * 28 * 28), ("max_pixels", 1280 * 28 * 28))
PRISM_TEMPLATE = {"affection": "1", "composition": "2", "entity": "3", "imagination": "4",
                  "style": "5", "text_rendering": "6", "long_text": "7", "aesthetic": "8"}
# sha256 of evaluation/eval_qwen25.py in github.com/rongyaofang/prism-bench used for the paper.
PRISM_SCRIPT_SHA256 = "d56136f991f37b0b6621bdc5a1f211c2beb63fa4721b312041552c53523572b3"

DEFAULTS = {  # judge model and decoding budget per benchmark
    "prism": dict(judge_model="Qwen/Qwen2.5-VL-72B-Instruct", samples_per_prompt=1, batch_size=64,
                  max_model_len=16384, max_new_tokens=1024, max_failed=8),
    "corebench": dict(judge_model="Qwen/Qwen3-VL-32B-Thinking", samples_per_prompt=4, batch_size=256,
                      max_model_len=32768, max_new_tokens=4096, max_failed=64),
}


@dataclass(frozen=True)
class Request:
    prompt: Prompt
    sample: int
    metric: str
    detail: int
    image: Path
    system: str
    text: str
    max_score: float
    image_first: bool = False
    vision_id: bool = False
    image_options: tuple = ()

    @property
    def key(self) -> str:
        return f"{self.prompt.item_id}|{self.sample}|{self.metric}|{self.detail}"


class VLLMJudge:
    def __init__(self, model: str, tensor_parallel_size: int, max_model_len: int,
                 gpu_memory_utilization: float = 0.9, repetition_penalty: float = 1.0):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        options: dict[str, Any] = {}
        if "qwen3" in Path(model).name.lower():
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            options.update(dtype="bfloat16", distributed_executor_backend="mp")
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 8, "video": 0},
            mm_encoder_tp_mode="data",
            **options,
        )
        self.sampling_params = SamplingParams
        self.repetition_penalty = repetition_penalty

    def _input(self, request: Request) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        image = {"type": "image", "image": str(request.image.resolve()), **dict(request.image_options)}
        text = {"type": "text", "text": request.text}
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": [image, text] if request.image_first else [text, image]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=request.vision_id
        )
        images, _ = process_vision_info(messages)
        return {"prompt": prompt, "multi_modal_data": {"image": images}}

    def generate(self, requests: list[Request], max_new_tokens: int, temperature: float = 0.0, n: int = 1) -> list[list[str]]:
        params = self.sampling_params(
            temperature=temperature, repetition_penalty=self.repetition_penalty, max_tokens=max_new_tokens, n=n
        )
        outputs = self.llm.generate([self._input(r) for r in requests], sampling_params=params, use_tqdm=True)
        return [[candidate.text.strip() for candidate in output.outputs] for output in outputs]


def strip_thinking(text: str) -> str:
    if "<think>" in text and "</think>" not in text:
        raise ValueError("response ended inside an unfinished thinking block")
    return re.sub(r"<think>.*?</think>", "", text.split("</think>")[-1], flags=re.DOTALL).strip()


def extract_score(text: str, max_score: float) -> float:
    cleaned = strip_thinking(text)
    try:
        value = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)).get("score")
        if isinstance(value, (int, float)):
            return min(max(float(value), 0.0), max_score)
    except (json.JSONDecodeError, AttributeError):
        pass
    if max_score == 1:
        match = re.search(r"\bscore\s*[:=]\s*([01])\b", cleaned, re.IGNORECASE)
        if match:
            return float(match.group(1))
        answers = re.findall(r"\b(yes|no)\b", cleaned, re.IGNORECASE)
        if answers:
            return 1.0 if answers[-1].lower() == "yes" else 0.0
    match = re.search(r"score[\s*]*[:=]?\s*(-?\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    numbers = [match.group(1)] if match else re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if len(numbers) != 1:
        raise ValueError(f"could not parse a score from {text[:200]!r}")
    return min(max(float(numbers[0]), 0.0), max_score)


def write_jsonl(path: Path, rows) -> None:
    """Write through a temporary file, so an interrupted run never leaves a truncated cache."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def load_prism_templates(script: str | Path) -> dict[str, str]:
    source = Path(script).read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != PRISM_SCRIPT_SHA256:
        print(f"[judge] warning: {script} differs from the eval_qwen25.py used for the paper", flush=True)
    templates = dict(re.findall(r'messages_(\d)\s*=\s*"""(.*?)"""', source, re.DOTALL))
    if sorted(templates) != list("12345678"):
        raise ValueError(f"{script} does not define the eight PRISM rubric templates messages_1 .. messages_8")
    return templates


def requests_for(benchmark: str, prompt: Prompt, sample: int, image: Path, templates: dict[str, str]) -> list[Request]:
    if benchmark == "corebench":
        return [
            Request(prompt, sample, "checklist", i, image, COREBENCH_SYSTEM,
                    f'Original prompt: "{prompt.text}"\nQuestion: "{item["question"]}"', 1)
            for i, item in enumerate(prompt.metadata["Checklist"])
        ]
    return [
        Request(prompt, sample, metric, 0, image, PRISM_SYSTEM,
                templates[PRISM_TEMPLATE[prompt.group if metric == "alignment" else metric]].format(text_prompt=prompt.text),
                10, image_first=True, vision_id=True, image_options=PRISM_PIXELS)
        for metric in ("alignment", "aesthetic")
    ]


def summarize(benchmark: str, records: list[dict]) -> dict[str, Any]:
    """Scores on a 0-100 scale: judgments are averaged per image and metric,
    then over images (for PRISM: the mean of alignment and aesthetic)."""
    per_image: dict[tuple, list[float]] = {}
    for r in records:
        per_image.setdefault((r["group"], r["item_id"], r["sample"], r["metric"]), []).append(100 * r["score"] / r["max_score"])
    scores = {key: mean(values) for key, values in per_image.items()}

    def mean_by(field: int) -> dict[str, float]:
        names = sorted({key[field] for key in scores})
        return {name: mean(s for key, s in scores.items() if key[field] == name) for name in names}

    return {"benchmark": benchmark, "overall": mean(scores.values()), "metrics": mean_by(3), "groups": mean_by(0),
            "num_images": len({key[:3] for key in scores}), "num_judgments": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score PRISM-Bench or T2I-CoReBench images with a VLM judge.")
    parser.add_argument("--benchmark", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prism-script", help="evaluation/eval_qwen25.py from the PRISM-Bench repository")
    parser.add_argument("--judge-model")
    parser.add_argument("--samples-per-prompt", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-failed", type=int, help="unparseable judgments tolerated in the summary")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    args = parser.parse_args()
    for key, value in DEFAULTS[args.benchmark].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    templates = {}
    if args.benchmark == "prism":
        if not args.prism_script:
            parser.error("--prism-script is required for PRISM-Bench")
        templates = load_prism_templates(args.prism_script)

    prompts = load_prompts(args.benchmark, args.data)
    requests, missing = [], 0
    for prompt in prompts:
        for sample in range(args.samples_per_prompt):
            image = image_path(args.image_dir, prompt, sample)
            if image.exists():
                requests += requests_for(args.benchmark, prompt, sample, image, templates)
            else:
                missing += 1
    if missing:
        raise FileNotFoundError(f"{missing} images are missing under {args.image_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    records = {}
    if results_path.exists():
        records = {r["key"]: r for r in map(json.loads, results_path.read_text(encoding="utf-8").splitlines())}
    pending = [r for r in requests if r.key not in records]
    print(f"[judge] {args.benchmark}: {len(requests)} judgments, {len(pending)} to run", flush=True)
    if pending:
        judge = VLLMJudge(args.judge_model, args.tensor_parallel_size, args.max_model_len,
                          repetition_penalty=1.05 if args.benchmark == "prism" else 1.0)
        for attempt in range(args.retries):  # unparseable answers are retried with a larger token budget
            max_new_tokens = min(args.max_new_tokens * 2**attempt, args.max_model_len // 2)
            retry = []
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                for request, (response,) in zip(batch, judge.generate(batch, max_new_tokens)):
                    try:
                        score = extract_score(response, request.max_score)
                    except ValueError:
                        retry.append(request)
                        continue
                    records[request.key] = {"key": request.key, "item_id": request.prompt.item_id, "group": request.prompt.group,
                                            "sample": request.sample, "metric": request.metric, "score": score,
                                            "max_score": request.max_score, "response": response}
                write_jsonl(results_path, records.values())
            pending = retry
            if not pending:
                break

    failed = [r.key for r in requests if r.key not in records]
    if len(failed) > args.max_failed:
        raise RuntimeError(f"{len(failed)} judgments could not be parsed (tolerance {args.max_failed}), e.g. {failed[:3]}")
    summary = summarize(args.benchmark, [records[r.key] for r in requests if r.key in records])
    summary["failed_judgments"] = len(failed)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
