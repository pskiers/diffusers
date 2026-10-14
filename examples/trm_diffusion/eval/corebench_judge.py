"""
eval/corebench_judge.py — score generated T2I-CoReBench images with a VLM judge,
running one of the two reference implementations unchanged. Runs in the
separate judge environment (vLLM + qwen-vl-utils, scripts/setup_judge_env.sh):
Qwen3-VL needs newer vLLM/transformers than the training environment has.

Protocols (eval/corebench_eval.py writes the matching image layout):
  looped_dit (default) — third_party/looped_dit/looped_dit/eval/judge.py, the
      protocol behind Looped-DiT's reported CoReBench numbers
      (judge Qwen/Qwen3-VL-32B-Thinking).
      Images: <image-dir>/<task>/<item_id>/<sample:05d>.png
  official — third_party/t2i_corebench/evaluate.py, the benchmark authors' code
      (judge key as in their evaluate.py, default Qwen3_VL_30B_A3B_Thinking).
      Images: <image-dir>/<model-name>/<task>/<item_id>-<sample>.png

The one thing changed in either is the share of GPU memory vLLM may take
(--gpu-memory-utilization): during training the GPUs also hold the paused
training processes. It does not affect any judgment.

Writes <out-dir>/scores.json:
    {"protocol", "judge_model", "dimensions": {task: score 0-100}, "overall": score 0-100, ...}
"overall" is each protocol's own: looped_dit averages all images, official
(whose code stops at per-dimension scores) the 12 dimensions, as in its paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# vLLM's FlashInfer sampler is JIT-compiled at startup with the system nvcc,
# which fails when that nvcc is older than vLLM's CUDA. Both protocols decode
# greedily (temperature 0), which never uses it, so vLLM's own sampler gives
# the same judgments without needing a compiler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

COREBENCH_DATA = ROOT / "third_party" / "t2i_corebench" / "data"
TASKS = ("C-MI", "C-MA", "C-MR", "C-TR", "R-LR", "R-BR", "R-HR", "R-PR", "R-GR", "R-AR", "R-CR", "R-RR")

# third_party/t2i_corebench/evaluate.py, its `__main__` block (not importable).
OFFICIAL_MLLMS = {
    "Qwen2_5_VL_72B_Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen3_VL_8B_Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3_VL_8B_Thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen3_VL_32B_Instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen3_VL_32B_Thinking": "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen3_VL_30B_A3B_Instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen3_VL_30B_A3B_Thinking": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen3_VL_235B_A22B_Instruct": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen3_VL_235B_A22B_Thinking": "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "Qwen3_5_9B": "Qwen/Qwen3.5-9B",
    "Qwen3_5_27B": "Qwen/Qwen3.5-27B",
    "Qwen3_5_35B_A3B": "Qwen/Qwen3.5-35B-A3B",
    "Gemini_2_5_Flash": ["gemini-2.5-flash", os.getenv("GEMINI_API_KEY")],
}


def run_looped_dit(args) -> dict:
    from third_party.looped_dit.looped_dit.eval import judge

    class Judge(judge.VLLMJudge):
        def __init__(self, model, tensor_parallel_size, max_model_len, gpu_memory_utilization=0.9, repetition_penalty=1.0):
            super().__init__(model, tensor_parallel_size, max_model_len, args.gpu_memory_utilization, repetition_penalty)

    judge.VLLMJudge = Judge
    argv = ["judge", "--benchmark", "corebench", "--data", str(args.data), "--image-dir", str(args.image_dir),
            "--out-dir", str(args.out_dir), "--samples-per-prompt", str(args.samples_per_prompt),
            "--tensor-parallel-size", str(args.tensor_parallel_size)]
    for flag in ("judge_model", "batch_size", "max_model_len", "max_new_tokens", "max_failed"):
        if getattr(args, flag) is not None:
            argv += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
    sys.argv = argv
    judge.main()
    summary = json.loads((args.out_dir / "summary.json").read_text())
    return {"judge_model": args.judge_model or judge.DEFAULTS["corebench"]["judge_model"],
            "dimensions": summary["groups"], "overall": summary["overall"],
            "failed_judgments": summary["failed_judgments"], "num_images": summary["num_images"]}


def run_official(args) -> dict:
    official = ROOT / "third_party" / "t2i_corebench"
    sys.path.insert(0, str(official))
    os.chdir(official)  # evaluate.py reads data/<task>.json relative to the working directory
    import evaluate

    real_llm = evaluate.LLM
    evaluate.LLM = lambda **kwargs: real_llm(**{**kwargs, "gpu_memory_utilization": args.gpu_memory_utilization})
    mllm = args.mllm
    ns = argparse.Namespace(model=args.model_name, mllm=mllm, gen_eval_file=", ".join(TASKS),
                            output_path=str(args.image_dir), update=False, seed=0)
    evaluate.seed_everything(ns.seed)
    if "Qwen" in mllm:
        evaluate.start_evaluation_qwen(ns, OFFICIAL_MLLMS[mllm])
    if "Gemini" in mllm:
        evaluate.start_evaluation_gemini(ns, OFFICIAL_MLLMS[mllm])
    dims = {}
    for task in TASKS:
        result = json.loads((args.image_dir / args.model_name / f"{task}-{mllm}.json").read_text())
        dims[task] = 100 * result["mean_score"]
        (args.out_dir / f"{task}-{mllm}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return {"judge_model": OFFICIAL_MLLMS[mllm] if "Qwen" in mllm else mllm,
            "dimensions": dims, "overall": sum(dims.values()) / len(dims)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--protocol", choices=("looped_dit", "official"), default="looped_dit")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # looped_dit
    parser.add_argument("--data", type=Path, default=COREBENCH_DATA, help="looped_dit: prompt files (e.g. a subset, for tests)")
    parser.add_argument("--judge-model", help="looped_dit: HF id (default Qwen/Qwen3-VL-32B-Thinking)")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-failed", type=int)
    # official (tensor parallelism = all GPUs in CUDA_VISIBLE_DEVICES, as in their code)
    parser.add_argument("--mllm", default="Qwen3_VL_30B_A3B_Thinking", choices=sorted(OFFICIAL_MLLMS))
    parser.add_argument("--model-name", default="model", help="official: image sub-directory name")
    args = parser.parse_args()
    args.image_dir, args.out_dir = args.image_dir.resolve(), args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores = run_looped_dit(args) if args.protocol == "looped_dit" else run_official(args)
    scores = {"protocol": args.protocol, **scores}
    (args.out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
