# ABOUTME: Samples CLEVR scenes steered by the oracle layout (centroid-mask blends) through a screen's oracle translator.
# ABOUTME: Scored as clevr_relation_split.py: the sampling ceiling, and whether a faint layout cue suffices to sample.
"""What does oracle layout steering buy in sampled images, at full and at faint information?

The model is the screen's (clevr_saturation_screen.py: same overrides, its trained oracle translator, the painter
blinded if the overrides blind it). In every painter call of the sampler the translator's input is the oracle's
token grid at a condition <path>:<alpha> (paths and levels as in the screen: where = uni -> anon, binding =
anon -> bound, full = uni -> bound) instead of the thinker's output; the thinker is skipped. CFG's unconditional
pass (objects masked out by null_condition_sample) gets the oracle of an empty scene (all-zero tokens), as a
thinker reading the null condition. Sampling (steps, CFG scale) follows the config's sampling pipeline;
SAMPLING_CFG=<scale> overrides the guidance scale. Each condition is scored by clevr_relation_split's scoring:
spatial, given / implied, scene_all_rel, precision / recall / perfect_gen.

    [N_SCENES=512] [SEED=0] [BATCH=64] [CONDS=full:0,full:0.1,full:0.25,full:1,where:1] [SAMPLING_CFG=s] \\
        python experiments/diagnostics/clevr_oracle_sampling.py <overrides.txt> <painter_ckpt> <translator.pt> <out_dir>

Writes <out_dir>/oracle_sampling.json (summary per condition + per-scene records), results.md, samples_<cond>.png
(first 16) and run_meta.json. Smoke: N_SCENES=4 CONDS=full:1.
"""
import datetime
import json
import os
import platform
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/diagnostics")
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torchvision.utils import save_image

from clevr_relation_split import fmt, score, summarize
from datasets.clevr_dataset import calibrate_mask_projection
from eval.clevr_eval_callbacks import ClevrMetricsCallback, _generate_images, _scene_to_data_sample
from factory import build_model

N_SCENES = int(os.environ.get("N_SCENES", 512))
SEED = int(os.environ.get("SEED", 0))
BATCH = int(os.environ.get("BATCH", 64))
CONDS = [(c.split(":")[0], float(c.split(":")[1]))
         for c in os.environ.get("CONDS", "full:0,full:0.1,full:0.25,full:1,where:1").split(",")]
C = 16
dev = torch.device("cuda")


def levels(s, grid):
    """(uni, anon, bound), each (B, grid², 16): clevr_saturation_screen.levels."""
    m = s.spatial_conditions.float()
    assert m.shape[1:] == (C, 32, 32), m.shape
    bound = F.adaptive_avg_pool2d(m, grid).flatten(2).transpose(1, 2)
    uni = bound.mean(1, keepdim=True).expand_as(bound)
    e, om = s.embedding_conditions.float(), s.embedding_mask.float()
    size, mat = e[..., 2].long(), e[..., 3].long()
    attr = torch.cat([e[..., 7:15], e[..., 4:7], F.one_hot(mat, 2).float(), F.one_hot(size, 2).float()], -1)
    hist = (attr * om[..., None]).sum(1) / om.sum(1, keepdim=True).clamp(min=1)
    anon = torch.cat([bound[..., 15:] * hist[:, None], bound[..., 15:]], -1)
    return uni, anon, bound


def oracle_tokens(s, grid, path, alpha):
    uni, anon, bound = levels(s, grid)
    start, end = {"where": (uni, anon), "binding": (anon, bound), "full": (uni, bound)}[path]
    x = start + alpha * (end - start)
    null = s.embedding_mask.float().sum(1) == 0  # CFG's unconditional pass
    return torch.where(null[:, None, None], torch.zeros_like(x), x)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    ov_file, painter_ckpt, tr_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, start = time.time(), datetime.datetime.now().isoformat(timespec="seconds")
    ov = [line.strip() for line in open(ov_file)
          if line.strip() and not line.startswith(("RUN=", "painter.checkpoint=", "eval"))]
    ov += [f"painter.checkpoint={painter_ckpt}", "run.wandb_project=null", f"eval.batch_size={BATCH}"]
    if os.environ.get("SAMPLING_CFG"):
        ov.append(f"sampling.predictor.scale={os.environ['SAMPLING_CFG']}")
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=ov)
    cfg.eval_callbacks = None
    model = build_model(cfg, instantiate(cfg.diffusion))
    tr = model.thinker_painter_translator
    tr.load_state_dict(torch.load(tr_path, map_location="cpu", weights_only=True))
    model = model.to(dev).eval()
    assert model.thinker.vocab_size == C
    grid = tr._grid
    pipe = model.sampling_pipeline
    print(f"oracle translator {tr_path}; blinded {bool(model.train_cfg.force_unconditional_painter)}; "
          f"{pipe.num_inference_steps} steps, CFG {getattr(pipe.predictor, 'scale', None)}, batch {pipe.batch_size}",
          flush=True)

    COND = {"calls": 0}
    run_painter = model.run_painter

    def oracle_run_painter(s, logits):
        x = oracle_tokens(s, grid, *COND["c"])
        if COND["calls"] < 2:  # the first conditional and unconditional pass
            print(f"painter call {COND['calls']}: {int((s.embedding_mask.sum(1) == 0).sum())} of {x.shape[0]} "
                  f"samples get the empty-scene oracle; token norm {x.norm(dim=-1).mean():.3f}", flush=True)
        COND["calls"] += 1
        assert x.shape == (logits.shape[0], grid * grid, C), (x.shape, logits.shape)
        return run_painter(s, x)

    model.run_painter = oracle_run_painter
    model.forward = partial(model.forward, null_steering=True)  # the thinker's output is replaced anyway

    cb = ClevrMetricsCallback(root_dir=cfg.data.clevr_root, mode=cfg.data.mode)
    cb._load(dev)
    H_inv = calibrate_mask_projection(cb._scenes)
    scenes = random.Random(SEED).sample(cb._scenes, min(N_SCENES, len(cb._scenes)))
    random.seed(SEED)
    np.random.seed(SEED)
    cond_samples = [_scene_to_data_sample(s, cfg.data.mode, include_centroid_mask=True, mask_size=32, H_inv=H_inv)
                    for s in scenes]
    print(f"first scene: {len(scenes[0]['objects'])} objects, mask {tuple(cond_samples[0].spatial_conditions.shape)}",
          flush=True)

    results = {}
    for path, alpha in CONDS:
        name = f"{path}:{alpha:g}"
        COND["c"] = (path, alpha)
        torch.manual_seed(SEED)
        imgs = _generate_images(model, cond_samples, dev)
        save_image(imgs[:16].float().cpu(), out_dir / f"samples_{path}_{alpha:g}.png", nrow=4)
        records = score(cond_samples, scenes, imgs, cb)
        results[name] = {"summary": summarize(records), "records": records}
        s = results[name]["summary"]
        print(f"{name}: spatial {s['spatial']:.3f} given {fmt(s['given'])} scene_all_rel {s['scene_all_rel']:.3f} "
              f"recall {s['recall']:.3f} perfect_gen {s['perfect_gen']:.3f}", flush=True)

    meta = {"command": " ".join([f"N_SCENES={N_SCENES} SEED={SEED} BATCH={BATCH} "
                                 f"CONDS={os.environ.get('CONDS', '')} SAMPLING_CFG={os.environ.get('SAMPLING_CFG', '')}",
                                 "python", *sys.argv]),
            "overrides": ov, "seed": SEED, "git_sha": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", ".")),
            "hardware": f"{torch.cuda.get_device_name(dev)} ({platform.node()})", "torch": torch.__version__,
            "start": start, "end": datetime.datetime.now().isoformat(timespec="seconds"), "wall_clock_s": time.time() - t0}
    (out_dir / "oracle_sampling.json").write_text(json.dumps(results))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    L = ["# CLEVR sampling under oracle layout steering", "",
         f"Translator `{tr_path}`, {len(scenes)} val scenes (seed {SEED}).  ", f"Command: `{meta['command']}`  ",
         f"Git {meta['git_sha']} (dirty: {meta['git_dirty']}), {meta['hardware']}, {meta['wall_clock_s']:.0f} s. "
         "Definitions: the script's docstring.", "",
         "| condition | spatial | given | implied | scene all-rel | precision | recall | perfect_gen |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        s = r["summary"]
        L.append(f"| {name} | {s['spatial']:.3f} | {fmt(s['given'])} | {fmt(s['implied'])} | {s['scene_all_rel']:.3f}"
                 f" | {s['precision']:.3f} | {s['recall']:.3f} | {s['perfect_gen']:.3f} |")
    (out_dir / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
