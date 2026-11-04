"""TEMPORARY (Claude): do the loop-trained models (plain TRM, PaTh thinker) handle mazes with loops better than the originals?

Test sets: runs/claude_loop_data/maze/square/n{N}_{kind}_test.parquet, kind in none / side (loop off the solution,
"lollipop") / shortcut / detour_longer / detour_shorter; GT = the shortest path.
Models:
  PaTh (env PATH_CKPTS=name:thinker_ckpt,...; same frozen painter from the config): official sampling
       (model.sampling_pipeline, 1 attempt), scored by the Amaze evaluator (eval/amaze_eval.py): pass (drawn cell set ==
       GT), gt_cell_coverage, background_violation (= fraction of drawn cells that are not on the GT path: kept loops);
       plus token-level numbers from the drawn image (token >= 10 % blue = path) for comparison with plain
  plain (env PLAIN_CKPTS=name:ckpt,...): own head after n_sup steps vs the training token labels
Token-level per model: exact boards (every known token right), extra = off tokens marked path / off tokens,
missed = path tokens marked off / path tokens.
Output: <out_dir>/eval_loops.pt, printed table.
"""
import logging
import os
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

from datasets.amaze_dataset import AmazeDataset
from eval.amaze_eval import AmazeMetrics
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from interp_template import CONFIG_DIR, G, TOK, blue, ints

KINDS = ("none", "side", "shortcut", "detour_shorter", "detour_longer")


def tok_scores(pred, lab):
    """pred (B, T) bool, lab (B, T) in {1, 0, -1}."""
    k = lab >= 0
    exact = ((pred == (lab == 1)) | ~k).all(1).float()
    extra = (pred & (lab == 0)).sum() / (lab == 0).sum().clamp_min(1)
    missed = (~pred & (lab == 1)).sum() / (lab == 1).sum().clamp_min(1)
    return dict(tok_exact=float(exact.mean()), tok_extra=float(extra), tok_missed=float(missed))


def parse(env):
    return [s.split(":", 1) for s in os.environ.get(env, "").split(",") if s]


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    sizes = ints(ic.get("maze_sizes", "8,13"))
    kinds = [k for k in str(ic.get("kinds", ",".join(KINDS))).split(",") if k]
    data_tmpl = str(ic.get("data_tmpl", "runs/claude_loop_data/maze/square/n{N}_{K}_test.parquet"))
    n_boards, bs, seed = int(ic.get("num_boards", 200)), int(ic.get("batch_size", 100)), int(ic.get("seed", 0))
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    sets = {}
    for N in sizes:
        for K in kinds:
            ds = AmazeDataset(dataset_path=data_tmpl.format(N=N, K=K), split="test", image_size=144,
                              condition_field="m_original_img", target_field="sol_img", num_channels=3,
                              include_metadata=True)
            sets[(N, K)] = (ds, min(n_boards, len(ds)))

    res = {}                                                     # (model, N, kind) -> metrics

    from train_plain_trm import PlainTRM
    from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation
    for name, path in parse("PLAIN_CKPTS"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        pm = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16)), noisy=bool(ck["args"].get("noisy", False)))
        pm.load_state_dict(ck["ema"]); pm = pm.to(device).eval()
        logging.info(f"[plain {name}] {path}: step {ck['step']}")
        for (N, K), (ds, nb) in sets.items():
            preds, labs = [], []
            for k0 in range(0, nb, bs):
                b = AmazeDataset.collate_fn([ds[i] for i in range(k0, min(nb, k0 + bs))])
                labs.append(ZPathEncodingAblation._token_labels(b.images, G, 0.1).long())
                with torch.no_grad():
                    preds.append((pm.think(b.spatial_conditions.to(device)).float().argmax(-1) == 1).cpu())
            res[(f"plain {name}", N, K)] = dict(n=nb, **tok_scores(torch.cat(preds), torch.cat(labs)))
            print(f"plain {name:8s} n{N:<2d} {K:15s} {res[(f'plain {name}', N, K)]}", flush=True)
        del pm

    for name, path in parse("PATH_CKPTS"):
        model = build_model(cfg, instantiate(cfg.diffusion))
        load_checkpoint(model, path, use_ema=cfg.get("use_ema", True), device="cpu")
        model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
        logging.info(f"[PaTh {name}] {path}")
        for (N, K), (ds, nb) in sets.items():
            ev = AmazeMetrics(task="maze")
            rows, preds, labs = [], [], []
            for k0 in range(0, nb, bs):
                samples = [ds[i] for i in range(k0, min(nb, k0 + bs))]
                batch = ds.collate_fn(samples)
                cond = model._batch_to_sample(batch, device)
                with torch.no_grad():
                    gen = model.sampling_pipeline.sample_one_batch(
                        model, cond, device, generator=torch.Generator(device=device).manual_seed(seed + k0))
                    imgs = model.decode_for_eval(gen).float().cpu().clamp(0, 1)
                rows += [ev._compute_maze_metrics(im, s.metadata) for im, s in zip(imgs, samples)]
                dm = blue(imgs)
                preds.append(dm.reshape(len(imgs), G, TOK, G, TOK).mean((2, 4)).reshape(len(imgs), -1) >= 0.1)
                labs.append(ZPathEncodingAblation._token_labels(batch.images, G, 0.1).long())
            m = {k: sum(r[k] for r in rows) / len(rows) for k in ("pass", "gt_cell_coverage", "background_violation")}
            m["any_extra"] = sum(r["background_violation"] > 0 for r in rows) / len(rows)
            res[(f"PaTh {name}", N, K)] = dict(n=nb, **m, **tok_scores(torch.cat(preds), torch.cat(labs)))
            print(f"PaTh  {name:8s} n{N:<2d} {K:15s} {res[(f'PaTh {name}', N, K)]}", flush=True)
            torch.save(res, out_dir / "eval_loops.pt")
        del model

    torch.save(res, out_dir / "eval_loops.pt")
    models = list(dict.fromkeys(m for m, _, _ in res))
    for N in sizes:
        print(f"\n=== n{N}: exact boards (PaTh: evaluator pass | token-level exact); kept-extra = PaTh: share of "
              f"boards with any drawn non-path cell, plain: off tokens marked path")
        print(f"   {'model':16s} " + " ".join(f"{k:>17s}" for k in kinds))
        for mo in models:
            cells = []
            for K in kinds:
                r = res.get((mo, N, K))
                if r is None:
                    cells.append(f"{'-':>17s}")
                elif mo.startswith("PaTh"):
                    cells.append(f"{r['pass']:5.1%}|{r['tok_exact']:5.1%} x{r['any_extra']:4.0%}")
                else:
                    cells.append(f"{r['tok_exact']:9.1%} x{r['tok_extra']:5.1%}")
            print(f"   {mo:16s} " + " ".join(cells))
    print(f"saved {out_dir}/eval_loops.pt")


if __name__ == "__main__":
    main()
