"""PaTh rebuttal, "is diffusion needed for single-solution boards": deterministic one-pass baselines,
trained and then scored with the official Amaze evaluator in the same job.

  --model trm_unet : PaTh's architecture with a maze-only condition encoder (variant C's: no x_t) and a TRAINABLE UNet
                     painter (models.painter_base.JointControlNetUNetPainter, the amaze_unet backbone = PaTh's painter
                     architecture, random init) steered by the same ControlNet translator. No diffusion: the UNet's
                     input "x_noisy" is the unsolved maze, timestep fixed at 0 (translator without timestep embedding);
                     output = the solved image. Trained exactly like the PaTh thinker otherwise: n_sup=16 deep-supervision
                     steps per batch with the carry passed on (detached), one optimizer step per reasoning step, the
                     model's own build_optimizers (AdamATan2 for the thinker; AdamW for encoder / translator / UNet with
                     the joint painter's optim_cfg), loss / (micro_batch * n_micro) and separate grad clipping of the
                     thinker and the encoder parameters, as ThinkerFrozenPainterBase.train_step does.
  --model unet     : a plain UNet2DModel (DownBlock2D/UpBlock2D, channels --unet_channels; default 64,128,192 = 9.7 M
                     parameters, about PaTh's total of 8.7 M trainable + 1.5 M painter): unsolved maze in, solved image
                     out, timestep 0, one optimizer step per batch. Optimised like the PaTh painter (AdamW betas
                     0.9/0.999, lr 1e-4, weight decay 0.01, 2000 warm-up steps, constant; loss / (micro_batch * n_micro);
                     grad clip 1).
Data: the full maze training set (maze_dataset_train_144.parquet), batch 768 (4 micro-batches of 192), EMA 0.999,
40k optimizer steps. Scoring: the official sample_and_score + build_maze_result, EMA weights, one attempt per maze
(deterministic -> exact5 = exact1), all shapes x MAZE_SCALES plus the OOD sizes (MAZE_OOD_SCALES, 10 in the jobs).
Resumable (<out_dir>/checkpoint_last.pt). wandb (--wandb_project): group diffusion_needed, run diffneed_<model>; training
curves, the quick square-8x8 check and the final tables land in the same run.

python <this> --model trm_unet|unet --data_root <.../data/amaze> --out_dir <dir> [--steps 40000] [--max_minutes 1200]
"""
import argparse
import copy
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from datasets.amaze_dataset import AmazeDataset
from eval.amaze_eval import MAZE_GEOMETRIES, MAZE_OOD_SCALES, MAZE_SCALES, aggregate, build_maze_result, log_tables
from experiments.amaze_generate_and_calculate_metrics import _build_amaze_dataset, sample_and_score
from factory import build_model
from models.optim_utils import ScheduledOptimizer, apply_lr_and_step
from models.painter_base import PainterBase

ENC_C = ["condition_encoder._target_=models.condition_encoders.SpatialConditionEncoder",   # variant C: maze only
         "condition_encoder.in_channels=3", "condition_encoder.with_timestep_emb=false",
         "~condition_encoder.noisy_dropout_p_max", "~condition_encoder.num_train_timesteps"]


def compose_cfg(data_root):
    with initialize_config_dir(config_dir=str(Path.cwd() / "configs"), version_base=None):
        return compose(config_name="config", overrides=[
            "experiment=amaze_thinker_v2_controlnet", f"data.amaze_root={data_root}/maze",
            "painter=mnist_unet_controlnet_joint", "painter.painter_dtype=bfloat16",   # trainable, random-init UNet painter
            "translator.with_timestep_emb=false", "train.cfg_prob=0.0", *ENC_C])


class TRMUNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.m = build_model(cfg, instantiate(cfg.diffusion))
        assert all(p.requires_grad for p in self.m.painter.unet.parameters()), "painter UNet must be trainable"

    def det_sample(self, maze, images=None):
        from datasets.data_sample import DataSample
        return DataSample(spatial_conditions=maze, x_noisy=maze, target=images,
                          timesteps=torch.zeros(maze.shape[0], dtype=torch.long, device=maze.device))

    def init_carry(self, n, device):
        z_H, z_L = self.m.get_initial_states(n)
        return z_H.to(device), z_L.to(device)

    def step(self, maze, carry):
        pred, _, z_H, z_L = self.m.reasoning_step(self.det_sample(maze), *carry)
        return pred.float(), (z_H, z_L)

    def build_optimizers(self, steps):
        # PaTh's own: AdamATan2 for the thinker + AdamW (joint painter optim_cfg) for encoder/translator/UNet.
        return self.m.build_optimizers(1, steps)

    def clip(self):
        # As ThinkerFrozenPainterBase.train_step: thinker and encoder (+translator) clipped separately.
        torch.nn.utils.clip_grad_norm_(self.m.get_thinker_params(), 1.0)
        enc = self.m.get_encoder_params()
        if enc:
            torch.nn.utils.clip_grad_norm_(enc, 1.0)

    @torch.no_grad()
    def predict(self, maze, n_sup):
        carry = self.init_carry(maze.shape[0], maze.device)
        for _ in range(n_sup):
            pred, carry = self.step(maze, carry)
        return pred


class PlainUNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        from diffusers import UNet2DModel
        self.unet = UNet2DModel(sample_size=144, in_channels=3, out_channels=3, block_out_channels=channels,
                                down_block_types=["DownBlock2D"] * len(channels), up_block_types=["UpBlock2D"] * len(channels),
                                norm_num_groups=32, layers_per_block=2)

    def forward(self, maze):
        return self.unet(maze, torch.zeros(maze.shape[0], device=maze.device)).sample.float()

    def build_optimizers(self, steps):
        # The PaTh painter's recipe (PainterBase.build_optimizers with amaze_unet's optim_cfg).
        opt = torch.optim.AdamW(self.parameters(), lr=0, weight_decay=0.01, betas=(0.9, 0.999))
        return [ScheduledOptimizer(opt, base_lr=1e-4, warmup_steps=2000, num_steps=steps, min_ratio=1.0)]

    def clip(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    @torch.no_grad()
    def predict(self, maze, n_sup=None):
        return self(maze)


class Adapter:
    """What sample_and_score needs from a model: _batch_to_sample, sampling_pipeline.sample_one_batch, decode_for_eval."""

    def __init__(self, net, n_sup):
        self.net, self.n_sup, self.sampling_pipeline = net, n_sup, self

    def _batch_to_sample(self, batch, device):
        return PainterBase._batch_to_sample(None, batch, device)

    def sample_one_batch(self, model, conditions, device, generator=None):
        return self.net.predict(conditions.spatial_conditions.to(device), self.n_sup)

    def decode_for_eval(self, x):
        return x.clamp(0, 1)


class GTAdapter(Adapter):
    """Returns the ground-truth solution images: the evaluator must give ~100% exact on them (cluster sanity check)."""

    def __init__(self):
        super().__init__(None, 0)

    def sample_one_batch(self, model, conditions, device, generator=None):
        return conditions.images.to(device)


def score(adapter, cfg, data_root, device, out_dir, combos, batch_size=128):
    per_combo = {}
    for g, s in combos:
        ds = _build_amaze_dataset(cfg, str(Path(data_root) / "maze" / g / f"n{s}_test.parquet"))
        rows, _ = sample_and_score(adapter, ds, "maze", device, 1, 0, batch_size, combo_name=f"{g}_{s}",
                                   samples_dir=Path(out_dir) / "samples", generations_dir=None)
        per_combo[f"{g}_{s}"] = rows
    return per_combo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["trm_unet", "unet"])
    ap.add_argument("--data_root", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--micro_batch", type=int, default=192); ap.add_argument("--n_micro", type=int, default=4)
    ap.add_argument("--n_sup", type=int, default=16); ap.add_argument("--ema", type=float, default=0.999)
    # lr / warm-up / weight decay come from the optimizers (PaTh's own for trm_unet, the painter's for unet).
    ap.add_argument("--eval_every", type=int, default=2000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--unet_channels", default="64,128,192"); ap.add_argument("--device", default=None)
    ap.add_argument("--max_train", type=int, default=0, help="debug: use only the first N training mazes")
    ap.add_argument("--eval_sets", default="", help="debug: e.g. square:8,circle:5 (default: all shapes x MAZE_SCALES)")
    ap.add_argument("--skip_sanity", action="store_true")
    ap.add_argument("--eval_only", action="store_true"); ap.add_argument("--count_only", action="store_true")
    ap.add_argument("--wandb_project", default=None, help="e.g. PaTh-rebuttal (workspace from WANDB_ENTITY); none = off")
    ap.add_argument("--wandb_name", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); torch.set_float32_matmul_precision("high")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = compose_cfg(a.data_root)
    if not a.skip_sanity:                                          # evaluator / third_party / data check on this cluster
        gt = score(GTAdapter(), cfg, a.data_root, device, out, [("square", 8), ("triangle", 16), ("circle", 16)])
        ex = {k: aggregate(v)["exact1"] for k, v in gt.items()}
        print(f"sanity: ground truth scored by the official evaluator, exact1 {ex}", flush=True)
        assert min(ex.values()) > 0.95, "official evaluator does not accept the ground truth: check eval / third_party / data"
    net = (TRMUNet(cfg) if a.model == "trm_unet" else PlainUNet([int(c) for c in a.unet_channels.split(",")])).to(device)
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"{a.model}: {n_par:.2f} M parameters", flush=True)
    if a.model == "trm_unet":
        m = net.m
        for name in ("condition_encoder", "thinker", "thinker_painter_translator", "painter"):
            print(f"  {name}: {sum(p.numel() for p in getattr(m, name).parameters()) / 1e6:.2f} M", flush=True)
    if a.count_only:
        return
    n_sup = a.n_sup if a.model == "trm_unet" else 1
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizers = net.build_optimizers(a.steps)
    print("optimizers: " + ", ".join(f"{type(o.optimizer).__name__}(lr={o.base_lr}, wd={o.optimizer.defaults.get('weight_decay')}, "
                                      f"betas={o.optimizer.defaults.get('betas')})" for o in optimizers), flush=True)
    step = 0
    ck = out / "checkpoint_last.pt"
    if ck.exists():
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(sd["model"]); ema.load_state_dict(sd["ema"]); step = sd["step"]
        for o, o_sd in zip(optimizers, sd["opt"]):
            o.load_state_dict(o_sd)
        print(f"resumed at step {step}", flush=True)

    def save(path):
        torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "opt": [o.state_dict() for o in optimizers],
                    "step": step, "args": vars(a)}, path)

    run = None
    if a.wandb_project:
        import wandb
        run_name = a.wandb_name or f"diffneed_{a.model}"
        id_file = out / "wandb_run_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id() if hasattr(wandb.util, "generate_id") \
            else __import__("wandb.sdk.lib.runid", fromlist=["generate_id"]).generate_id()
        run = wandb.init(project=a.wandb_project, id=run_id, resume="allow", name=run_name, group="diffusion_needed",
                         job_type="train", tags=["diffusion_needed", a.model, "deterministic"],
                         config={**vars(a), "params_M": n_par, "n_sup": a.n_sup if a.model == "trm_unet" else 1},
                         settings=wandb.Settings(init_timeout=300))
        id_file.write_text(run_id)

    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    if not a.eval_only and step < a.steps:
        td = cfg.data.train_dataset
        train_ds = AmazeDataset(dataset_path=f"{a.data_root}/maze/maze_dataset_train_144.parquet", split="train",
                                image_size=144, condition_field=td.condition_field, target_field=td.target_field,
                                num_channels=3, include_metadata=False)
        if a.max_train:
            train_ds = torch.utils.data.Subset(train_ds, range(a.max_train))
        dl = DataLoader(train_ds, batch_size=a.micro_batch * a.n_micro, shuffle=True, drop_last=True, num_workers=a.workers,
                        collate_fn=AmazeDataset.collate_fn, persistent_workers=a.workers > 0)
        print(f"train {len(train_ds)} mazes, device {device}", flush=True)
        net.train()
        done = False
        while not done:
            for b in dl:
                maze, img = b.spatial_conditions.to(device, non_blocking=True), b.images.to(device, non_blocking=True)
                mbs = list(zip(maze.split(a.micro_batch), img.split(a.micro_batch)))
                carries = [net.init_carry(m_.shape[0], device) for m_, _ in mbs] if a.model == "trm_unet" else None
                for _ in range(n_sup):                         # deep supervision (TRM) / one step (UNet)
                    tot = 0.0
                    for k, (m_, y) in enumerate(mbs):
                        if a.model == "trm_unet":
                            pred, c = net.step(m_, carries[k])
                            carries[k] = (c[0].detach(), c[1].detach())
                        else:
                            pred = net(m_)
                        loss = F.mse_loss(pred, y)
                        # PaTh: backward(loss / (global_batch_size * K)), global_batch_size = micro batch.
                        (loss / (a.micro_batch * len(mbs))).backward()
                        tot += float(loss) / len(mbs)
                    net.clip()
                    lr = apply_lr_and_step(optimizers, step)   # sets lr from the schedule, steps, zero_grad
                    with torch.no_grad():
                        for pe, pm in zip(ema.parameters(), net.parameters()):
                            pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                        for be, bm in zip(ema.buffers(), net.buffers()):
                            be.copy_(bm)
                    step += 1
                    if step % 100 == 0:
                        rec = {"step": step, "loss": tot, "lr": lr, "minutes": (time.time() - t0) / 60}
                        log.write(json.dumps(rec) + "\n"); log.flush()
                        if run is not None:
                            run.log({"train/loss": tot, "train/lr": lr}, step=step)
                        if step % 500 == 0:
                            print(rec, flush=True)
                    if step % a.eval_every == 0:
                        quick = aggregate(score(Adapter(ema, n_sup), cfg, a.data_root, device, out, [("square", 8)])["square_8"])
                        rec = {"step": step, "square8_exact1": quick["exact1"], "square8_pass1": quick["pass1"]}
                        log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
                        if run is not None:
                            run.log({"val/square8_exact1": quick["exact1"], "val/square8_pass1": quick["pass1"]}, step=step)
                        save(ck)
                    if step >= a.steps or (time.time() - t0) / 60 > a.max_minutes:
                        done = True
                        break
                if done:
                    break
        save(ck)
        if step >= a.steps:
            save(out / "checkpoint_final.pt")
        print(f"training stopped at step {step} after {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- full official-style evaluation (EMA weights, 1 attempt, in-distribution sizes only)
    ema.eval()
    combos = ([(x.split(":")[0], int(x.split(":")[1])) for x in a.eval_sets.split(",")] if a.eval_sets
              else [(g, s) for g in MAZE_GEOMETRIES for s in MAZE_SCALES])
    per_combo = score(Adapter(ema, n_sup), cfg, a.data_root, device, out, combos)
    ood_combos = [] if a.eval_sets else [(g, s) for g in MAZE_GEOMETRIES for s in MAZE_OOD_SCALES]
    ood_combo = score(Adapter(ema, n_sup), cfg, a.data_root, device, out, ood_combos) if ood_combos else {}
    res = build_maze_result(per_combo, ood_combo)
    res.update({"code": os.getcwd(), "checkpoint": str(ck), "step": step, "model": a.model, "params_M": n_par,
                "n_puzzles": sum(map(len, per_combo.values())), "attempts": 1})
    (out / "amaze_metrics_maze.json").write_text(json.dumps(res, indent=1))
    print(f"overall exact1 {res['overall']['exact1']:.3f} pass1 {res['overall']['pass1']:.3f} (step {step})")
    if run is not None:
        log_tables(run, "maze", res)
        run.finish()
    for g in res["per_shape"]:
        print(g, " | ".join(f"n{s} {m['exact1']:.2f}" for s, m in res["per_shape"][g].items()), flush=True)


if __name__ == "__main__":
    main()
