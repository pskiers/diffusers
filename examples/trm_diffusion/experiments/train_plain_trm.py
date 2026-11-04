"""TEMPORARY (Claude, amaze interpretability): a plain TRM on Amaze mazes, for comparison with PaTh's thinker.

Maze image in, per-token path / off out: no painter, no diffusion, no noisy image.
  - thinker: the same SpatialTRM as PaTh's (hidden 512, 8 heads, 2 layers, H_cycles 3, L_cycles 6, n_sup 16, bf16, RoPE,
    144 tokens), with a 2-class output head instead of the 11-dim plan
  - encoder: the same CNN as PaTh's condition encoder (128 channels, [128, 256, 256], 12-px tokens), reading only the
    maze (3 channels); PaTh's also reads the noisy image and the timestep
  - labels per 12-px token from sol_img: path if >= 10% of the token's pixels are path, off if none, else ignored
    (ZPathEncodingAblation._token_labels, the labels of the path probe)
  - training as PaTh's thinker (models/painter_thinkers.py): per batch n_sup reasoning steps with the carry passed on
    (detached), one optimizer step per reasoning step, 4 micro-batches of 192 = 768 per step, 40,000 steps; AdamW
    lr 1e-4, wd 0.3, betas (0.9, 0.95), 2000 warm-up steps, grad clip 1.0, EMA 0.999; cross-entropy instead of the
    diffusion loss
  - data: maze_dataset_train_144.parquet (PaTh's amaze training set); eval on n8_test.parquet (8x8 squares): token
    accuracy and exact boards after n_sup steps, with the EMA weights
--noisy: "plain TRM trained like PaTh": the encoder is PaTh's NoisySpatialConditionEncoder (maze + noisy image x_t,
    6 channels, + timestep embedding); x_t = the solution image noised with PaTh's schedule (DDPM, 100 train steps,
    squaredcos_cap_v2) at a random t per maze, fixed over the n_sup steps of a batch (as PaTh). Targets unchanged.
    think() without x_t starts from pure noise at the last timestep (the maze alone decides).
--loss mse: mean squared error between the path probability (softmax) and the 0/1 label per token, instead of
    cross-entropy (to separate "MSE vs cross-entropy" from "pixel space through the painter").
Checkpoints (resumable): <out_dir>/checkpoint_last.pt, <out_dir>/checkpoint_final.pt; log: <out_dir>/train_log.jsonl

python experiments/train_plain_trm.py --train <parquet> --test <parquet> --out_dir runs/claude_plain_trm [--max_minutes 110]
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.amaze_dataset import AmazeDataset
from experiments.interpretability.z_path_encoding_ablation import ZPathEncodingAblation
from diffusers import DDPMScheduler

from models.condition_encoders import NoisySpatialConditionEncoder, SpatialConditionEncoder
from models.trm_wrappers import SpatialTRM

G = 12
T_TRAIN = 100                                   # PaTh's diffusion config (configs/diffusion/ddpm.yaml)


class PlainTRM(nn.Module):
    def __init__(self, n_sup=16, H_cycles=3, L_cycles=6, noisy=False):
        super().__init__()
        self.noisy = noisy
        if noisy:
            self.encoder = NoisySpatialConditionEncoder(in_channels=6, enc_channels=128, hidden_channels=[128, 256, 256],
                                                        output_dim=512, factor=G, num_train_timesteps=T_TRAIN,
                                                        with_timestep_emb=True)
        else:
            self.encoder = SpatialConditionEncoder(in_channels=3, enc_channels=128, hidden_channels=[128, 256, 256],
                                                   output_dim=512, factor=G)
        self.thinker = SpatialTRM(optim_cfg=None, vocab_size=2, seq_len=G * G, hidden_size=512, n_heads=8, L_layers=2,
                                  L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, expansion=4.0, forward_dtype="bfloat16",
                                  pos_encodings="rope", puzzle_emb_ndim=0, puzzle_emb_len=0)
        self.n_sup = n_sup

    def init_carry(self, B, device):
        z_H, z_L = self.thinker.get_initial_states(B)
        return z_H.to(device), z_L.to(device)

    def step(self, maze, z_H, z_L, x_t=None, t=None):
        """One reasoning step (H_cycles x (L_cycles z_L updates + 1 z_H update)). Returns logits (B, 144, 2), z_H, z_L."""
        emb = (self.encoder(maze, x_t, t) if self.noisy else self.encoder(maze)).enc_emb
        return self.thinker.reasoning_step(emb, z_H, z_L)

    @torch.no_grad()
    def think(self, maze, n_sup=None, x_t=None, t=None):
        if self.noisy and x_t is None:                                    # pure noise at the last timestep
            x_t = torch.randn_like(maze)
            t = torch.full((maze.shape[0],), T_TRAIN - 1, device=maze.device, dtype=torch.long)
        z_H, z_L = self.init_carry(maze.shape[0], maze.device)
        logits = None
        for _ in range(n_sup or self.n_sup):
            logits, z_H, z_L = self.step(maze, z_H, z_L, x_t, t)
        return logits


def collate(samples):
    b = AmazeDataset.collate_fn(samples)
    labels = ZPathEncodingAblation._token_labels(b.images, G, 0.1).long()      # 1 path, 0 off, -1 ignored
    return b.spatial_conditions, labels, b.images


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tok_ok = tok_n = boards_ok = boards = 0
    for maze, labels, _ in loader:
        maze, labels = maze.to(device), labels.to(device)
        pred = model.think(maze).float().argmax(-1)
        known = labels >= 0
        ok = (pred == labels) | ~known
        tok_ok += int(((pred == labels) & known).sum()); tok_n += int(known.sum())
        boards_ok += int(ok.all(1).sum()); boards += len(labels)
    model.train()
    return {"token_acc": tok_ok / max(1, tok_n), "boards_exact": boards_ok / max(1, boards), "n_boards": boards}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--micro_batch", type=int, default=192)
    ap.add_argument("--n_micro", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.3); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--max_minutes", type=float, default=110); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n_sup", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noisy", action="store_true", help="PaTh-like input: maze + noisy solution image x_t + timestep")
    ap.add_argument("--loss", choices=["ce", "mse"], default="ce")
    ap.add_argument("--device", default=None, help="override (e.g. cpu for a local test; MPS lacks the encoder's pool)")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    model = PlainTRM(n_sup=args.n_sup, noisy=args.noisy).to(device)
    sched = DDPMScheduler(num_train_timesteps=T_TRAIN, beta_schedule="squaredcos_cap_v2", prediction_type="sample")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    step = 0
    ck = out / "checkpoint_last.pt"
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed at step {step}", flush=True)
    train_ds = AmazeDataset(dataset_path=args.train, split="train", image_size=144, include_metadata=False)
    test_ds = AmazeDataset(dataset_path=args.test, split="test", image_size=144, include_metadata=False)
    train_dl = DataLoader(train_ds, batch_size=args.micro_batch * args.n_micro, shuffle=True, drop_last=True,
                          num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    test_dl = DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=min(2, args.workers), collate_fn=collate)
    print(f"train {len(train_ds)} mazes, test {len(test_ds)}; params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M; "
          f"device {device}", flush=True)
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()

    def save(path):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "args": vars(args)}, path)

    done = step >= args.steps                   # a follow-up job after the last step only re-saves the final checkpoint
    model.train()
    while not done:
        for maze, labels, images in train_dl:
            maze, labels = maze.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if args.noisy:                                    # one random t per maze, fixed over the n_sup steps
                images = images.to(device, non_blocking=True)
                t = torch.randint(0, T_TRAIN, (images.shape[0],), device=device)
                x_t = sched.add_noise(images, torch.randn_like(images), t)
            else:
                x_t = t = torch.zeros(maze.shape[0], device=device)
            mbs = list(zip(maze.split(args.micro_batch), labels.split(args.micro_batch),
                           x_t.split(args.micro_batch), t.split(args.micro_batch)))
            carries = [model.init_carry(m.shape[0], device) for m, *_ in mbs]
            for _ in range(args.n_sup):                       # deep supervision, as PaTh's thinker
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                for k, (m, y, xt, tt) in enumerate(mbs):
                    logits, z_H, z_L = model.step(m, *carries[k], xt if args.noisy else None, tt if args.noisy else None)
                    carries[k] = (z_H, z_L)
                    if args.loss == "ce":
                        loss = F.cross_entropy(logits.float().reshape(-1, 2), y.reshape(-1), ignore_index=-1)
                    else:
                        known = y >= 0
                        p_path = torch.softmax(logits.float(), -1)[..., 1]
                        loss = ((p_path - y.float()) ** 2)[known].mean()
                    (loss / len(mbs)).backward()
                    tot += float(loss) / len(mbs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = args.lr * min(1.0, (step + 1) / args.warmup)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
                step += 1
                if step % 100 == 0:
                    rec = {"step": step, "loss": tot, "lr": lr, "minutes": (time.time() - t0) / 60}
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    if step % 500 == 0:
                        print(rec, flush=True)
                if step % args.eval_every == 0 or step >= args.steps:
                    ev = evaluate(ema, test_dl, device)
                    rec = {"step": step, "eval_ema": ev}
                    log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
                    save(ck)
                if step >= args.steps or (time.time() - t0) / 60 > args.max_minutes:
                    done = True
                    break
            if done:
                break
    save(ck)
    if step >= args.steps:
        save(out / "checkpoint_final.pt")
    print(f"stopped at step {step} after {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
