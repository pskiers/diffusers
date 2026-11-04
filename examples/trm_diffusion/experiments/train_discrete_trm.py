"""TEMPORARY (Claude, amaze interpretability): a token-per-cell TRM on discrete square mazes (experiments/discrete_maze.py).

Same thinker and training recipe as train_plain_trm.py (SpatialTRM: hidden 512, 8 heads, 2 layers, H_cycles 3, L_cycles 6,
n_sup 16, RoPE; deep supervision with the carry passed on, one optimizer step per reasoning step, batch 768, AdamW lr 1e-4,
wd 0.3, betas (0.9, 0.95), 2000 warm-up steps, grad clip 1.0, EMA 0.999, cross-entropy), but the input is one token per
maze cell (embedding of: open passages up/right/down/left + start/goal marker; padding outside the maze) on a 16 x 16
grid (256 tokens) instead of the CNN encoding of the image (12 x 12 image tokens). No cell/token misalignment.
Train: data/discrete/train.npz (amaze square generator, training seeds and algorithms; sizes 5, 7, 8, 9, 11, 13, 16).
Eval: exact boards on the square test sets (data/discrete/n<size>_test.npz) with the EMA weights.
Checkpoints: <out_dir>/checkpoint_last.pt (resumable), checkpoint_step<k>.pt at --keep_steps, checkpoint_final.pt.

python experiments/train_discrete_trm.py --train data/discrete/train.npz --test_dir data/discrete --out_dir runs/claude_discrete_trm
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.trm_wrappers import SpatialTRM

GRID = 16
N_IDS = 49


class DiscreteTRM(nn.Module):
    def __init__(self, n_sup=16, H_cycles=3, L_cycles=6, hidden=512):
        super().__init__()
        self.embed = nn.Embedding(N_IDS, hidden)
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(hidden))      # unit scale after the thinker's embed_scale
        self.thinker = SpatialTRM(optim_cfg=None, vocab_size=2, seq_len=GRID * GRID, hidden_size=hidden, n_heads=8, L_layers=2,
                                  L_cycles=L_cycles, H_cycles=H_cycles, n_sup=n_sup, expansion=4.0, forward_dtype="bfloat16",
                                  pos_encodings="rope", puzzle_emb_ndim=0, puzzle_emb_len=0)
        self.n_sup = n_sup

    def init_carry(self, B, device):
        z_H, z_L = self.thinker.get_initial_states(B)
        return z_H.to(device), z_L.to(device)

    def step(self, ids, z_H, z_L):
        """One reasoning step (H_cycles x (L_cycles z_L updates + 1 z_H update)). Returns logits (B, 256, 2), z_H, z_L."""
        return self.thinker.reasoning_step(self.embed(ids), z_H, z_L)

    @torch.no_grad()
    def think(self, ids, n_sup=None):
        z_H, z_L = self.init_carry(ids.shape[0], ids.device)
        logits = None
        for _ in range(n_sup or self.n_sup):
            logits, z_H, z_L = self.step(ids, z_H, z_L)
        return logits


def load(path):
    d = np.load(path)
    return torch.from_numpy(d["ids"]), torch.from_numpy(d["labels"])


@torch.no_grad()
def evaluate(model, tests, device):
    model.eval()
    out = {}
    for name, (ids, lab) in tests.items():
        ok, tok_ok, tok_n = 0, 0, 0
        for k in range(0, len(ids), 100):
            x, y = ids[k:k + 100].to(device), lab[k:k + 100].to(device)
            pred = model.think(x).argmax(-1)
            known = y >= 0
            ok += int(((pred == y) | ~known).all(1).sum())
            tok_ok += int(((pred == y) & known).sum()); tok_n += int(known.sum())
        out[name] = round(ok / len(ids), 3)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_sizes", default="5,8,10,11,16")
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--micro_batch", type=int, default=96)
    ap.add_argument("--n_micro", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.3); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--keep_steps", default="4000,10000,20000")
    ap.add_argument("--max_minutes", type=float, default=1e9); ap.add_argument("--n_sup", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = DiscreteTRM(n_sup=args.n_sup).to(device)
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
    ids, lab = load(args.train)
    tests = {f"n{n}": load(Path(args.test_dir) / f"n{n}_test.npz") for n in args.test_sizes.split(",")}
    dl = DataLoader(TensorDataset(ids, lab), batch_size=args.micro_batch * args.n_micro, shuffle=True, drop_last=True)
    keep = {int(x) for x in args.keep_steps.split(",") if x}
    print(f"train {len(ids)} mazes, tests {list(tests)}; params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()

    def save(path):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "args": vars(args)}, path)

    done = step >= args.steps
    model.train()
    while not done:
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            mbs = list(zip(x.split(args.micro_batch), y.split(args.micro_batch)))
            carries = [model.init_carry(m.shape[0], device) for m, _ in mbs]
            for _ in range(args.n_sup):
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                for k, (m, yy) in enumerate(mbs):
                    logits, z_H, z_L = model.step(m, *carries[k])
                    carries[k] = (z_H, z_L)
                    loss = F.cross_entropy(logits.float().reshape(-1, 2), yy.reshape(-1), ignore_index=-1)
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
                    rec = {"step": step, "eval_ema_exact": evaluate(ema, tests, device)}
                    log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
                    save(ck)
                if step in keep:
                    save(out / f"checkpoint_step{step}.pt")
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
