"""TEMPORARY (Claude): is PaTh's thinker already worse than plain TRM at the FIRST denoising step (pure noise)?

The plain TRMs at evaluation = one "think" from the maze (+ pure noise for plain-noisy), fresh carry, n_sup steps.
PaTh's thinker at the first denoising step is the same situation (pure-noise x_t, fresh carry, n_sup steps); later
steps see what the painter has already drawn. Per maze size (square test sets), 100 boards:
  plain TRMs (env PLAIN_CKPTS=name:path,...): own 2-class head; linear probes (ridge, 2-fold over boards) on the final z_L and z_H
  PaTh: real CFG sampling; at denoising steps `record_steps` (step 0 = first, pure noise) the conditional pass's final
        z_L / z_H, read by the same probes; plus the final drawn image (token >= 10 % blue = path)
Score per readout: token balanced accuracy and exact boards (every known token right; ambiguous tokens ignored).
Output: <out_dir>/first_step.pt, printed table.
"""
import dataclasses
import logging
import os
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from interp_template import CONFIG_DIR, G, TOK, blue, ints, ridge_fit_predict, token_targets


def score(pred, lab):
    """pred, lab (B, T); lab -1 = ignored."""
    k = lab >= 0
    accs = [((pred == c) & k & (lab == c)).sum() / ((lab == c) & k).sum().clamp_min(1) for c in (0, 1)]
    exact = ((pred == lab) | ~k).all(1).float().mean()
    return float(sum(accs) / 2), float(exact)


def probe(X, lab, device):
    """2-fold ridge over boards on known tokens; returns (B, T) 0/1 predictions."""
    B = X.shape[0]
    fold = torch.arange(B) % 2
    out = torch.zeros(lab.shape)
    for fo in (0, 1):
        tr, te = fold == fo, fold != fo
        Xtr, ytr = X[tr].flatten(0, 1).float().to(device), lab[tr].flatten().to(device)
        k = ytr >= 0
        s = ridge_fit_predict(Xtr[k], (2 * ytr[k] - 1)[:, None], X[te].flatten(0, 1).float().to(device))[:, 0]
        out[te] = (s > 0).float().cpu().reshape(int(te.sum()), -1)
    return out


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig):
    ic = cfg.get("interprebability_config", {})
    out_dir = Path(str(ic.out_dir)); out_dir.mkdir(parents=True, exist_ok=True)
    sizes = ints(ic.get("maze_sizes", "8,11,13,16"))
    record_steps = ints(ic.get("record_steps", "0,5,10,19"))
    data_tmpl = str(ic.get("data_tmpl", "data/amaze/maze/square/n{N}_test.parquet"))
    n_boards, seed = int(ic.get("num_boards", 100)), int(ic.get("seed", 0))
    plains = [s.split(":", 1) for s in os.environ.get("PLAIN_CKPTS", "").split(",") if s]   # name:path,...
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    from train_plain_trm import PlainTRM, T_TRAIN
    plain_models = {}
    for name, path in plains:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = PlainTRM(n_sup=int(ck["args"].get("n_sup", 16)), noisy=bool(ck["args"].get("noisy", False)))
        m.load_state_dict(ck["ema"]); plain_models[name] = m.to(device).eval()
        logging.info(f"[{name}] {path}: step {ck['step']}, noisy={m.noisy}")

    model = build_model(cfg, instantiate(cfg.diffusion))
    load_checkpoint(model, str(ic.checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.unwrap_model(accelerator.prepare(model)); model.eval()
    scale = instantiate(cfg.sampling).cfg_scale
    st = {"rec": False}
    orig_rs = model.thinker.reasoning_step

    def rs(*a, **k):
        lg, zh, zl = orig_rs(*a, **k)
        if st["rec"]:
            st.update(z_H=zh.detach().float().cpu(), z_L=zl.detach().float().cpu())
        return lg, zh, zl

    model.thinker.reasoning_step = rs

    res = {}
    for N in sizes:
        ds = AmazeDataset(dataset_path=data_tmpl.format(N=N), split="test", image_size=144, condition_field="m_original_img",
                          target_field="sol_img", num_channels=3)
        nb = min(n_boards, len(ds))
        batch = AmazeDataset.collate_fn([ds[i] for i in range(nb)])
        lab = token_targets(blue(batch.images.float()))["path"]
        rows = {}
        maze = batch.spatial_conditions.to(device)
        for name, pm in plain_models.items():
            with torch.no_grad():
                x_t = t = None
                if pm.noisy:
                    x_t = torch.randn(maze.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))
                    t = torch.full((nb,), T_TRAIN - 1, device=device, dtype=torch.long)
                enc = (pm.encoder(maze, x_t, t) if pm.noisy else pm.encoder(maze)).enc_emb
                z_H, z_L = pm.init_carry(nb, device)
                for _ in range(pm.n_sup):
                    lg, z_H, z_L = pm.thinker.reasoning_step(enc, z_H, z_L)
            rows[f"{name} head"] = score((lg.argmax(-1) == 1).float().cpu(), lab)
            rows[f"{name} z_L probe"] = score(probe(z_L.float().cpu(), lab, device), lab)
            rows[f"{name} z_H probe"] = score(probe(z_H.float().cpu(), lab, device), lab)

        cond = model._batch_to_sample(batch, device)
        model.scheduler.set_timesteps(20, device=device)
        x = torch.randn((nb, *model.noise_shape), device=device, generator=torch.Generator(device=device).manual_seed(seed))
        with torch.no_grad():
            for i, t in enumerate(model.scheduler.timesteps):
                s = dataclasses.replace(cond, x_noisy=x, timesteps=t.expand(nb).to(device))
                st["rec"] = True
                pc = model(s).pred
                st["rec"] = False
                if i in record_steps:
                    rows[f"PaTh step {i:2d} (t={int(t):2d}) z_L"] = score(probe(st["z_L"], lab, device), lab)
                    rows[f"PaTh step {i:2d} (t={int(t):2d}) z_H"] = score(probe(st["z_H"], lab, device), lab)
                pu = model(model.null_condition_sample(s)).pred
                x = model.scheduler.step(pu + scale * (pc - pu), t, x).prev_sample
        dm = blue(model.decode_for_eval(x).float())
        drawn = (dm.reshape(nb, G, TOK, G, TOK).mean((2, 4)).reshape(nb, -1) >= 0.1).float()
        rows["PaTh drawn (final image)"] = score(drawn, lab)
        res[N] = rows
        print(f"\n=== n{N} ({144 / N / TOK:.2f} tokens per cell side), {nb} boards: token balanced acc | exact boards")
        for k, (a, e) in rows.items():
            print(f"   {k:32s} {a:6.3f} {e:6.1%}")
        torch.save(res, out_dir / "first_step.pt")
    print(f"saved {out_dir}/first_step.pt")


if __name__ == "__main__":
    main()
