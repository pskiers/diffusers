"""
experiments/amaze_interpretability.py — interpretability probe (AMAZE, 8x8 square).

1. Generate boards from pairs of corrupted and correct conditionings from the AMAZE 8x8 square dataset on t=70 timestamp.
The thinker sees x_noisy (amaze_spatial_v1 concatenates it with the maze).
So any difference between the two runs' thinker activations is caused by the dead-end.

2. Analyze Thinker's internal state z across all recurrent steps (on the one diffusion step t=70). Compare z_H between the corrupted and clean conditionings to identify the effect of the dead-end strokes.
    2.1. Paired difference-of-means:
        - Compute the mean of z_H for the corrupted conditionings.
        - Compute the mean of z_H for the clean conditionings.
        - Subtract the two means to isolate the effect of the dead-end strokes (per batch, see if it consistently appears across different batches).
    2.2. Linear probe (logistic regression)
        - Train a logistic regression model to predict the presence of dead-end strokes from z_H.
        - Evaluate the model's performance to quantify how well z_H encodes the dead-end information.

3. Intervention (add or ablate the direction)
    - Modify z_H by adding or removing the component corresponding to the dead-end strokes.
    - Observe the effect on the Thinker's output to assess the causal role of the dead-end information.

4. Analyze Thinker's internal state z across all recurrent steps (on the one diffusion step t=30).
    - z_H is reset at every denoising step, so later steps are separate measurements.
    - They show the signal fading as the dead-end disappears from x_t.
"""
# NA POCZĄTEK:
# Mam wewnetrzne stanh thinkera, chce policzyc ich srednia roznice w kazdym batchu per podane timestepy.
# Czy mozna jakos zmapowac te 512-dim wektory na 2D do wizualizacji?
# porownam jak wygladaja te roznice miedzy batchami na roznych timestampach
import json
import logging
import os
from pathlib import Path

import hydra
import maze_corruption_lib as mcl
import torch
from accelerate import Accelerator
from datasets.amaze_dataset import AmazeDataset
from eval.checkpoint_utils import load_checkpoint
from factory import build_model
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import random
import cv2
import dataclasses
import matplotlib
matplotlib.use("Agg")  # no display needed, the chart is written to a file
import matplotlib.pyplot as plt


PREFIX_FRAC = 0.5         # add wrong path after 50% of the correct path

@torch.no_grad()
def _denoise_from_full(
        model, conditions, x_t: torch.Tensor, run_timesteps: torch.Tensor, states: dict, generator: torch.Generator | None = None, record_at: set = frozenset()
    ) -> tuple:
    """Run the reverse process from x_t down to t=0 over `run_timesteps`.

    At every t in `record_at`, also records into `states[t]` every z_L update of the thinker
    (conditional pass only: n_sup * H_cycles * L_cycles = 288 of them), averaged over the
    tokens and over the batch: (n_z_L_updates, hidden).
    """
    predictor = model.sampling_pipeline.predictor
    x = x_t

    for t in run_timesteps:
        t_batch = t.expand(x.shape[0]).to(x.device)
        step_sample = dataclasses.replace(conditions, x_noisy=x, timesteps=t_batch)
        record = int(t.item()) in record_at

        # The TRM runs every update through one block, L_level: per H-cycle L_cycles times
        # for z_L, then once for z_H. A hook on it sees all of them, in order.
        level_outputs = []
        if record:
            handle = model.thinker.inner.L_level.register_forward_hook(
                lambda module, inputs, output: level_outputs.append(output.float().mean(1).cpu())  # (B, hidden), mean over tokens
            )
        pred_cond, _, _ = model.forward_with_carry(step_sample)
        if record:
            handle.remove()
        pred_uncond = model(model.null_condition_sample(step_sample)).pred
        pred = pred_uncond + predictor.scale * (pred_cond.pred - pred_uncond)

        if record:  # record the thinker's states at this timestep
            per_cycle = model.thinker.inner.config.L_cycles + 1  # the last call of each cycle is the z_H update
            z_L_all = torch.stack([z for i, z in enumerate(level_outputs) if i % per_cycle != per_cycle - 1], 1)  # (B, 288, hidden)
            states[int(t.item())] = z_L_all.mean(0)  # mean over the boards in this batch
        kw = {"generator": generator} if generator is not None else {}
        x = model.scheduler.step(pred, t, x, **kw).prev_sample

@torch.no_grad()
def _run_one_tstart(
        model, ds, device, t_start, run_timesteps, n_total, batch_size, seed, states_path
    ) -> tuple:
    processed = 0
    batch_means = []  # one entry per batch: {"n": B, "corrupted": {t: mean z_L}, "clean": {t: mean z_L}}

    # 1. Generate pairs of corrupted and correct conditionings from the AMAZE dataset on t=70 timestamp.
    for start in range(0, n_total, batch_size):
        base_seed = seed + start * 977
        generator = torch.Generator(device=device).manual_seed(base_seed)

        # Fetch a batch of raw samples from the dataset.
        raw = [ds[i] for i in range(start, min(start + batch_size, n_total))]
        pairs = [(s, mcl.build_maze_sample(s.metadata)) for s in raw]
        pairs = [(s, ms) for s, ms in pairs if ms is not None]  # drops boards that build_maze_sample couldn't parse.
        if not pairs:
            continue
        samples, mss = [s for s, _ in pairs], [ms for _, ms in pairs]

        # Construct the conditions for the model - first, partially solved boards from GT
        partially_solved_boards = [mcl.render_partial(ms, PREFIX_FRAC) for ms in mss]

        kept_samples, corrupted_images, clean_images = [], [], []
        for i, ms in enumerate(mss):
            rng = random.Random(base_seed + i * 131)

            # Construct the corrupted version of the partial board
            partial_board, shown_cells = partially_solved_boards[i]
            partial_board_corrupted, wrong_cells, added_correct_cells, diverge = mcl.apply_add(partial_board, ms, shown_cells, rng)
            if not wrong_cells: # walk reached the goal, no dead-end -> skip board
                continue

            kept_samples.append(samples[i])
            corrupted_images.append(partial_board_corrupted)

            # Construct the clean version of the partial board with the added correct cells
            img_clean = partial_board.copy()
            full = [shown_cells[-1]] + added_correct_cells
            for x, y in zip(full[:-1], full[1:]):
                cv2.line(
                    img_clean,
                    mcl._center(ms.id2rc[x], ms.cell_px),
                    mcl._center(ms.id2rc[y], ms.cell_px),
                    ms.blue_color, ms.stroke
                )
            clean_images.append(img_clean)

        if not kept_samples:
            continue

        B = len(kept_samples) # It can be smaller if build_maze_sample couldn't parse some boards or no dead-end was added.
        processed += B

        conditions = model._batch_to_sample(AmazeDataset.collate_fn(kept_samples), device)
        corrupted_images_tensor = torch.stack([ds.transform(Image.fromarray(img)) for img in corrupted_images]).to(device)
        clean_images_tensor = torch.stack([ds.transform(Image.fromarray(img)) for img in clean_images]).to(device)

        noise = torch.randn(corrupted_images_tensor.shape, generator=generator, device=device)
        timesteps = torch.full((B,), t_start, device=device, dtype=torch.long)

        # Generate output from the corrupted images
        x_corr = model.scheduler.add_noise(corrupted_images_tensor, noise, timesteps)
        corr_states = {}
        _denoise_from_full(
            model, conditions, x_corr, run_timesteps, corr_states,
            torch.Generator(device=device).manual_seed(base_seed + 1), {int(t) for t in run_timesteps}  # record at every denoising step
        )

        # Generate output from the clean images
        x_clean = model.scheduler.add_noise(clean_images_tensor, noise, timesteps)
        clean_states = {}
        _denoise_from_full(
            model, conditions, x_clean, run_timesteps, clean_states,
            torch.Generator(device=device).manual_seed(base_seed + 1), {int(t) for t in run_timesteps}
        )

        batch_means.append({"n": B, "corrupted": corr_states, "clean": clean_states})

    Path(states_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch_means, states_path)

    return batch_means, processed


def cosine_similarity(batch_means_diff: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between the difference vectors of different batches, per z_L update.

    Args:
        batch_means_diff (torch.Tensor): corrupted - clean batch means, shape (num_batches, n_steps, hidden).

    Returns:
        torch.Tensor: (n_steps,) cosine averaged over all pairs of different batches.
    """
    unit = batch_means_diff / batch_means_diff.norm(dim=-1, keepdim=True)
    cos = torch.einsum("isd,jsd->sij", unit, unit)  # (n_steps, num_batches, num_batches)
    different_batches = ~torch.eye(batch_means_diff.shape[0], dtype=torch.bool)
    return cos[:, different_batches].mean(-1)


def plot_z_L_differences(cosine: dict, relative_norm: dict, updates_per_sup: int, path: str) -> None:
    """Two panels over the z_L updates, one line per recorded timestep:
    top - cosine between batches, bottom - ‖corrupted - clean‖ / ‖clean‖."""
    ts = sorted(cosine, reverse=True)
    t_norm = plt.Normalize(min(ts), max(ts))
    colors = [plt.cm.viridis(t_norm(t)) for t in ts]  # one color per timestep, ordered by t
    fig, (ax_cos, ax_norm) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")

    for color, t in zip(colors, ts):
        steps = range(1, len(cosine[t]) + 1)
        ax_cos.plot(steps, cosine[t], color=color, lw=2, label=f"t={t}")
        mean, std = relative_norm[t].mean(0), relative_norm[t].std(0)
        ax_norm.plot(steps, mean, color=color, lw=2, label=f"t={t}")
        ax_norm.fill_between(steps, (mean - std).clamp(min=1e-6), mean + std, color=color, alpha=0.15, lw=0)

    ax_norm.axhline(0.01, color="#52514e", lw=1, ls="--")
    ax_norm.text(1, 0.011, "~bf16 rounding", color="#52514e", fontsize=8, va="bottom")

    for ax in (ax_cos, ax_norm):
        for s in range(updates_per_sup, len(cosine[t]), updates_per_sup):  # outer reasoning-step boundaries
            ax.axvline(s + 0.5, color="#52514e", lw=0.5, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    ax_cos.set(ylabel="cosine between batches", ylim=(-1, 1))
    ax_cos.axhline(0, color="#52514e", lw=0.8)
    fig.colorbar(plt.cm.ScalarMappable(norm=t_norm, cmap="viridis"), ax=[ax_cos, ax_norm], label="denoising timestep t")
    ax_norm.set(yscale="log", ylabel="‖corrupted − clean‖ / ‖clean‖",
                xlabel=f"z_L update (thin lines: every {updates_per_sup} = one outer reasoning step)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # CONFIG
    interprebability_config = cfg.get("interprebability_config", {})
    num_samples = int(interprebability_config.get("num_samples", 256))
    batch_size = int(interprebability_config.get("batch_size", 16))
    seed = int(interprebability_config.get("seed", 0))
    t_start_list = [
        int(t) for t in (interprebability_config.get("t_start", [70]))
    ]
    checkpoint = interprebability_config.get("checkpoint", None)
    model_name = interprebability_config.get("model_name", "painter_thinker_default")
    if checkpoint is None:
        raise SystemExit("Set +checkpoint=<...>.pt")

    data_parquet = str(interprebability_config.get("data_parquet"))
    output_dir = str(interprebability_config.get("output_dir", "runs/maze_corruption/"))
    out_path = f"{output_dir}/{model_name}.json"

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator(mixed_precision=cfg.precision.mixed_precision)
    device = accelerator.device

    scheduler = instantiate(cfg.diffusion)
    model = build_model(cfg, scheduler)
    load_checkpoint(model, str(checkpoint), use_ema=cfg.get("use_ema", True), device="cpu")
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()

    num_inference_steps = int(interprebability_config.get("num_inference_steps", model.sampling_pipeline.num_inference_steps))
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    full_ts = model.scheduler.timesteps

    if not os.path.isabs(data_parquet):
        data_parquet = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_parquet)

    vd = cfg.data.get("val_dataset", {})
    ds = AmazeDataset(
        dataset_path=data_parquet, split="test",
        image_size=int(cfg.data.get("image_size", 144)),
        condition_field=vd.get("condition_field", "m_original_img"),
        target_field=vd.get("target_field", "sol_img"),
        num_channels=int(vd.get("num_channels", 3)), include_metadata=True,
    )
    n_total = min(num_samples, len(ds))
    logging.info(f"[{model_name}] dataset={data_parquet} using {n_total}/{len(ds)} puzzles")

    cosine_similarities = {}
    for ts in t_start_list:
        run_timesteps = full_ts[full_ts <= ts]
        states_path = f"{output_dir}/{model_name}_z_L_batch_means_t{ts}.pt"

        if not os.path.exists(states_path):
            batch_means, processed = _run_one_tstart(model, ds, device, ts, run_timesteps, n_total, batch_size, seed, states_path)
        else:
            print(f"Loading precomputed z_L (batch means) states from {states_path}")
            batch_means = torch.load(states_path)

        # Calculate difference between corrupted and clean, for every recorded timestep
        cosine, relative_norm = {}, {}
        for t in batch_means[0]["corrupted"]:
            corrupted = torch.stack([b["corrupted"][t] for b in batch_means])  # (num_batches, 288, hidden)
            clean = torch.stack([b["clean"][t] for b in batch_means])
            batch_means_diff = corrupted - clean
            cosine[t] = cosine_similarity(batch_means_diff)                       # (288,)
            relative_norm[t] = batch_means_diff.norm(dim=-1) / clean.norm(dim=-1)  # (num_batches, 288)
        cosine_similarities[ts] = {
            t: {"cosine": cosine[t].tolist(), "relative_norm": relative_norm[t].mean(0).tolist()} for t in cosine
        }

        # Plot the differences for the current timestep
        cfg_thinker = model.thinker.inner.config
        plot_z_L_differences(
            cosine, relative_norm, cfg_thinker.H_cycles * cfg_thinker.L_cycles, f"{output_dir}/{model_name}_z_L_t{ts}.png"
        )

    if accelerator.is_main_process:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(cosine_similarities, f, indent=2)
        logging.info(f"[{model_name}] results -> {out_path}")

    return cosine_similarities


if __name__ == "__main__":
    main()  # Hydra will pass the config object automatically
