"""What does z represent in the TRM's internal states?"""
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets.amaze_dataset import AmazeDataset
from experiments import maze_corruption_lib as mcl
from PIL import Image

from .base_ablation import Z_KEYS, BatchRuns, InterpretabilityAblation
from .plot_utils import cycle_boundaries, draw_boundaries

SETS = ("empty", "half", "corrupted")
PAIRS = (("empty", "half"), ("half", "corrupted"), ("empty", "corrupted"))  # (reference a, other b)
GROUPS = ("changed", "drawn_both", "rest")
GROUP_COLORS = {"changed": "tab:red", "drawn_both": "tab:blue", "rest": "tab:gray"}


def _token_any(mask: torch.Tensor, grid: int) -> torch.Tensor:
    """(B, H, W) bool pixel mask -> (B, grid * grid) bool: is any pixel of the token's patch set (tokens row-major, like the thinker's encoder)."""
    B, H, W = mask.shape
    return mask.reshape(B, grid, H // grid, grid, W // grid).any(dim=4).any(dim=2).reshape(B, -1).cpu()


def _blue(images: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) images in [0, 1] -> (B, H, W) bool: the drawn (blue) path pixels."""
    return torch.from_numpy(np.stack([
        mcl.blue_mask((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for img in images
    ]))


class ZRepresentationAblation(InterpretabilityAblation):
    """What does z represent in the TRM's internal states?

    The same boards are run from three versions of the drawn input:
        empty:     the maze with no path drawn
        half:      the first prefix_frac of the solution, plus the correct cells apply_add walked before turning off
        corrupted: half + a wrong branch to a dead-end (mcl.apply_add)
    The drawing goes into x_t (the image being denoised, which the thinker's encoder reads together with the maze);
    the maze conditioning (spatial_conditions) is the same in all three. All three start from the same noise at t_start
    and use the same step noise, so the differences between them come from the drawing only.

    For every pair (reference a, other b), t and z_L / z_H, per board and token, then averaged per token group:
        rel: ||z_b - z_a|| / ||z_a||    on the raw states
        cos: cos(z_a - T, z_b - T)       without the shared template T (mean over the template boards and all three sets)
    Token groups per pair: changed (the two inputs differ in the token's patch), drawn_both (path drawn in both, unchanged), rest.
    """
    output_subdir = "z_representation_ablation"

    def _make_runs(self, model, ds, indices, base_seed, device):
        """Mirrors amaze_interpretability._run_one_tstart: the boards without a dead-end are skipped in all three sets."""
        t_start = int(self.interprebability_config.get("t_start", 70))
        prefix_frac = float(self.interprebability_config.get("prefix_frac", 0.5))
        if t_start not in {int(s) for s in self.full_ts}:
            raise SystemExit(f"t_start {t_start} not in the denoising timesteps {self.full_ts.tolist()}")

        samples, images = [], {name: [] for name in SETS}
        for i in indices:
            sample = ds[i]
            ms = mcl.build_maze_sample(sample.metadata)
            if ms is None:  # build_maze_sample couldn't parse the board
                continue

            partial_board, shown_cells = mcl.render_partial(ms, prefix_frac)
            rng = random.Random(base_seed + i * 131)
            corrupted, wrong_cells, gt_walked, _ = mcl.apply_add(partial_board, ms, shown_cells, rng)
            if not wrong_cells:  # walk reached the goal, no dead-end -> skip board
                continue

            # The half-solved board: the prefix plus the correct cells the walk followed before it turned off (as the clean twin there)
            half = partial_board.copy()
            full = [shown_cells[-1]] + gt_walked
            for a, b in zip(full[:-1], full[1:]):
                cv2.line(half, mcl._center(ms.id2rc[a], ms.cell_px), mcl._center(ms.id2rc[b], ms.cell_px), ms.blue_color, ms.stroke)

            samples.append(sample)
            for name, img in (("empty", ms.morig), ("half", half), ("corrupted", corrupted)):
                images[name].append(ds.transform(Image.fromarray(img)))

        if not samples:
            return None

        B = len(samples)
        conditions = model._batch_to_sample(AmazeDataset.collate_fn(samples), device)
        images = {name: torch.stack(imgs).to(device) for name, imgs in images.items()}  # (B, C, H, W) in [0, 1]

        generator = torch.Generator(device=device).manual_seed(base_seed)
        noise = torch.randn(images["empty"].shape, generator=generator, device=device)  # the same noise for all three sets
        timesteps = torch.full((B,), t_start, device=device, dtype=torch.long)
        run_timesteps = self.full_ts[self.full_ts <= t_start]
        runs = {name: (conditions, model.scheduler.add_noise(img, noise, timesteps), run_timesteps) for name, img in images.items()}

        grid = int(round(model.thinker.inner.config.seq_len ** 0.5))  # 12x12 tokens, row-major
        info = {
            "masks": {pair: self._token_groups(images[pair[0]], images[pair[1]], grid) for pair in PAIRS},
            "images": {name: img[0].cpu() for name, img in images.items()},  # the first board, for the example maps
        }
        return BatchRuns(runs=runs, info=info)

    @staticmethod
    def _token_groups(img_a: torch.Tensor, img_b: torch.Tensor, grid: int) -> dict:
        """Split the tokens of every board into 3 groups for one pair of input images (B, C, H, W):
            changed:    a pixel of the token's patch differs between the two images (the dead-end, or the drawn path)
            drawn_both: path is drawn in the token's patch in both images, and nothing there changed
            rest:       everything else

        Returns:
            dict: {group: (B, grid * grid) float mask}
        """
        changed = _token_any((img_a - img_b).abs().amax(dim=1) > 0.05, grid)
        drawn_both = _token_any(_blue(img_a), grid) & _token_any(_blue(img_b), grid) & ~changed
        rest = ~changed & ~drawn_both
        return {"changed": changed.float(), "drawn_both": drawn_both.float(), "rest": rest.float()}

    def _accumulate_metrics_per_batch(self, model, states, info, metrics):
        """metrics:
            {pair: {t: {"x_diff_sum": float, "n_boards": int,
                        "z_L" | "z_H": {group: {"rel_sum": (n_updates,), "cos_sum": (n_updates,), "count": float}}}},
             "example": {"images": {set: (C, H, W)}, "changed": {pair: (n_tokens,)}, "maps": {(pair, t, k): {"rel" | "cos": (n_tokens,)}}}}
        """
        first_batch = "example" not in metrics
        if first_batch:  # one example board (the first of the first batch) for the maps
            metrics["example"] = {"images": info["images"], "changed": {p: info["masks"][p]["changed"][0] for p in PAIRS}, "maps": {}}

        for pair in PAIRS:
            ref, other = pair
            masks = info["masks"][pair]
            for t in states[ref]:
                s_a, s_b = states[ref][t], states[other][t]
                acc_t = metrics.setdefault(pair, {}).setdefault(t, {"x_diff_sum": 0.0, "n_boards": 0})

                # How different the two inputs still are at t: the model may already have redrawn the drawing
                x_a, x_b = s_a["x"].flatten(1), s_b["x"].flatten(1)
                acc_t["x_diff_sum"] += (torch.norm(x_b - x_a, dim=1) / torch.norm(x_a, dim=1)).sum().item()
                acc_t["n_boards"] += x_a.shape[0]

                for k in Z_KEYS:
                    z_a, z_b = s_a[k].float(), s_b[k].float()  # (B, n_updates, n_tokens, hidden)
                    assert z_a.shape[2] == masks["changed"].shape[1], f"{z_a.shape[2]} z tokens vs {masks['changed'].shape[1]} mask tokens"

                    rel = torch.norm(z_b - z_a, p=2, dim=-1) / torch.norm(z_a, p=2, dim=-1)  # (B, n_updates, n_tokens), raw states
                    template = self.template[t][k]
                    cos = F.cosine_similarity(z_a - template, z_b - template, dim=-1)       # (B, n_updates, n_tokens), without the template

                    acc_k = acc_t.setdefault(k, {g: {"rel_sum": 0.0, "cos_sum": 0.0, "count": 0.0} for g in GROUPS})
                    for g in GROUPS:
                        m = masks[g][:, None, :]  # (B, 1, n_tokens)
                        acc_k[g]["rel_sum"] = acc_k[g]["rel_sum"] + (rel * m).sum(dim=(0, 2))  # (n_updates,)
                        acc_k[g]["cos_sum"] = acc_k[g]["cos_sum"] + (cos * m).sum(dim=(0, 2))
                        acc_k[g]["count"] += m.sum().item()  # (board, token) pairs in the group
                    if first_batch:
                        metrics["example"]["maps"][(pair, t, k)] = {"rel": rel[0, -1].clone(), "cos": cos[0, -1].clone()}  # last update

    def _visualise_metrics(self, model, metrics, output_dir):
        """4 grids (rel / cos x z_L / z_H): rows = pairs, columns = t; one line per token group, over the updates."""
        inner_config = model.thinker.inner.config
        record_at = sorted(metrics[PAIRS[0]].keys(), reverse=True)  # t_start first
        ylabels = {"rel": "||z_b - z_a|| / ||z_a||", "cos": "cos(z_a - T, z_b - T)"}

        for k in Z_KEYS:
            for metric, ylabel in ylabels.items():
                fig, axes = plt.subplots(len(PAIRS), len(record_at), figsize=(5 * len(record_at), 3.5 * len(PAIRS)),
                                         squeeze=False, sharex=True, sharey=True)
                for r, pair in enumerate(PAIRS):
                    for c, t in enumerate(record_at):
                        ax = axes[r][c]
                        acc_t = metrics[pair][t]
                        n_updates = 0
                        for g in GROUPS:
                            acc_g = acc_t[k][g]
                            if acc_g["count"] == 0:  # f.ex. no token has path drawn in both inputs of (empty, half)
                                continue
                            curve = acc_g[f"{metric}_sum"] / acc_g["count"]
                            n_updates = curve.shape[0]
                            ax.plot(range(n_updates), curve.cpu().numpy(), color=GROUP_COLORS[g], label=g)

                        # The value at update u compares the two runs at the same update, so the line sits between u = b - 1 and b
                        minor_boundaries, major_boundaries = cycle_boundaries(
                            k, n_updates, inner_config.L_cycles, inner_config.H_cycles, self.keep_L_indices
                        )
                        draw_boundaries(ax, minor_boundaries, major_boundaries, offset=-0.5, label=False)
                        ax.set_title(f"{pair[1]} vs {pair[0]}, t={t} (input diff {acc_t['x_diff_sum'] / acc_t['n_boards']:.3f})", fontsize=9)
                        ax.grid(True, alpha=0.3)
                        if r == len(PAIRS) - 1:
                            ax.set_xlabel("Update")
                        if c == 0:
                            ax.set_ylabel(ylabel)
                axes[0][0].legend(fontsize=8)
                fig.suptitle(f"{k}: {ylabel}, per token group (mean over boards and the group's tokens)")
                fig.tight_layout()
                fig.savefig(output_dir / f"{metric}_{k}.png", dpi=200)
                plt.close(fig)

        self._plot_example(metrics["example"], record_at, output_dir)
        print(f"Saved z representation plots to {output_dir}")

    @staticmethod
    def _plot_example(example: dict, record_at: list, output_dir):
        """One board: per pair its two inputs, and ||z_b - z_a|| / ||z_a|| per token at the last update (grid maps).
        The white outline marks the changed tokens."""
        n_cols = 2 + len(record_at) * len(Z_KEYS)
        fig, axes = plt.subplots(len(PAIRS), n_cols, figsize=(2.6 * n_cols, 2.6 * len(PAIRS)), squeeze=False)
        for r, pair in enumerate(PAIRS):
            for c, name in enumerate(pair):
                axes[r][c].imshow(example["images"][name].permute(1, 2, 0).clamp(0, 1).numpy())
                axes[r][c].set_title(name, fontsize=9)

            changed = example["changed"][pair]
            grid = int(round(changed.numel() ** 0.5))
            c = 2
            for t in record_at:
                for k in Z_KEYS:
                    ax = axes[r][c]
                    im = ax.imshow(example["maps"][(pair, t, k)]["rel"].reshape(grid, grid).numpy(), cmap="magma")
                    if changed.any():
                        ax.contour(changed.reshape(grid, grid).numpy(), levels=[0.5], colors="white", linewidths=0.8)
                    ax.set_title(f"{pair[1]} vs {pair[0]}: {k}, t={t}", fontsize=7)
                    fig.colorbar(im, ax=ax, fraction=0.046)
                    c += 1

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Example board: ||z_b - z_a|| / ||z_a|| per token at the last update (white outline: changed tokens)")
        fig.tight_layout()
        fig.savefig(output_dir / "example_maps.png", dpi=200)
        plt.close(fig)
