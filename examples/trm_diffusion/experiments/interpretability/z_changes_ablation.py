"""How does z change over recurrence in the TRM's internal states?"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets.amaze_dataset import AmazeDataset

from .base_ablation import Z_KEYS, BatchRuns, InterpretabilityAblation
from .plot_utils import cycle_boundaries, draw_boundaries


class ZChangesAblation(InterpretabilityAblation):
    """Create 4 plots that examine how the z changes over time compare to other z states. 2 plots per z_L and z_H.
    PLOT 1: Relative difference per update, ||z_{i+1} - z_i|| / ||z_i||, on the raw states (incl. the part shared by all boards)
    PLOT 2: Cosine similarity to the final recurrence state (heatmap), on the states without the template
        - X axis: tokens (n_tokens dimension of the z state)
        - Y axis: updates in the recurrence (n_updates dimension of the z state)
        - Color intensity: cosine similarity value. How similar the state of a given token at a given recurrence step is to its final (end) version.
    Both metrics are computed per board first and only then averaged over the N boards.
    """
    output_subdir = "z_changes_ablation"

    def _make_runs(self, model, ds, indices, base_seed, device):
        batch = AmazeDataset.collate_fn([ds[i] for i in indices])
        conditions = model._batch_to_sample(batch, device)
        generator = torch.Generator(device=device).manual_seed(base_seed)
        x_t = torch.randn(batch.images.shape, generator=generator, device=device)  # pure noise: the full reverse process (images = sol_img; target is None here)
        return BatchRuns(runs={"noise": (conditions, x_t, self.full_ts)})

    def _accumulate_metrics_per_batch(self, model, states, info, metrics):
        """metrics: {t: {"z_L" | "z_H": {"rel_diff_sum": (n_updates - 1,), "cos_sum": (n_updates, n_tokens), "n_boards": int}}}"""
        for t, s in states["noise"].items():
            for k in Z_KEYS:
                v_raw = s[k].float()                        # (B, n_updates, n_tokens, hidden)
                acc = metrics.setdefault(t, {}).setdefault(k, {"rel_diff_sum": 0.0, "cos_sum": 0.0, "n_boards": 0})

                # PLOT 1 - Relative difference between next updates (per board and token)
                diff = v_raw[:, 1:] - v_raw[:, :-1]                                                   # shape: (B, n_updates - 1, n_tokens, hidden)
                token_diffs = torch.norm(diff, p=2, dim=-1) / torch.norm(v_raw[:, :-1], p=2, dim=-1)  # shape: (B, n_updates - 1, n_tokens)
                del diff  # free the full-size tensor before plot 2
                acc["rel_diff_sum"] = acc["rel_diff_sum"] + token_diffs.mean(dim=2).sum(dim=0)        # mean over tokens, sum over boards

                # PLOT 2 - Cosine similarity to the final recurrence state, without the template (per board and token)
                v = v_raw - self.template[t][k]
                acc["cos_sum"] = acc["cos_sum"] + F.cosine_similarity(v, v[:, -1:], dim=-1).sum(dim=0)  # (n_updates, n_tokens)
                acc["n_boards"] += v_raw.shape[0]

    def _visualise_metrics(self, model, metrics, output_dir):
        inner_config = model.thinker.inner.config
        L_cycles = inner_config.L_cycles
        H_cycles = inner_config.H_cycles

        for t, z in metrics.items():
            for k, acc in z.items():
                rel_diff_per_update = acc["rel_diff_sum"] / acc["n_boards"]    # shape: (n_updates - 1,)
                cosine_similarities = acc["cos_sum"] / acc["n_boards"]         # shape: (n_updates, n_tokens)
                n_updates = cosine_similarities.shape[0]
                minor_boundaries, major_boundaries = cycle_boundaries(k, n_updates, L_cycles, H_cycles, self.keep_L_indices)

                # PLOT 1 - x = i is the change from update i - 1 to i, so the line at b marks the change across the boundary
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(f"Timestep {t} - {k}: Relative Difference per Update")
                ax.plot(range(1, n_updates), rel_diff_per_update.cpu().numpy(), color="tab:blue")
                draw_boundaries(ax, minor_boundaries, major_boundaries)
                if major_boundaries or minor_boundaries:
                    ax.legend()
                ax.set_xlabel("Update")
                ax.set_ylabel("||z_{i+1} - z_i|| / ||z_i||")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(output_dir / f"{t}_{k}_relative_diff.png", dpi=300)
                plt.close(fig)

                # PLOT 2 - Heatmap: X axis is tokens, Y axis is updates in the recurrence
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_title(f"Timestep {t} - {k}: Cosine Similarity to Final State per Token")
                im = ax.imshow(cosine_similarities.cpu().numpy(), aspect='auto', cmap='viridis', origin='lower')
                fig.colorbar(im, ax=ax, label="Cosine Similarity")
                # Row b spans b - 0.5 .. b + 0.5, so the line between rows b - 1 and b is at b - 0.5
                draw_boundaries(ax, minor_boundaries, major_boundaries, horizontal=True, offset=-0.5, major_color="white", label=False)
                ax.set_xlabel("Token Index")
                ax.set_ylabel("Update Index")
                fig.tight_layout()
                fig.savefig(output_dir / f"{t}_{k}_cosine_similarity.png", dpi=300)
                plt.close(fig)

        print(f"Saved z change plots to {output_dir}")
