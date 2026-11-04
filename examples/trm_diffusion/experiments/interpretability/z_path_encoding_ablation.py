"""Does z represent the solution of the board from the conditioning? (Point 3)

Train a probe that classifies every z_L and z_H token as path / not path, then draw the z states in which its
predictions change.
"""
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets.amaze_dataset import AmazeDataset
from experiments import maze_corruption_lib as mcl
from matplotlib.patches import Rectangle
from PIL import Image

from .base_ablation import Z_KEYS, BatchRuns, InterpretabilityAblation
from .plot_utils import draw_boundaries
from .z_representation_ablation import _blue

PROBE_KINDS = ("mlp", "linear")


class TokenPathClassifier(nn.Module):
    """Per-token probe: (..., hidden) -> (...,) logits of "this token is on the solution path".

    use_nonlinearity=False: a linear probe (logistic regression); True: an MLP with one hidden layer.
    The input is standardized with the training data's mean / std (set_normalization).
    """

    def __init__(self, hidden_size=512, mlp_dim=128, use_nonlinearity=True):
        super().__init__()
        if use_nonlinearity:
            self.net = nn.Sequential(nn.Linear(hidden_size, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, 1))
        else:
            self.net = nn.Linear(hidden_size, 1)
        self.register_buffer("mean", torch.zeros(hidden_size))
        self.register_buffer("std", torch.ones(hidden_size))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))

    def forward(self, x):
        # Raw logits: train with BCEWithLogitsLoss (stable), torch.sigmoid only to read probabilities
        return self.net((x - self.mean) / self.std).squeeze(-1)


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC AUC of scores (n,) for 0/1 labels (n,): the probability that a random path token scores above a random
    not-path token (Mann-Whitney U; ties are rare for continuous scores)."""
    scores = scores.double()
    ranks = torch.empty_like(scores)
    ranks[scores.argsort()] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    n_pos = labels.sum().double()
    n_neg = len(labels) - n_pos
    return ((ranks[labels.bool()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def _call_to_update(call: int, L_cycles: int) -> tuple:
    """Call index of L_level (per H-cycle: L_cycles z_L calls, then one z_H call) -> (z_L / z_H, update index)."""
    per_cycle = L_cycles + 1
    if call % per_cycle == L_cycles:
        return "z_H", call // per_cycle
    return "z_L", call - call // per_cycle


class ZPathEncodingAblation(InterpretabilityAblation):
    """Does z represent the solution of the board from the conditioning?

    Runs: the empty board (the maze, nothing drawn) noised at t_start, so the path can't be read from the drawing in x_t.
    Labels: per token, from the ground-truth solution image (sol_img): 1 = path (>= label_threshold of the token's pixels
            are path), 0 = no path pixel at all, -1 = ambiguous (the path only grazes the token), left out of training and AUC.
    Fit phase (boards [0, K)): per recorded t and z_L / z_H, an MLP and a linear probe are trained on the LAST update.
    Metric phase (boards [K, K + N), held out): both probes are applied to every update; the path probabilities are kept.

    Plots, one set per t:
        probe_auc_flips_t{t}.png:   AUC per call (MLP, linear, position-only baseline) and how many token predictions flip
        probe_calls_t{t}.png:       the decoded path probability after selected calls, for a few boards
        probe_corrections_t{t}.png: confident self-corrections in reasoning step 1 (dropped: p > 0.8, then < 0.2 at its end)
    """
    output_subdir = "z_path_encoding_ablation"

    def _make_runs(self, model, ds, indices, base_seed, device):
        if self.keep_L_indices is not None:
            raise SystemExit("ZPathEncodingAblation needs every z_L update: unset keep_L_indices")
        cfg = self.interprebability_config
        t_start = int(cfg.get("t_start", 70))
        if t_start not in {int(s) for s in self.full_ts}:
            raise SystemExit(f"t_start {t_start} not in the denoising timesteps {self.full_ts.tolist()}")

        samples, empty, boards = [], [], []
        for i in indices:
            sample = ds[i]
            ms = mcl.build_maze_sample(sample.metadata)
            if ms is None:  # build_maze_sample couldn't parse the board
                continue
            samples.append(sample)
            empty.append(ds.transform(Image.fromarray(ms.morig)))
            boards.append(i)
        if not samples:
            return None

        batch = AmazeDataset.collate_fn(samples)
        conditions = model._batch_to_sample(batch, device)
        empty = torch.stack(empty).to(device)
        generator = torch.Generator(device=device).manual_seed(base_seed)
        noise = torch.randn(empty.shape, generator=generator, device=device)
        timesteps = torch.full((len(samples),), t_start, device=device, dtype=torch.long)
        x_t = model.scheduler.add_noise(empty, noise, timesteps)

        grid = int(round(model.thinker.inner.config.seq_len ** 0.5))  # 12x12 tokens, row-major
        info = {
            "labels": self._token_labels(batch.images, grid, float(cfg.get("label_threshold", 0.1))),  # images = sol_img
            "solutions": batch.images.cpu(),
            "boards": boards,
        }
        return BatchRuns(runs={"empty": (conditions, x_t, self.full_ts[self.full_ts <= t_start])}, info=info)

    @staticmethod
    def _token_labels(solutions: torch.Tensor, grid: int, threshold: float) -> torch.Tensor:
        """(B, C, H, W) solution images in [0, 1] -> (B, grid * grid) labels: 1 path, 0 not path, -1 ambiguous."""
        path = _blue(solutions).float()  # (B, H, W)
        B, H, W = path.shape
        frac = path.reshape(B, grid, H // grid, grid, W // grid).mean(dim=(2, 4)).reshape(B, -1)
        labels = torch.full_like(frac, -1.0)
        labels[frac >= threshold] = 1.0
        labels[frac == 0] = 0.0
        return labels

    # ── Fit phase: train the probes ─────────────────────────────────────────

    def _accumulate_fit(self, states, info, acc):
        """Training data: the last update of z_L / z_H at every recorded t, with the token labels."""
        acc.setdefault("labels", []).append(info["labels"])
        for t, s in states["empty"].items():
            for k in Z_KEYS:
                acc.setdefault((t, k), []).append(s[k][:, -1].float())  # (B, n_tokens, hidden)

    def _finish_fit(self, acc, device):
        self.device = device
        steps = int(self.interprebability_config.get("probe_steps", 3000))
        labels = torch.cat(acc.pop("labels"))  # (K, n_tokens)
        logging.info(
            f"[probe] fit boards: {labels.shape[0]}; tokens path {(labels == 1).float().mean():.3f}, "
            f"not path {(labels == 0).float().mean():.3f}, ambiguous (left out) {(labels == -1).float().mean():.3f}"
        )

        # Position-only baseline: P(path | token index) on the fit boards
        known = labels >= 0
        self.position_prior = ((labels == 1).float().sum(0) / known.float().sum(0).clamp_min(1)).cpu()  # (n_tokens,)

        self.probes, self.fit_report = {}, {}
        y = labels.flatten()
        keep = y >= 0
        for (t, k), xs in acc.items():
            x = torch.cat(xs).flatten(0, 1)[keep]  # (n_known_tokens, hidden)
            for kind in PROBE_KINDS:
                probe = self._train_probe(x, y[keep], use_nonlinearity=(kind == "mlp"), steps=steps, device=device)
                with torch.no_grad():
                    train_auc = _auc(probe(x.to(device)).cpu(), y[keep])
                self.probes[(t, k, kind)] = probe
                self.fit_report[f"t{t}_{k}_{kind}_train_auc"] = train_auc
                logging.info(f"[probe] t={t} {k} {kind}: train AUC {train_auc:.4f}")

    @staticmethod
    def _train_probe(x, y, use_nonlinearity, steps, device, lr=3e-3, weight_decay=1e-2, batch_size=256):
        """A fixed number of optimizer steps, whatever the data size: with too few steps the linear probe stays near its
        random start, and "linear vs MLP" would compare training time instead of what z encodes."""
        torch.manual_seed(0)
        probe = TokenPathClassifier(x.shape[-1], use_nonlinearity=use_nonlinearity).to(device)
        probe.set_normalization(x.mean(0).to(device), x.std(0).to(device))
        pos_weight = ((y == 0).sum() / (y == 1).sum().clamp_min(1)).to(device)  # path tokens are a minority
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        x, y = x.to(device), y.to(device)
        generator = torch.Generator(device=device).manual_seed(0)
        probe.train()
        for _ in range(steps):
            idx = torch.randint(len(x), (batch_size,), device=device, generator=generator)
            optimizer.zero_grad()
            criterion(probe(x[idx]), y[idx]).backward()
            optimizer.step()
        probe.eval()
        return probe

    # ── Metric phase: apply the probes to every update of the held-out boards ──

    def _accumulate_metrics_per_batch(self, model, states, info, metrics):
        """metrics: {"labels": [(B, n_tokens)], "solutions": [(B, C, H, W)], "boards": [int],
                     "probs": {(t, k, kind): [(B, n_updates, n_tokens) float16]}, "position_prior", "fit_report"}"""
        if "labels" not in metrics:
            metrics.update(labels=[], solutions=[], boards=[], probs={},
                           position_prior=self.position_prior, fit_report=self.fit_report)
        metrics["labels"].append(info["labels"])
        metrics["solutions"].append(info["solutions"])
        metrics["boards"] += info["boards"]
        with torch.no_grad():
            for t, s in states["empty"].items():
                for k in Z_KEYS:
                    for kind in PROBE_KINDS:
                        probe = self.probes[(t, k, kind)]
                        p = torch.cat([torch.sigmoid(probe(z.to(self.device))).cpu() for z in s[k].split(4)])  # 4 boards at a time
                        metrics["probs"].setdefault((t, k, kind), []).append(p.half())

    # ── Plots ──────────────────────────────────────────────────────────────

    def _visualise_metrics(self, model, metrics, output_dir):
        inner_config = model.thinker.inner.config
        L_cycles, H_cycles = inner_config.L_cycles, inner_config.H_cycles
        cfg = self.interprebability_config
        labels = torch.cat(metrics["labels"])          # (N, n_tokens)
        solutions = torch.cat(metrics["solutions"])    # (N, C, H, W)
        boards = metrics["boards"]
        probs = {key: torch.cat(v).float() for key, v in metrics["probs"].items()}  # (N, n_updates, n_tokens)
        record_at = sorted({t for t, _, _ in probs}, reverse=True)
        grid = int(round(labels.shape[1] ** 0.5))

        known = labels >= 0
        y = labels[known]
        position_auc = _auc(metrics["position_prior"].expand_as(labels)[known], y)
        summary = {"position_only_auc": position_auc, "n_boards": len(boards), **metrics["fit_report"]}

        for t in record_at:
            # The two series (z_L, z_H) on one axis: the call index of L_level
            series = {}
            for k in Z_KEYS:
                p = probs[(t, k, "mlp")]
                n_updates = p.shape[1]
                calls = [u + u // L_cycles if k == "z_L" else (u + 1) * (L_cycles + 1) - 1 for u in range(n_updates)]
                auc = {kind: [_auc(probs[(t, k, kind)][:, u][known], y) for u in range(n_updates)] for kind in PROBE_KINDS}
                pred = p > 0.5
                flips = (pred[:, 1:] != pred[:, :-1]).float().mean(dim=(0, 2))  # fraction of tokens whose prediction flips
                series[k] = {"calls": calls, "auc": auc, "flips": flips}
                for kind in PROBE_KINDS:
                    summary[f"t{t}_{k}_{kind}_test_auc_last_update"] = auc[kind][-1]
            n_calls = max(series["z_H"]["calls"]) + 1
            self._plot_auc_flips(t, series, position_auc, n_calls, L_cycles, H_cycles, output_dir)

            default_calls = [0, 1, 2, 5, 6, 8, 12, 13, 19, 20, 41, 83, 167, n_calls - 1]
            calls = [int(c) for c in cfg.get("probe_calls", default_calls) if int(c) < n_calls]
            n_examples = int(cfg.get("n_example_boards", 2))
            self._plot_calls(t, probs, calls, solutions, boards, n_examples, grid, L_cycles, output_dir)

            window = int(cfg.get("correction_window", H_cycles * L_cycles))  # z_L updates of reasoning step 1
            summary[f"t{t}_corrections"] = self._plot_corrections(
                t, probs[(t, "z_L", "mlp")], labels, solutions, boards, window, grid, output_dir
            )

        with open(output_dir / "probe_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"[probe] summary: {json.dumps(summary)}")
        print(f"Saved path probe plots to {output_dir}")

    @staticmethod
    def _plot_auc_flips(t, series, position_auc, n_calls, L_cycles, H_cycles, output_dir):
        per_step = H_cycles * (L_cycles + 1)  # calls per reasoning step
        minor = [c for c in range(L_cycles + 1, n_calls, L_cycles + 1) if c % per_step]
        major = list(range(per_step, n_calls, per_step))
        colors = {"z_L": "tab:blue", "z_H": "tab:orange"}

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for k, s in series.items():
            axes[0].plot(s["calls"], s["auc"]["mlp"], color=colors[k], label=f"{k} MLP", marker="." if k == "z_H" else None)
            axes[0].plot(s["calls"], s["auc"]["linear"], color=colors[k], linestyle=":", label=f"{k} linear")
            axes[1].plot(s["calls"][1:], s["flips"].numpy(), color=colors[k], label=k, marker="." if k == "z_H" else None)
        axes[0].axhline(position_auc, color="gray", linestyle="--", label="position only")
        axes[0].set_ylabel("AUC (held-out boards)")
        axes[0].set_title(f"t={t}: path / not path probe, trained on the last update, applied to every call")
        axes[1].set_ylabel("tokens whose prediction flips\n(p > 0.5, vs the previous update)")
        axes[1].set_xlabel("L_level call")
        for ax in axes:
            draw_boundaries(ax, minor, major, offset=-0.5, label=False)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(output_dir / f"probe_auc_flips_t{t}.png", dpi=200)
        plt.close(fig)

    @staticmethod
    def _plot_calls(t, probs, calls, solutions, boards, n_examples, grid, L_cycles, output_dir):
        """Like the earlier figure: per board, the GT solution and the decoded path probability after selected calls."""
        n_examples = min(n_examples, len(boards))
        fig, axes = plt.subplots(n_examples, len(calls) + 1, figsize=(1.6 * (len(calls) + 1), 1.75 * n_examples), squeeze=False)
        for r in range(n_examples):
            axes[r][0].imshow(solutions[r].permute(1, 2, 0).clamp(0, 1).numpy())
            axes[r][0].set_title(f"board {boards[r]}: GT", fontsize=7)
            for c, call in enumerate(calls, start=1):
                k, u = _call_to_update(call, L_cycles)
                axes[r][c].imshow(probs[(t, k, "mlp")][r, u].reshape(grid, grid).numpy(), cmap="Blues", vmin=0, vmax=1)
                axes[r][c].set_title(f"call {call} ({k})", fontsize=7)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"t={t}: decoded path probability (MLP probe) after selected L_level calls", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"probe_calls_t{t}.png", dpi=200)
        plt.close(fig)

    @staticmethod
    def _plot_corrections(t, p, labels, solutions, boards, window, grid, output_dir, n_boards=8) -> dict:
        """Confident self-corrections in the first `window` z_L updates (reasoning step 1). "Confident" means for at least
        2 updates in a row, so that a one-update spike of the probe doesn't count:
            dropped: p > 0.8 at 2 consecutive updates, < 0.2 at the last 2 updates of the window; drawn as an orange box,
                     and at the bottom coloured by the z_L update of the drop (the last crossing below 0.2)
            added late: p < 0.2 at the first 2 updates, > 0.8 at the last 2, first > 0.8 (twice in a row) in the window's
                        second half; green
        p: (N, n_updates, n_tokens) MLP path probabilities of z_L. Returns summary numbers.
        """
        w = p[:, :window]
        high2 = (w[:, 1:] > 0.8) & (w[:, :-1] > 0.8)  # (N, window - 1, n_tokens): confident path at updates u - 1 and u
        dropped = high2.any(dim=1) & (w[:, -2:] < 0.2).all(dim=1)  # (N, n_tokens)
        first_high = torch.where(high2.any(dim=1), high2.float().argmax(dim=1) + 1, torch.full_like(w[:, 0], window).long())
        added = (w[:, :2] < 0.2).all(dim=1) & (w[:, -2:] > 0.8).all(dim=1) & (first_high >= window // 2)
        crossed = (w[:, :-1] >= 0.2) & (w[:, 1:] < 0.2)  # (N, window - 1, n_tokens)
        steps = torch.arange(1, window).view(1, -1, 1)
        drop_update = torch.where(crossed, steps, torch.zeros_like(steps)).amax(dim=1)  # last crossing, 0 = none

        order = dropped.sum(1).argsort(descending=True)[: min(n_boards, len(boards))]
        cell = solutions.shape[-1] / grid
        fig, axes = plt.subplots(2, len(order), figsize=(2.3 * len(order), 4.8), squeeze=False)
        for c, b in enumerate(order.tolist()):
            ax = axes[0][c]
            ax.imshow(solutions[b].permute(1, 2, 0).clamp(0, 1).numpy())
            for tok in dropped[b].nonzero().flatten().tolist():
                ax.add_patch(Rectangle(((tok % grid) * cell, (tok // grid) * cell), cell, cell, fill=False, edgecolor="tab:orange", linewidth=1.2))
            for tok in added[b].nonzero().flatten().tolist():
                ax.add_patch(Rectangle(((tok % grid) * cell + cell / 4, (tok // grid) * cell + cell / 4), cell / 2, cell / 2, color="tab:green"))
            ax.set_title(f"board {boards[b]}: {int(dropped[b].sum())} dropped", fontsize=8)

            ax = axes[1][c]
            ax.imshow((labels[b] == 1).float().reshape(grid, grid).numpy(), cmap="Blues", vmin=0, vmax=4)  # light blue: GT path
            drop_map = np.ma.masked_where(~dropped[b].reshape(grid, grid).numpy(), drop_update[b].reshape(grid, grid).numpy())
            im = ax.imshow(drop_map, cmap="viridis", vmin=1, vmax=window - 1)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[1][0].set_ylabel(f"z_L update of the drop\n(dark = 1 ... yellow = {window - 1})", fontsize=7)
        fig.colorbar(im, ax=axes[1].tolist(), fraction=0.02)
        fig.suptitle(f"t={t}: confident self-corrections in the first {window} z_L updates. "
                     "Orange box: dropped (p > 0.8, then < 0.2); green: added late", fontsize=9)
        fig.savefig(output_dir / f"probe_corrections_t{t}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        return {"dropped_per_board": dropped.sum(1).float().mean().item(), "added_late_per_board": added.sum(1).float().mean().item(),
                "dropped_on_gt_path_frac": (dropped & (labels == 1)).sum().item() / max(int(dropped.sum()), 1)}
