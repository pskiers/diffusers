"""Steering-reuse-after-step ablation: puzzle accuracy as a function of how
many denoising steps actually reason before the steering is frozen and
reused unchanged for the rest of the trajectory (experiments/
ablate_trm_loop_budget.py's "reuse" axis, reset_every=20, n_sup=16).

Shows that reasoning saturates well before the last denoising step: freezing
after step 8-10 (of 20) matches or beats reasoning at every single step,
at under half the total reasoning compute.

Data from runs/reuse_after_step_ablation.json (produced by:
python experiments/ablate_trm_loop_budget.py experiment=mnist_thinker_v1_controlnet
  painter.checkpoint=runs/mnist_unet_painter.pt condition_encoder=x0_hint_v1
  condition_encoder.threshold=80 +condition_encoder.enabled=false
  condition_encoder.inner.with_timestep_emb=false thinker.optim_cfg.weight_decay=0.3
  train.seed=3 +checkpoint=runs/sudoku-painter-thinker.pt eval.batch_size=32
  +ablation.axes=[reuse] +ablation.num_samples=256
  +ablation.reuse_steps_values=[1,2,3,4,6,8,10,12,16,20] +ablation.reuse_reset_every=20
  +ablation.out=runs/reuse_after_step_ablation.json)

Usage:
    python paper_plots/plot_reuse_after_step_ablation.py
Produces paper_plots/reuse_after_step_ablation.pdf
"""
import json

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "axes.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_LINE = "#0072B2"
COLOR_BASELINE = "#999999"

with open("runs/reuse_after_step_ablation.json") as f:
    data = json.load(f)

num_inference_steps = 20  # full trajectory length (never freezes)

points = []
for key, r in data["results"].items():
    n_reason = int(key.split("n_reason=")[1].split("/")[0])
    points.append((100.0 * n_reason / num_inference_steps, 100.0 * r["puzzle_acc"]))
points.sort()
x = [p[0] for p in points]
acc_values = [p[1] for p in points]

baseline_acc = dict(points)[100.0]

fig, ax = plt.subplots(figsize=(5.6, 3.6))

ax.axhline(baseline_acc, color=COLOR_BASELINE, linestyle="--", linewidth=1.5, zorder=2)
ax.text(
    100, baseline_acc + 2.0, "reason every step", ha="right", va="bottom",
    fontsize=17, color=COLOR_BASELINE,
)
ax.plot(x, acc_values, marker="o", markersize=9, linewidth=2.5, color=COLOR_LINE, zorder=3)

ax.set_xlim(0, 105)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_xlabel("Reusing after")
ax.set_ylabel("Puzzle accuracy (%)")
ax.set_ylim(-3, 82)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.3)
fig.savefig("paper_plots/reuse_after_step_ablation.pdf", bbox_inches="tight")
print("Wrote paper_plots/reuse_after_step_ablation.pdf")
