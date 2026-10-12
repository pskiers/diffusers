"""Main results table: Maze & Queens, PaTh/DM vs. proprietary and open-source
image editors.

Most baseline rows (proprietary, base open-source, non-fine-tuned CoT) are
transcribed verbatim from the AMAZE benchmark paper's Table 1 (values only --
the paper's per-cell heatmap coloring is dropped in favor of a plain,
booktabs-style table matching this repo's other paper_plots/*_tables.tex
output). The paper doesn't report Exact@1/Exact@5 at all, so those two cells
are "--" for every transcribed row.

Four rows are instead our own fine-tuning + eval runs on the same benchmark
split:
  - DM, PaTh (ours)                     <- runs/amaze_metrics/{maze,queens}/{dm,path}/metrics*.json
  - Bagel (fine-tuned), no CoT          <- runs/amaze_metrics/{maze,queens}/bagel/metrics*.json
  - Janus-Pro (fine-tuned), no CoT      <- runs/amaze_metrics/{maze,queens}/janus/metrics*.json
  - Bagel (fine-tuned)+CoT              <- runs/amaze_metrics_{maze,queens}_bagel_ft_cot.json
  - Janus-Pro (fine-tuned)+CoT          <- runs/amaze_metrics_{maze,queens}_janus_ft_cot.json
(The runs/amaze_metrics/{maze,queens}/{bagel,janus}/ directory holds the
*non*-CoT fine-tuned reruns -- confirmed directly by the user after an
earlier version of this table had bagel/janus's CoT and non-CoT reruns
swapped; fixed by renaming the flat amaze_metrics_{maze,queens}_bagel_ft*.json
files rather than guessing again.)

Two on-disk schemas are mixed across these files -- {maze,queens}/{dm,janus}
and the flat _ft_cot.json files use "mse_inside"/"pass1"/"exact1";
{maze,queens}/{bagel,path} use the abbreviated "mse_in"/"pass@1"/"exact@1"
-- so _extract_overall() detects which is present per file rather than
assuming one.

Layout/styling (table position, font size, row-group color, merged group
blocks with no separating \\midrule between a group's base and fine-tuned
sub-rows, \\ours macro for the row label, params shown without the
active/total qualifier, one-sentence caption) matches the version the user
hand-edited directly in the .tex output -- this script reproduces that
structure so it can still be regenerated from data.

Bold = best value in that column across all rows (matches the paper table's
own convention, which is global-best-per-column, not per-section -- verified
against the source values), skipping "--" cells.

MSE-In/MSE-Out are loaded (needed for other consumers of this data) but
dropped from the displayed table via DROP_COLS, per the user's request to
leave them out of this table for now.

Usage:
    python paper_plots/plot_maze_queens_main_table.py
Produces paper_plots/maze_queens_main_table.tex
"""
import json

# Column order per task, after loading: Violation, Coverage, MSE-In, MSE-Out,
# Pass@1, Pass@5, Exact@1, Exact@5 (Maze = cols 0-7, Queens = cols 8-15).
# MSE-In/MSE-Out (indices 2,3 and 10,11) are dropped from display below --
# DROP_COLS -- leaving Violation, Coverage, Pass@1, Pass@5, Exact@1, Exact@5
# per task (Maze = cols 0-5, Queens = cols 6-11) for the header/bold logic.
DROP_COLS = {2, 3, 10, 11}
MIN_BETTER = {0, 6}
MAX_BETTER = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11}

# ── Baseline rows, transcribed from the AMAZE paper's Table 1 (%) ──────────
# Each row: (label, group, params, [maze V,C,MSEin,MSEout,P1,P5, queens V,C,MSEin,MSEout,P1,P5])
# -- Exact@1/Exact@5 are appended as [None, None] below since the paper
# doesn't report them; group in {"prop", "open", "cot"}; a new \rowcolor
# section header is emitted only the first time a group is seen.
#
# Params are web-verified, not from the AMAZE paper (which doesn't report
# them): proprietary models (GPT-image-1, NanoBanana-Pro/Gemini-3-Pro-Image,
# Seedream-4.5) don't have an officially disclosed count -> N/A. Open models:
# FLUX.1 Kontext [dev] 12B (Black Forest Labs); Qwen-Image-Edit 20B DiT
# (+8.3B Qwen2.5-VL text encoder, not counted here); BAGEL-7B-MoT is a
# Mixture-of-Transformer-Experts model (14B total / 7B active, shown here as
# just "14B"); Janus-Pro assumed to be the 7B variant (the paper doesn't say
# which of the 1B/7B releases was used).
BASELINE_ROWS = [
    ("GPT-image-1", "prop", "N/A", [62.88, 58.97, 41.16, 52.76, 5.40, 6.06, 62.91, 37.09, 11.84, 5.87, 0.00, 2.28]),
    ("NanoBanana-Pro", "prop", "N/A", [47.76, 64.21, 24.20, 17.21, 4.82, 9.28, 32.56, 67.43, 9.10, 1.62, 30.35, 35.58]),
    ("Seedream-4.5", "prop", "N/A", [16.90, 25.67, 28.82, 30.96, 2.14, 3.21, 76.86, 23.14, 11.55, 5.95, 2.86, 2.86]),
    ("Flux-Kontext-Dev", "open", "12B", [23.84, 30.24, 30.96, 18.31, 0.36, 3.57, 78.63, 21.37, 11.48, 7.71, 0.92, 2.34]),
    ("Qwen-Image-Edit", "open", "20B", [19.37, 28.51, 18.82, 5.70, 1.43, 2.14, 69.52, 30.47, 8.83, 5.30, 2.86, 4.00]),
    ("Bagel", "open", "14B", [28.91, 27.15, 11.64, 5.84, 0.00, 1.00, 61.57, 38.43, 8.94, 1.22, 0.00, 0.00]),
    ("Janus-Pro", "open", "7B", [5.41, 1.85, 57.47, 76.80, 0.00, 0.00, 84.24, 15.76, 12.97, 9.83, 0.00, 0.57]),
    ("Bagel (fine-tuned)", "open", "14B", None),  # filled in below from our own runs
    ("Janus-Pro (fine-tuned)", "open", "7B", None),  # filled in below from our own runs
    ("Bagel", "cot", "14B", [34.06, 30.31, 14.77, 3.97, 0.00, 0.57, 98.41, 1.59, 9.63, 1.40, 0.00, 0.00]),
    ("Bagel (fine-tuned)", "cot", "14B", None),  # filled in below from our own runs
    ("Janus-Pro", "cot", "7B", [6.03, 0.89, 53.02, 73.98, 0.00, 0.00, 82.91, 17.09, 10.93, 8.04, 0.00, 0.70]),
    ("Janus-Pro (fine-tuned)", "cot", "7B", None),  # filled in below from our own runs
]
# Pad transcribed 12-value rows (no Exact@1/Exact@5) to 16 with None -> "--".
BASELINE_ROWS = [
    (label, group, params, (values[:6] + [None, None] + values[6:] + [None, None]) if values is not None else None)
    for (label, group, params, values) in BASELINE_ROWS
]

GROUP_TITLES = {
    "prop": "proprietary models",
    "open": "open-source models (w/o chain-of-thought reasoning)",
    "cot": "w/ chain-of-thought reasoning",
    "ours": "ours (diffusion models)",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _extract_overall(d):
    """Auto-detects the on-disk schema (dotted "mse_inside"/"pass1"/"exact1"
    vs. abbreviated "mse_in"/"pass@1"/"exact@1") and returns the 8 raw
    (0-1 scale) values in [V, C, MSEin, MSEout, P1, P5, E1, E5] order."""
    o = d["overall"]
    if "pass@1" in o:
        return [o["violation"], o["coverage"], o["mse_in"], o["mse_out"],
                o["pass@1"], o["pass@5"], o.get("exact@1"), o.get("exact@5")]
    return [o["violation"], o["coverage"], o["mse_inside"], o["mse_outside"],
            o["pass1"], o["pass5"], o.get("exact1"), o.get("exact5")]


def _row(path):
    return [None if v is None else 100.0 * v for v in _extract_overall(_load(path))]


maze_dm = _row("runs/amaze_metrics/maze/dm/metrics.json")
maze_path = _row("runs/amaze_metrics/maze/path/metrics_pt.json")
queens_dm = _row("runs/amaze_metrics/queens/dm/metrics.json")
queens_path = _row("runs/amaze_metrics/queens/path/metrics.json")

OURS_ROWS = [
    ("DM", "ours", "62.7M", maze_dm + queens_dm),
    (r"\ours (ours)", "ours", "10.3M", maze_path + queens_path),
]

# Rows with values=None above are our own fine-tuning + eval reruns of
# specific baseline configs (see module docstring); fill them in here.
OUR_FT_RERUNS = {
    ("Bagel (fine-tuned)", "open"): (
        _row("runs/amaze_metrics/maze/bagel/metrics_bagel.json") + _row("runs/amaze_metrics/queens/bagel/metrics.json")
    ),
    ("Janus-Pro (fine-tuned)", "open"): (
        _row("runs/amaze_metrics/maze/janus/metrics.json") + _row("runs/amaze_metrics/queens/janus/metrics.json")
    ),
    ("Bagel (fine-tuned)", "cot"): (
        _row("runs/amaze_metrics_maze_bagel_ft_cot.json") + _row("runs/amaze_metrics_queens_bagel_ft_cot.json")
    ),
    ("Janus-Pro (fine-tuned)", "cot"): (
        _row("runs/amaze_metrics_maze_janus_ft_cot.json") + _row("runs/amaze_metrics_queens_janus_ft_cot.json")
    ),
}
BASELINE_ROWS = [
    (label, group, params, OUR_FT_RERUNS[(label, group)] if values is None else values)
    for (label, group, params, values) in BASELINE_ROWS
]

ALL_ROWS = BASELINE_ROWS + OURS_ROWS

# Drop MSE-In/MSE-Out from every row's values (see DROP_COLS above).
ALL_ROWS = [
    (label, group, params, [v for j, v in enumerate(values) if j not in DROP_COLS])
    for (label, group, params, values) in ALL_ROWS
]

# ── Bold = best value per column, globally across all rows, skipping None ──
best_row_idx = {}
for j in range(12):
    candidates = [(i, r[3][j]) for i, r in enumerate(ALL_ROWS) if r[3][j] is not None]
    if not candidates:
        continue
    key = (lambda p: p[1]) if j in MIN_BETTER else (lambda p: -p[1])
    best_row_idx[j] = min(candidates, key=key)[0]


def _fmt(v, row_i, col_j):
    if v is None:
        return "--"
    s = f"{v:.2f}"
    return rf"\textbf{{{s}}}" if best_row_idx.get(col_j) == row_i else s


def build_table():
    lines = [
        r"\begin{table*}[b!]\footnotesize",
        r"\renewcommand{\arraystretch}{1}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l@{\hspace{3pt}} c@{\hspace{10pt}} cccccc @{\hspace{20pt}} cccccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Params}} "
        r"& \multicolumn{6}{c}{\textbf{Continuous (Maze) Task}} "
        r"& \multicolumn{6}{c}{\textbf{Discrete (Queen) Task}} \\",
        r"\cmidrule(lr{20pt}){3-8} \cmidrule(lr){9-14}",
        r"& & \textbf{Violation}$\downarrow$ & \textbf{Coverage}$\uparrow$"
        r" & \textbf{Pass@1}$\uparrow$ & \textbf{Pass@5}$\uparrow$"
        r" & \textbf{Exact@1}$\uparrow$ & \textbf{Exact@5}$\uparrow$"
        r"& \textbf{Violation}$\downarrow$ & \textbf{Coverage}$\uparrow$"
        r" & \textbf{Pass@1}$\uparrow$ & \textbf{Pass@5}$\uparrow$"
        r" & \textbf{Exact@1}$\uparrow$ & \textbf{Exact@5}$\uparrow$ \\",
        r"\midrule",
    ]

    prev_group = None
    for i, (label, group, params, values) in enumerate(ALL_ROWS):
        # A new \rowcolor section header at the start of each group; a
        # group's base and fine-tuned sub-rows stay merged with no
        # separating \midrule between them.
        if group != prev_group:
            if prev_group is not None:
                lines.append(r"\midrule")
            lines.append(r"\rowcolor{gray}")
            lines.append(rf"\multicolumn{{14}}{{c}}{{\textit{{{GROUP_TITLES[group]}}}}} \\")
            lines.append(r"\midrule")
        cells = [label, params] + [_fmt(v, i, j) for j, v in enumerate(values)]
        lines.append(" & ".join(cells) + r" \\")
        prev_group = group

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Results on the AMAZE benchmark's Maze and Queens tasks. At a fraction of parameters, "
        r"\ours is able to outperform both standard diffusion models (DM) trained on the same data (DM) and "
        r"the image editing models evaluated on the benchmark, including proprietary systems.}",
        r"\label{tab:results_amaze}",
        r"\vspace{-1em}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main():
    out = build_table()
    with open("paper_plots/maze_queens_main_table.tex", "w") as f:
        f.write(out)
    print("Wrote paper_plots/maze_queens_main_table.tex")
    print("\n" + out)


if __name__ == "__main__":
    main()
