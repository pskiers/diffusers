# amaze_faint_eval: faint-path-aware AMAZE maze scorer (recovered, verbatim)

The standard AMAZE scorer (`eval/amaze_eval.py` → `third_party/ear_amaze/infer/maze_metrics.extract_blue_path`)
misses semi-transparent ("ghost") path strokes. `faint_metrics.py` subclasses `AmazeMetrics` and swaps only the path
detector (blue tint B−R ≥ 10 instead of HSV saturation ≥ 50), plus a short-stroke filter. Maze only: there is no
Queens counterpart here.

**Provenance.** Written by a Claude session on Helios (2026-09-23/24) in
`/net/scratch/hscra/plgrid/plgpawel269/faint_eval/`, which no longer exists. The files here were rebuilt by
replaying that session's Write/Edit calls (transcript `~/.claude/projects/-net-scratch-hscra-plgrid-plgmgrzanka/
30d0ac41-….jsonl` on Helios); `faint_metrics.py`, `calibrate.py` and `stage1_visual.py` are byte-identical to the
session's file-history snapshots. They are kept unmodified, so paths still point at Helios (`TRM_ROOT`, `run.sh`,
the Bagel generation dirs in `BAGEL_RUNS`).

**What used it.** The paper's Table 2 Maze rows for PaTh (`runs/amaze_metrics/maze/path/metrics_pt.json`,
Exact@1 0.7514; standard scorer 0.7543) and fine-tuned Bagel (`.../maze/bagel/metrics_bagel.json`, 0.4339) carry the
tag "faint-path aware (blue-red tint >= 10)". The Maze DM row (0.1896) and all Queens rows use the standard scorer.
The script that rescored the PaTh generations was not found (not on Helios or Athena; WCSS not yet checked).

**This is not the final version.** The paper JSONs record "edge frame 0.25 cells" and "native (no 144px
downscale)"; this `faint_metrics.py` has no edge-frame logic. The transcript stops mid-work (stage-1 visuals and GT
calibration ran; its last two edits failed), so the scorer kept evolving in a session or machine not found yet.
Do not use these files to reproduce Table 2 until the final version is located.
