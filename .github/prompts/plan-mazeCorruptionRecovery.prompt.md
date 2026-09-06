# Plan: Maze corruption-recovery experiment (AMAZE)

Build a new probe — the maze analog of `examples/trm_diffusion/experiments/corruption_recovery_grid_probe.py` — that injects a deliberate mistake into the blue solution **path**, noises it, denoises to the end, and measures whether the model fixes it. Three corruption types (GAP / ADD / WALL) × three context levels (10/45/80% of the path shown) × two injection scenarios, scored against a floor and a controlled ceiling.

## Phase 0 — Helpers (foundation)

> Why these exist: the raw data gives an _unordered_ set of path cells (`path_cell_ids`) and an RGB-packed pixel→cell map — neither directly supports "show the first p% of the path" or "edit cell X's pixels." These helpers translate raw data → the operations we need.

1. **Path-ordering helper** — _needed because "first p%" and "the frontier" require the path as an ordered sequence start→goal, but the metadata stores it as an unordered set._ Extract red start/end via `extract_red_markers`, build corridor adjacency from the decoded `cell_map` (ids ≠ 0), BFS start→goal for the ordered path, assert its set equals `path_cell_ids`. Reuses the BFS in `third_party/amaze/infer/maze_metrics.py` (~L343).
2. **Cell-geometry helper** — _needed because we reason in cells but paint/erase pixels._ `cell_map` → per-cell pixel regions, corridor-cell set, wall-pixel mask (id 0), corridor adjacency graph (for the ADD random walk + WALL nearest-wall direction).
3. **Maze cached-batch builder** — _load + prep each test maze once so the grid loop is fast and consistent._ Analog of `_build_cached_batches` in `ablate_trm_loop_budget.py`; per sample returns condition, `sol_img` target, decoded cell_map, ordered path, start/end, mask, raw metadata.

## Phase 1 — Corruption operators (parallel with Phase 0; maze analog of `_corrupt_with_wrong_digits`)

4. `render_partial_path(p)` — draw the first p% of the ordered path in path-blue on the maze background; rest blank. Sample blue color + stroke width from the real `sol_img`.
5. **GAP** — erase a contiguous **interior** chunk of the shown prefix (margin both sides); record erased cells.
6. **ADD** — from the frontier, random-walk through corridor cells in a continuous direction, **avoiding the entire GT path and its own trail**, stepping until no legal move remains (a natural dead-end). Paint the whole walk path-blue; record the added cells. (Length is maze-dependent, not fixed.)
7. **WALL** — from the frontier, straight blue line through the nearest wall **and a few pixels past its far side** (so it visibly pokes out, not ending flush with the wall); record wall-line pixels. Note: the in-wall segment (id 0) is invisible to the cell metric → caught by `still_through_wall`; the poked-out segment may land on a corridor → shows as violation.

## Phase 2 — Two injection scenarios (depends on Phase 1)

8. **Scenario A** (noise-from-start) — `add_noise` → splice only the corrupted region → `_denoise_from`. Mirrors `corruption_recovery_grid_probe.py`.
9. **Scenario B** (mid-generation) — real trajectory from noise, splice corruption at `corrupt_fraction`, continue; paired same-seed. Mirrors `targeted_corruption_probe.py`.

## Phase 3 — Metrics & baselines (depends on Phase 2)

10. Three references per condition:
    - **Floor** — score the corrupted **clean** image (mistake drawn, _before_ diffusion noise, no denoise). Manipulation check = "how bad is the sabotage." (Do NOT score the noised tensor — that's just noise → zeros.)
    - **Controlled ceiling** — take the clean context (same p%, **no** mistake) → noise to t_start → denoise. "What the model does WITHOUT sabotage at this p, t_start." Recovery is judged relative to this, not to a perfect 1.0.
    - **Corrupted-denoised** — the actual run (corrupted context → noise → denoise).
11. Metrics (grouped):
    - **Per-corruption recovery (headline).** GAP = fraction of erased cells restored; ADD = fraction of added cells removed; WALL = fraction of wall-line pixels cleared. Report each as **recovery_rate** (pooled + per-board mean) and **full_recovery_rate** (% boards with ALL corrupted units fixed).
    - **Kept-the-mistake (adapt).** ADD `adapt_rate` (added branch kept), WALL `still_through_wall_rate` (any blue-on-wall remaining). GAP has no adapt (restored vs still-missing).
    - **Whole-solution quality** via `AmazeMetrics(task=maze)`: coverage, background_violation, pass, mse_in/out — reported for corrupted-denoised **and** controlled-ceiling so the gap is visible.
    - **recovered_and_valid_rate** — recovery achieved AND final path valid/exact.
    - **collateral_break_rate** — previously-correct (shown-context) path cells broken by the fix.
    - **matches_clean_run_rate** — corrupted-denoised output identical to the clean (controlled-ceiling) run → corruption left no trace.
    - _(Scenario B only)_ **recovery_latency** — denoising step at which the fix first sticks.
    - Aggregations: pooled (accumulated) **and** per-board average.
12. Reuse `AmazeMetrics(task="maze")` for coverage/violation/pass/mse on all three references.

## Phase 4 — Runner & outputs (depends on Phase 3)

13. New `experiments/maze_corruption_recovery_probe.py` (Hydra, mirrors the open file). Grid over type × level × scenario × (t_start | corrupt_fraction) → JSON keyed `maze/{scenario}/{type}/level={p}/…`.
14. _(Optional)_ cluster sweep runner; _(optional)_ plotting script (recovery vs level, floor/ceiling bands).

## Relevant files

- `experiments/corruption_recovery_grid_probe.py` — scenario-A skeleton (grid loop, clean-reuse, JSON).
- `experiments/targeted_corruption_probe.py` — scenario-B skeleton (paired same-seed, latency, collateral).
- `experiments/painter_only_denoise_probe.py` — `_denoise_from` (model-agnostic), `_get_painter`.
- `eval/amaze_eval.py` — `AmazeMetrics`, `decode_cell_map_ids`, `_compute_maze_metrics`.
- `third_party/amaze/infer/maze_metrics.py` — `extract_red_markers`, `extract_blue_path`, BFS pathfinding.
- `datasets/amaze_dataset.py` — metadata exposure (cell_map / mask_img / sol_img / metadata).
- `ablate_trm_loop_budget.py` — `_build_cached_batches`, `_load_checkpoint` patterns.

## Verification

1. Path-order unit check: BFS ordered-path set == `path_cell_ids` on N samples.
2. Manipulation check: floor moves as expected (GAP coverage↓, ADD violation↑, WALL `still_through_wall`≈1).
3. Sanity: controlled ceiling at low t_start / high p ≈ near-perfect pass.
4. Visual spot-check: dump a few corrupted images per type/level.
5. Small smoke run (num_samples≈16, 1 t_start, 1 level) before the full grid.

## Decisions locked

- Option A (level = % of path shown); GAP ∈ {45,80}, ADD/WALL ∈ {10,45,80}; GAP interior with margin.
- Only WALL is a true violation → `corruption_type` axis; WALL recovery pixel-based + `still_through_wall`.
- Perfect maze ⇒ valid ≈ exact; three references (floor/ceiling/corrupted).

## Domain facts (verified)

- Maze answer = connected BLUE path from red start (circle) to red end (X), scored as a SET of cell ids (coverage / background_violation / pass) in `eval/amaze_eval.py._compute_maze_metrics`. `predicted.discard(0)` ⇒ WALL pixels (cell id 0) are invisible to the cell metric (only MSE-outside sees them).
- `AmazeDataset` with `include_metadata=True` (auto for test/val) exposes: `spatial_conditions` (=maze), `images` (=solution), and metadata `{ metadata (JSON w/ path_cell_ids), sol_img, mask_img, cell_map, m_original_img }`.
- `cell_map` = RGB-packed pixel→cell id (id = R|G<<8|B<<16, 0=wall). Corridor cells = ids ≠ 0. GT path = `path_cell_ids` (consumed everywhere as an unordered SET → Option A needs BFS-derived order).
- These are PERFECT mazes (unique GT path + Pass@1 exact-set-match) ⇒ no alternative valid routes ⇒ "valid" ≈ "exact"; every detour dead-ends; only WALL is a true violation.

## Models under test (TRM vs DiT) — same forward noising, different sampling

Both: DDPMScheduler, prediction_type=sample, num_train_timesteps=100, beta squaredcos_cap_v2,
EMA on (use_ema=True), condition=spatial_conditions. Load via build_model(cfg) +
eval.checkpoint_utils.load_checkpoint; conditions via model.\_batch_to_sample(collate_fn(batch),device);
decode via model.decode_for_eval (see experiments/sample_amaze_metrics.py; AmazeMetrics(task=maze), no classifier).

- TRM (Painter+Thinker): experiment=amaze_thinker_v2_controlnet (cell_size12/seq144/grid12).
  Needs BOTH checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt
  AND painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt.
  Sampling: CFGPredictor cfg_scale=2, 20 steps.
- DiT (standalone): experiment=amaze_dit_maze (ConcatDiTPainter, 6->3, patch12).
  Single checkpoint=runs/dit_maze_final/checkpoint_final.pt. Sampling: DirectPredictor (cfg_scale=1), 99 steps.

Noise position parameterized by FRACTION of 100 (snap per model, since 20 vs 99 steps); each keeps native
sampling. cell_size (12/20)=model grid, not maze cell. Data: val split of train_maze/all_train_size144
(include_metadata=true) or gen_amaze.py test maze.

## Open questions

1. **Path-ordering source** — trust the metadata list, or derive by BFS-from-markers? Recommend **BFS + validate against `path_cell_ids`** (robust to storage format).
2. **Corruption size** — fixed small (≈K cells / ~10% of path) vs. swept like sudoku `n_cells`. Recommend **fixed small first**, add a sweep later.
3. **Which checkpoint/model** to probe (TRM+painter vs. DiT baseline)? Sets the condition keys and eval callback — needed before the runner is concrete.
4. **Scenario priority** — A first (matches the open file), then B. Confirm.
