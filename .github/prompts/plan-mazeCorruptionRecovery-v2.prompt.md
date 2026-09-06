# Plan: Maze corruption-recovery experiment (AMAZE)

Build a new probe — the maze analog of `examples/trm_diffusion/experiments/corruption_recovery_grid_probe.py` — that injects a deliberate mistake into the blue solution **path**, noises it, denoises to the end, and measures whether the model fixes it. Three corruption types (GAP / ADD / WALL) × three context levels (10/45/80% of the path shown) × two injection scenarios, scored against a floor and a controlled ceiling. Run on **two models (TRM vs DiT)** and diff.

## Models under test (TRM vs DiT) — same forward noising, different sampling

Both: `DDPMScheduler`, `prediction_type=sample`, `num_train_timesteps=100`, beta `squaredcos_cap_v2`, EMA on (load `use_ema=True`), condition = `spatial_conditions` (the maze). Load via `build_model(cfg)` + `eval.checkpoint_utils.load_checkpoint`; build conditions with `model._batch_to_sample(collate_fn(batch), device)`, decode with `model.decode_for_eval` (see `experiments/sample_amaze_metrics.py`; scores with `AmazeMetrics(task="maze")`, no classifier).

- **TRM (Painter+Thinker)** — `experiment=amaze_thinker_v2_controlnet` (cell_size 12, seq_len 144, grid 12). Needs BOTH checkpoints: `checkpoint=runs/pt_maze_final_thinker/checkpoint_final.pt` AND `painter.checkpoint=runs/pt_maze_final_painter/checkpoint_final.pt`. Sampling: **CFGPredictor, cfg_scale=2, 20 inference steps**.
- **DiT (standalone painter)** — `experiment=amaze_dit_maze` (`ConcatDiTPainter`, 6→3, patch 12). Single `checkpoint=runs/dit_maze_final/checkpoint_final.pt`. Sampling: **DirectPredictor (no CFG → cfg_scale=1), 99 inference steps**.

Implications:

- **`cell_size` 12 vs 20 is the MODEL reasoning grid, NOT the maze cell.** Corruption + scoring use `cell_map` from metadata, so they're identical and comparable across both models.
- **Noise level `t_start` is identical for both** (shared `num_train_timesteps=100`). Schedules: TRM 20-step = multiples of 5 `{0,5,…,95}`; DiT 99-step = every integer `{0,…,98}`. So **any multiple of 5 (default `t_start=50`) is valid on BOTH directly — no fraction/snap.** The probe validates `t_start` against each schedule and prints it (errors if off-grid). From t=50, TRM takes ~11 denoise steps and DiT ~51 — each model's **native** sampling (the thing being compared); the start point is identical, so it's fair. Optional later sweep: 30/50/70.
- `_denoise_from` already skips the null-cond pass when `cfg_scale==1` (DiT) — no code change needed.
- Data: use the **val split** of `.../data/amaze/train_maze/all_train_size144` (`include_metadata=true`), or the canonical per-scale test set from `gen_amaze.py test maze`.

## Phase 0 — Helpers (foundation)

> Why these exist: the raw data gives an _unordered_ set of path cells (`path_cell_ids`) and an RGB-packed pixel→cell map — neither directly supports "show the first p% of the path" or "edit cell X's pixels." These helpers translate raw data → the operations we need.

1. **Path-ordering helper** — _needed because "first p%" and "the frontier" require the path as an ordered sequence start→goal, but the metadata stores it as an unordered set._ Extract red start/end via `extract_red_markers`, build corridor adjacency from the decoded `cell_map` (ids ≠ 0), BFS start→goal for the ordered path, assert its set equals `path_cell_ids`. Reuses the BFS in `third_party/amaze/infer/maze_metrics.py` (~L343).
2. **Cell-geometry helper** — _needed because we reason in cells but paint/erase pixels._ `cell_map` → per-cell pixel regions, corridor-cell set, wall-pixel mask (id 0), corridor adjacency graph (for the ADD random walk + WALL nearest-wall direction).
3. **Maze cached-batch builder** — _load + prep each test maze once so the grid loop is fast and consistent._ Follow `experiments/sample_amaze_metrics.py`: `model._batch_to_sample(collate_fn(batch), device)` for the condition sample, `model.decode_for_eval` for [0,1] images, `AmazeMetrics(task="maze")` for scoring (NO classifier — that's sudoku-only). Per sample also stash decoded cell_map, ordered path, start/end, mask, raw metadata.

## Phase 1 — Corruption operators (parallel with Phase 0; maze analog of `_corrupt_with_wrong_digits`)

4. `render_partial_path(p)` — draw the first p% of the ordered path in path-blue on the maze background; rest blank. Sample blue color + stroke width from the real `sol_img`.
5. **GAP** — erase a contiguous **interior** chunk of the shown prefix (margin both sides); record erased cells.
6. **ADD** — from the frontier, random-walk through corridor cells in a continuous direction, **avoiding the entire GT path and its own trail**, stepping until no legal move remains (a natural dead-end). Paint the whole walk path-blue; record the added cells. (Length is maze-dependent, not fixed.)
7. **WALL** — from the frontier, straight blue line through the nearest wall **and a few pixels past its far side** (so it visibly pokes out, not ending flush with the wall); record wall-line pixels. Note: the in-wall segment (id 0) is invisible to the cell metric → caught by `still_through_wall`; the poked-out segment may land on a corridor → shows as violation.

## Phase 2 — Two injection scenarios (depends on Phase 1)

8. **Scenario A** (noise-from-start) — `add_noise` → splice only the corrupted region → `_denoise_from`. Mirrors `corruption_recovery_grid_probe.py`. **Do first.**
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

13. New `experiments/maze_corruption_recovery_probe.py` (Hydra, mirrors the open file). Run once per model (`experiment=amaze_thinker_v2_controlnet` / `amaze_dit_maze`, native sampling, `t_start=50`). Grid over type × level × scenario → JSON keyed `maze/{model}/{scenario}/{type}/level={p}/…`; diff the two models' JSONs.
14. _(Optional)_ cluster sweep runner; _(optional)_ plotting script (recovery vs level, floor/ceiling bands).

## Relevant files

- `experiments/corruption_recovery_grid_probe.py` — scenario-A skeleton (grid loop, clean-reuse, JSON).
- `experiments/targeted_corruption_probe.py` — scenario-B skeleton (paired same-seed, latency, collateral).
- `experiments/painter_only_denoise_probe.py` — `_denoise_from` (model-agnostic; gates CFG), `_get_painter`.
- `experiments/sample_amaze_metrics.py` — maze model-load + `AmazeMetrics` scoring for both TRM & DiT (`_batch_to_sample`, `decode_for_eval`, `sample_one_batch`).
- `eval/amaze_eval.py` — `AmazeMetrics`, `decode_cell_map_ids`, `_compute_maze_metrics`.
- `eval/checkpoint_utils.py` — `load_checkpoint` (handles painter+thinker + EMA).
- `third_party/amaze/infer/maze_metrics.py` — `extract_red_markers`, `extract_blue_path`, BFS pathfinding.
- `datasets/amaze_dataset.py` — metadata exposure (cell_map / mask_img / sol_img / metadata).
- `ablate_trm_loop_budget.py` — `_build_cached_batches` pattern (batch caching).

## Verification

1. Path-order unit check: BFS ordered-path set == `path_cell_ids` on N samples.
2. Manipulation check: floor moves as expected (GAP coverage↓, ADD violation↑, WALL `still_through_wall`≈1).
3. Sanity: controlled ceiling at low t_start / high p ≈ near-perfect pass.
4. Visual spot-check: dump a few corrupted images per type/level.
5. Small smoke run (num_samples≈16, t_start=50, 1 level) before the full grid.

## Decisions locked

- Option A (level = % of path shown); GAP ∈ {45,80}, ADD/WALL ∈ {10,45,80}; GAP interior with margin.
- Only WALL is a true violation → `corruption_type` axis; WALL recovery pixel-based + `still_through_wall`.
- Perfect maze ⇒ valid ≈ exact; three references (floor/ceiling/corrupted).
- Models: TRM (`amaze_thinker_v2_controlnet`, CFG=2, 20 steps; needs painter+thinker ckpts) vs DiT (`amaze_dit_maze`, no CFG, 99 steps). Each uses its **native** sampling.
- Noise: single **`t_start=50`** for both (a multiple of 5 → valid on both schedules; identical noise level). Optional later sweep 30/50/70. `cell_size` (12/20) = model grid, not maze cell.
- Scenario A first, then B.

## Domain facts (verified)

- Maze answer = connected BLUE path from red start (circle) to red end (X), scored as a SET of cell ids (coverage / background_violation / pass) in `eval/amaze_eval.py._compute_maze_metrics`. `predicted.discard(0)` ⇒ WALL pixels (cell id 0) are invisible to the cell metric (only MSE-outside sees them).
- `AmazeDataset` with `include_metadata=True` (auto for test/val) exposes: `spatial_conditions` (=maze), `images` (=solution), and metadata `{ metadata (JSON w/ path_cell_ids), sol_img, mask_img, cell_map, m_original_img }`.
- `cell_map` = RGB-packed pixel→cell id (id = R|G<<8|B<<16, 0=wall). Corridor cells = ids ≠ 0. GT path = `path_cell_ids` (consumed everywhere as an unordered SET → Option A needs BFS-derived order).
- These are PERFECT mazes (unique GT path + Pass@1 exact-set-match) ⇒ no alternative valid routes ⇒ "valid" ≈ "exact"; every detour dead-ends; only WALL is a true violation.

## Open questions

1. **Corruption size** — fixed small (≈K cells / ~10% of path) vs. swept like sudoku `n_cells`. Recommend **fixed small first**, add a sweep later.
2. **Path-ordering source** — trust the metadata list, or derive by BFS-from-markers? Recommend **BFS + validate against `path_cell_ids`** (robust to storage format).
