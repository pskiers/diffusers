# Dataset

| Puzzle | Training samples | Training Distribution                                                                                                  | Test samples - general metrics                                                                                                                     | Test samples - generalization                                      |
| ------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Maze   | **30 000**       | combination of:<br>- **shapes**: hex, triangle, square, circle<br>- **sizes**: 5x5, 7x7, 8x8, 9x9, 11x11, 13x13, 16x16 | **2800** generated samples:<br>- **shapes**: 700 samples per one shape<br>- **sizes**: 100 samples within this 700 samples for each size           | **400** generated samples - 100 samples in 3x3 size per one shape. |
| Queens | **30 000**       | all 4x4, 5x5, 6x6, 7x7, 8x8, 9x9, 10x10 boards.                                                                        | **350** generated samples of 4x4, 5x5, 6x6, 7x7, 8x8, 9x9, 10x10 board sizes, each size with euqal number of samples in the dataset (50 per shape) | **100** generated 12x12 boards.                                    |

All boards from train and test datasets are generated for each size / shape randomly. To prevent model from learning small version of boards by heart from training set, the training set was created with appropriate number of samples per size, so it won't cover all possible boards for this size.

## Queens

| Size    | Number of samples in the training dataset |
| ------- | ----------------------------------------- |
| 4x4     | 60                                        |
| 5x5     | 3040                                      |
| 6x6     | 5380                                      |
| 7x7     | 5380                                      |
| 8x8     | 5380                                      |
| 9x9     | 5380                                      |
| 10x10   | 5380                                      |
| SUMMARY | 30 000                                    |

Coverage of all possible boards from boards from training is minimal:

| n   | different boards number | training S | S/M coverage |
| --- | ----------------------- | ---------- | ------------ |
| 4   | ~3000                   | 60         | ~2%          |
| 5   | ~200 000                | 3040       | ~1.5%        |
| 6   | > $1.3*10^{7}$          | 5380       | <0.1%        |
| 7   | >>                      | 5380       | <<           |
| 8   | >>                      | 5380       | <<           |
| 9   | >>                      | 5380       | <<           |
| 10  | >>                      | 5380       | <<           |

- **M** — estimated number of *distinct* solvable boards
- **training S** — boards of that size in the 30k training set.
- **S/M coverage** — fraction of the whole space the model actually sees. It is ≪ 100% everywhere - even at n=4 only ~2% — so the model has to generalise, not memorise.

## Maze

Boards are generated per size and shape. 30 000 training samples evenly across 4 shapes × 7 sizes ≈ **1071 per (shape, size)**.

| n   | different mazes | training S | S/M coverage |
| --- | --------------- | ---------- | ------------ |
| 5   | 5.6×10⁸         | 1071       | ~$10^{-4}$   |
| 7   | 2.0×10¹⁹        | 1071       | <<           |
| 8   | 1.3×10²⁶        | 1071       | <<           |
| 9   | 8.3×10³³        | 1071       | <<           |
| 11  | 4.0×10⁵²        | 1071       | <<           |
| 13  | 2.2×10⁷⁵        | 1071       | <<           |
| 16  | 1.2×10¹¹⁷       | 1071       | <<           |

# Models

**Tested models**: NanoBanana-Pro, GPT-image-1, Bagel, Janus-Pro, standard DDPM diffusion model, TRM Painter-Thinker.

## Comertial, big models

Closed, general-purpose image-editing / generation APIs (**NanoBanana-Pro**, **GPT-image-1**,
**Seedream-4.5**). They are **not** trained or fine-tuned on our data — they are evaluated
**few-shot / in-context**, exactly as in the AMAZE paper (arXiv:2604.22868): the puzzle image is
passed together with a fixed natural-language instruction and the model returns the edited (solved)
board. This measures the out-of-the-box visual-planning ability of frontier models.

Prompts (verbatim, per task):

- **Maze:** `Add the blue solution path for the maze, connect start point (solid red circle) to end point (red 'X' mark). Ensure all original maze elements (walls, points, etc.) remain unchanged—only add the path.`
- **Queens:** `Given the puzzle image, generate the solved board by placing one queen (represented by a solid black circle in the center of a grid cell) in each row, column, and colored region while ensuring queens do not touch in 8-neighborhood.`

## Open-source big models

Open-weight multimodal models that we **fully fine-tune** (no LoRA) on the **same 30 000-sample
training set** used by the DDPM and TRM models, so the only axis that differs from our own models is
architecture / pre-training — not the training data. We reuse the AMAZE paper's own training code
(vendored under `third_party/amaze/sft/`).

- **Bagel** (`BAGEL-7B-MoT`) — a unified understanding+generation model (Mixture-of-Transformers).
  Full fine-tune with the vision encoder (ViT) and the VAE **frozen** and the language/generation
  transformer trainable; LR `1e-5` (cosine, warmup), diffusion loss `mse_weight=1.0` +
  `ce_weight=1e-6`, `timestep_shift=1.0`, 4×A100 (FSDP, hybrid shard). Trained image-editing style:
  puzzle image + prompt → solved image.
- **Janus-Pro** (`Janus-Pro-7B`) — an autoregressive multimodal model that emits the solution as a
  grid of VQ image tokens. Full fine-tune, LR `5e-6`, cross-entropy over the ~576 solution VQ tokens
  (384 px), grad-accumulation 16. Being a text→image AR model it is the weakest structural fit for
  the task (matching the paper's finding) but is included for completeness.

Both are fine-tuned for a fixed epoch budget over the 30k set (no early stopping), then scored with
the identical `AmazeMetrics` evaluator used for every other model.

## DDPM standard model

A pixel-space **DDPM** diffusion baseline with a **Diffusion-Transformer (DiT)** backbone
(`diffusers.DiTTransformer2DModel`) — the "no-reasoning" reference: a single feed-forward denoiser,
with no iterative latent reasoning beyond the diffusion steps themselves.

- **Backbone:** 12 transformer layers, hidden size 512 (8 heads × 64), patch size 12 over the
  144×144 image → a 12×12 = **144-token** grid; timestep injected via adaLN-zero, GELU-approx MLPs.
- **Conditioning:** the puzzle image is concatenated channel-wise with the noisy image
  (`in_channels = 3 + 3 = 6`, `out_channels = 3`, Palette/SR3 style); classifier-free guidance is
  **off**.
- **Size:** **≈ 62.6 M** parameters.
- **Training:** from scratch on the 30k set, **80 000** steps, batch 128 × grad-accum 6 (effective
  768), 1000-step DDPM schedule.

## TRM Painter-Thinker model

A two-stage **reasoning** diffusion model: a small frozen **painter** renders the image while a tiny
recurrent **thinker** does the visual planning and steers the painter — the counterpart to the DiT
baseline that isolates the effect of explicit iterative reasoning.

- **Painter** — `diffusers.UNet2DModel` (RGB 3→3, `block_out_channels=[32,64,64]`, 2 layers/block,
  144×144), **≈ 1.5 M** parameters. Trained from scratch on the 30k set (**40 000** steps), then
  **frozen**.
- **Thinker** — `SpatialTRM`, a Tiny Recursive Model over the 12×12 = 144-cell grid: hidden size
  512, 8 heads, RoPE, 2 weight-shared transformer layers **re-used across `L_cycles=6 × H_cycles=3`
  recurrent cycles with `n_sup=16` deep-supervision steps** (effective compute is deep while the
  parameter count stays small). Trained **40 000** steps on the 30k set with the painter frozen.
- **Condition encoder** — `NoisySpatialConditionEncoder` (6→512, pooled to the 12×12 grid) feeds the
  noisy solution + puzzle into the thinker's token grid.
- **Translator** — a **ControlNet** that injects the thinker's per-cell reasoning into the frozen
  painter's UNet. Sampling uses classifier-free guidance.

(Thinker / encoder / translator exact parameter counts to be confirmed on the cluster; the module
has a cluster-only dependency and cannot be instantiated locally.)

## Evaluation protocol and metrics

- **Same data axis.** Every model we train or adapt — the DDPM/DiT baseline, the TRM Painter-Thinker,
  and the fine-tuned Bagel & Janus-Pro — uses the **same 30 000-sample training set** per puzzle. The
  commercial APIs are evaluated few-shot (no training). Fairness is enforced on **data**, not on step
  budget or model size.
- **Same evaluator.** All outputs are scored with one implementation of the AMAZE-paper metrics
  (`eval/amaze_eval.py`, `AmazeMetrics`, arXiv:2604.22868). Each test puzzle is generated **K = 5**
  times (independent noise) → Pass@1 (first attempt) and Pass@5 (best-of-5).
- **Two regimes.** _General_ metrics use in-distribution sizes (maze 5–16, queens 4–10);
  _generalization_ metrics use held-out out-of-distribution sizes (**maze 3×3**, **queens 12×12**)
  that appear only at test time.
- **Breakdowns.** Maze is reported per geometry (square / hex / triangle / circle); queens per board size.

Metric definitions (↑ = higher is better, ↓ = lower is better):

- **Coverage↑** = |predicted ∩ GT| / |GT| — fraction of the true solution cells recovered.
- **Violation↓** = |predicted ≠ GT| / |predicted| — fraction of predicted cells that are wrong.
- **Pass@1↑** = fraction of _exact_ solves on the first attempt (Coverage = 1 and Violation = 0).
- **Pass@5↑** = fraction with an exact solve within 5 attempts.
- **MSE-In↓ / MSE-Out↓** = pixel MSE inside / outside the solution mask.

# Results

## Queens

### Accumulated metrics

General metrics

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

Generalization metrics (12x12 boards)

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

### Per size

For top 3 models from accumulated metrics

## Maze

### Accumulated metrics

General metrics

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

Generalization metrics (3x3 boards)

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

### Per geometry, sizes accumulated

**Square**, General metrics

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

**Square**, Generalization metrics (3x3 boards)

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

---

**Hex**, General metrics

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

**Hex**, Generalization metrics (3x3 boards)

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

---

**Triangular**, General metrics

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

**Triangular**, Generalization metrics (3x3 boards)

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

---

**Circle**, General metrics

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

**Circle**, Generalization metrics (3x3 boards)

| Model                  | Violation↓ | Coverage↑ | MSE In↓ | MSE Out↓ | Pass@1↑ | Pass@5↑ |
| ---------------------- | ---------- | --------- | ------- | -------- | ------- | ------- |
| NanoBanana-Pro         |            |           |         |          |         |         |
| GPT-image-1            |            |           |         |          |         |         |
| Seedream-4.5           |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| (FINE TUNED) Bagel     |            |           |         |          |         |         |
| (FINE TUNED) Janus-Pro |            |           |         |          |         |         |
| DDPM diffusion model   |            |           |         |          |         |         |
| TRM Painter-Thinker    |            |           |         |          |         |         |

---

### Per geometry and size

For top 3 models from accumulated metrics

**Square**
(zdjęcia dla top 3 modeli)

**Hex**
(zdjęcia dla top 3 modeli)

**Triangular**
(zdjęcia dla top 3 modeli)

**Circle**
(zdjęcia dla top 3 modeli)

# Addition

Proof that conditioning works - painter trained on all sies still produces 7x7 boards.

## Denoising trajectory: TRM vs DiT

To compare _how_ the two diffusion models reach the answer (not only the final score) we capture the
intermediate denoising trajectory: at a set of evenly-spaced sampling steps we decode each model's
running estimate of the final image (the `x0` prediction) and save it as a per-timestep image series
plus a labelled filmstrip. Run with the same puzzles, seed and step count for TRM and DiT, the strips
line up step-for-step, so the difference in _reasoning dynamics_ — how early each model commits to a
coherent path / queen layout — can be read off directly. Implemented in
`experiments/sample_amaze_trajectory.py`, launched via `slurm_scripts/trajectory_amaze.sh`.

---

**MAZE**
WANDB_PROJECT=amaze_final RUN_NAME=pt_maze_final sbatch slurm_scripts/train_maze_painter_thinker.sh painter=true thinker=true sample=true

WANDB_PROJECT=amaze_final RUN_NAME=dit_maze_final sbatch slurm_scripts/train_maze_dit.sh

**QUEENS**
PAINTER_CKPT=runs/queens_painter_v2/checkpoint_final.pt THINKER_CKPT=runs/queens_pt_v2v2/checkpoint_final.pt WANDB_PROJECT=amaze_final \
RUN_NAME=pt_queens_final sbatch slurm_scripts/train_queens_painter_thinker.sh painter=false thinker=false sample=true

WANDB_PROJECT=amaze_final sbatch slurm_scripts/eval_dit.sh queens runs/queens_dit_baseline/checkpoint_final.pt

**TRAJEKTORIA**
COMBO=hexagon_n13 SEED=0 STEPS=8 sbatch slurm_scripts/trajectory_amaze.sh dit maze <dit.pt>
COMBO=n8 SEED=0 STEPS=8 sbatch slurm_scripts/trajectory_amaze.sh dit queens <dit.pt>

COMBO=square_n9 SEED=0 STEPS=8 sbatch slurm_scripts/trajectory_amaze.sh trm maze <thinker.pt> <painter.pt>
COMBO=n8 SEED=0 STEPS=8 sbatch slurm_scripts/trajectory_amaze.sh trm queens <thinker.pt> <painter.pt>

**FINE TUNING**

| #   | Gdzie     | Komenda                                                                                                                                                                  |
| --- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0   | lokalnie  | commit + push nowych wrapperów (patrz niżej)                                                                                                                             |
| 1   | **login** | `git pull` na klastrze                                                                                                                                                   |
| 2   | **login** | `bash third_party/amaze/setup_ft_code.sh` (dociąga bazowe repo Bagel/Janus) + `huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT`                                     |
| 3   | `sbatch`  | dane (jeśli jeszcze nie po zmianach −3×3/+8×8): `sbatch slurm_scripts/gen_amaze.sh train maze --shape all --size all` oraz `sbatch slurm_scripts/gen_amaze.sh test both` |
| 4   | `sbatch`  | `sbatch slurm_scripts/export_amaze_ft.sh both`                                                                                                                           |
| 5   | `sbatch`  | **smoke** eval bez modelu: `sbatch slurm_scripts/eval_ft.sh maze` (BACKEND=dummy) → sprawdza czy generate→score→`amaze_final` działa                                     |
| 6   | `sbatch`  | **smoke** FT: `TOTAL_STEPS=20 BAGEL_MODEL_PATH=<ścieżka> sbatch slurm_scripts/train_bagel_ft.sh maze`                                                                    |
| 7   | `sbatch`  | pełny FT: `BAGEL_MODEL_PATH=<ścieżka> sbatch slurm_scripts/train_bagel_ft.sh maze`                                                                                       |
| 8   | `sbatch`  | prawdziwy eval: `BACKEND=bagel CHECKPOINT=<ckpt> RUN_NAME=ft_bagel_maze sbatch slurm_scripts/eval_ft.sh maze`                                                            |
