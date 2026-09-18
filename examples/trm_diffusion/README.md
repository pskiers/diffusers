# TRM Diffusion
## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/pskiers/diffusers.git
cd diffusers
git checkout pskiers/trm-diffusion
pip install .
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```


Initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

And login to [Wandb](https://wandb.ai/) with:
```bash
wandb login
```

### Optional: closed-loop eval simulators (PushT / BlockPush / ToolHang)

Not required for training — `models/dp_eval_callbacks.py` gracefully skips
(logs a warning, returns `{}`) any callback whose simulator isn't installed.
Only needed for the periodic/standalone closed-loop coverage/success-rate
metrics on those three tasks.

```bash
# PushT — image/hybrid variant (pymunk<7 required — pymunk 7.0 removed
# Space.add_collision_handler() in favor of on_collision(), but gym-pusht,
# even the latest 0.1.6 on PyPI, still uses the old API and breaks with
# pymunk>=7):
pip install gym-pusht "pymunk<7" shapely

# PushT — lowdim (keypoints) variant, for the vendored 9-keypoint
# examples/trm_diffusion/third_party/pusht_keypoints/ env (real-stanford/
# diffusion_policy's own PushTKeypointsEnv — gym-pusht's built-in keypoints
# option uses a different, incompatible 8-keypoint scheme):
pip install "pymunk<7" shapely gym==0.26.2 pygame scikit-image

# ToolHang — pinned to match upstream real-stanford/diffusion_policy's own
# conda_environment.yaml as closely as possible (current robosuite releases
# have drifted API-wise from the version they benchmarked against).
# --no-deps + manual deps below: this fork's setup.py pins numba<=0.53.1,
# which has no wheels for modern Python — that pin is stale, not a real
# compatibility requirement (checked: this robosuite fork's only numba usage,
# in robosuite/utils/numba.py, is a plain `numba.jit(nopython=True, ...)`
# decorator, a stable API that works fine on any current numba).
pip install --no-deps "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"
pip install numba scipy "free-mujoco-py==2.1.6"
pip install robomimic==0.2.0
#
# mujoco_py builds its C extension lazily on first use, and free-mujoco-py==2.1.6
# has two bugs that combine to break GPU rendering on most HPC clusters:
#   1. builder.py's GPU-detection (get_nvidia_lib_dir()) only checks a couple
#      of Docker/desktop path conventions, missing plain /usr/lib/nvidia —
#      so it silently falls back to a CPU/OSMesa build that then fails
#      because system OSMesa dev headers (GL/osmesa.h) usually aren't
#      installed (and installing them needs root).
#   2. mujoco_py/__init__.py unconditionally OVERWRITES (not appends to)
#      LD_LIBRARY_PATH at import time — so even after exporting the driver
#      dir yourself, and even after fixing (1), it gets silently wiped the
#      moment `import mujoco_py` runs, and the later internal check for that
#      same path always fails.
#   3. Even with (1) and (2) fixed, its bundled gl/eglplatform.h is an old
#      (2015-era) Khronos header that unconditionally requires real X11 dev
#      headers (X11/Xlib.h) for any __unix__ build — headers most clusters
#      don't have installed (and installing them needs root), even though
#      mujoco_py's actual EGL usage never touches a real X11 display.
#   4. Even with (1)-(3) fixed, gl/eglshim.c needs GL/glew.h, which the
#      free-mujoco-py==2.1.6 sdist simply forgot to package (a packaging
#      bug — the prebuilt libglewegl.so it DOES bundle is untouched by
#      this, only the header used to compile against it is missing).
#   5. Even with (1)-(4) fixed, fixing (2) (appending instead of overwriting
#      LD_LIBRARY_PATH) exposes a *different* bug: builder.py's
#      fix_shared_library() calls hardcode the patchelf target as
#      f'{LD_LIBRARY_PATH}/{name}', assuming LD_LIBRARY_PATH is a single
#      directory — once it's a real colon-separated PATH (required for the
#      OS's own dynamic linker to see your driver dir), this embeds a
#      literal garbage path into the .so and it fails to dlopen at import
#      time. This script rewrites fix_shared_library() to actually search
#      each LD_LIBRARY_PATH directory for the real file.
# Run this once (and again after ANY reinstall of free-mujoco-py/robosuite/
# robomimic, which overwrites all five patches — it's idempotent, safe to
# re-run):
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python scripts/patch_mujoco_py_gpu.py
#
# Env creation also shells out to the `patchelf` binary (to fix RPATHs on
# compiled .so files at runtime) — install the PyPI package, which ships a
# prebuilt binary, no compiler/root needed:
pip install patchelf
#
# DO NOT run a plain `pip install robosuite` (or `pip install robosuite
# robomimic`) after this — even once, even later — it silently replaces the
# pinned 1.2.0 fork with current PyPI robosuite (1.5.2, which depends on the
# modern `mujoco` package instead of `mujoco_py`), and the two are NOT
# compatible: robosuite 1.2.0's controllers call the old
# `mujoco_py.cymj._mj_fullM(model, dst, qM)` C API directly, which breaks in
# confusing ways (wrong argument types, garbage output) if `self.sim.model`
# turns out to be a `mujoco._structs.MjModel` from the new package instead.
# If this ever happens, recover with:
#   pip uninstall -y robosuite mink qpsolvers
#   pip install --no-deps "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"
#   pip show robosuite | grep -i version   # must say 1.2.0, not 1.5.2

# BlockPush — the real physics env is vendored at
# examples/trm_diffusion/third_party/block_pushing/ from
# real-stanford/diffusion_policy's OWN fork of google-research/ibc's
# BlockPushMultimodal (Apache 2.0), NOT the raw google-research/ibc version.
# This matters, not just for import convenience: diffusion_policy's fork adds
# an explicit left/right bias (`add = 0.12 * rng.choice([-1, 1])`) when
# placing the two targets — that's the actual "multimodal" part of the task,
# and it's what the published multimodal_push_seed.zarr training data's
# initial conditions actually look like. Evaluating against the raw
# google-research/ibc env (no such bias) silently evaluates a
# diffusion_policy-trained painter against an out-of-distribution env and
# was enough on its own to produce ~0% success rate on an otherwise
# correctly-trained policy. The fork's _get_reward() also differs (partial
# credit per block reaching a target, not just a binary 0/1) — our
# BlockPushEvalCallback's p1/p2 scoring is written against that scheme.
# See third_party/block_pushing/__init__.py for the full rationale.
#
# Needs pybullet-svl (Stanford Vision & Learning Lab's pybullet fork, also
# pinned by upstream diffusion_policy's own conda_environment.yaml) instead of
# plain pybullet — pybullet-svl bundles the xarm6_robot.urdf asset that
# xarm_sim_robot.py needs and that isn't published in either
# google-research/ibc or real-stanford/diffusion_policy's public repos.
pip install gin-config absl-py six
pip uninstall -y pybullet   # pybullet-svl refuses to install alongside plain pybullet
pip install "setuptools<81"  # its old sdist needs pkg_resources, removed from newer setuptools
pip install --no-build-isolation pybullet-svl
```

## Training

### CIFAR-100 Unconditional
```bash
# Standard Diffusion Unconditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=uncond_cifar100_std

# TRM Diffusion Unconditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=uncond_cifar100_trm
```

### CIFAR-100 Conditional
```bash
# Standard Conditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=cond_cifar100_std

# Small Loop (TRM) Conditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=cond_cifar100_trm
```

### ImageNet
```bash
# Standard Conditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=cond_imgnet_std

# Small Loop (TRM) Conditional
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=cond_imgnet_trm
```

### Clevr
```bash
# Standard CLEVR
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=clevr_relative_std

# Small Loop (TRM) CLEVR
accelerate launch --mixed_precision="fp16" --num_processes=1 train.py experiment=clevr_relative_trm
```

## Sampling


### CIFAR-100 Unconditional
```bash
# Standard Diffusion Unconditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=uncond_cifar100_std \
  checkpoint_path="cifar100-standard-long/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=250

# TRM Diffusion Unconditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=uncond_cifar100_trm \
  checkpoint_path="cifar100-size-small-nsup1-long/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=5000
```

### CIFAR-100 Conditional
```bash
# Standard Conditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=cond_cifar100_std \
  checkpoint_path="cifar100-conditional-standard/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=250

# Small Loop (TRM) Conditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=cond_cifar100_trm \
  checkpoint_path="cifar100-conditional-hrm-nsup-4-T-3-n-6/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=5000
```

### ImageNet
```bash
# Standard Conditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=cond_imgnet_std \
  checkpoint_path="imagenet-conditional-standard-big/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=64

# Small Loop (TRM) Conditional
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=cond_imgnet_trm \
  checkpoint_path="imagenet-conditional-hrm-nsup-4-T-3-n-6-fix-ch256/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=64
```

### Clevr
```bash
# Standard CLEVR
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=clevr_relative_std \
  checkpoint_path="clevr-standard-att-early-cross-att-relative-fix/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=64

# Small Loop (TRM) CLEVR
accelerate launch --mixed_precision="fp16" sample.py \
  experiment=clevr_relative_trm \
  checkpoint_path="clevr-hrm-nsup-4-T-3-n-6-cross-att-pred-eps-relative-fix/checkpoint-<STEP>" \
  num_samples=10000 \
  sample_batch_size=64
```
## Vendored AMAZE benchmark code (`third_party/ear_amaze`)

The AMAZE benchmark code is vendored here rather than used as an external
checkout, because running it required a number of corrections. Everything under
`third_party/ear_amaze/` is the authors' code plus the fixes listed below; our
own scripts (`eval/`, `experiments/`, `slurm_scripts/`) import from it directly.

There used to be a second, partial copy at `third_party/amaze/`. It has been
removed: it lacked every fix below, plus `config/`, `dataset/`, `flow_grpo/`,
`data/*.py`, the maze-generator JS and `requirements.txt`. The directory is
named `ear_amaze` with an underscore because a hyphen is not a legal Python
module name — which is why the scorer previously had to import from the stale
copy.

Only source is committed. Model weights, generated images and training outputs
are excluded via `.gitignore` (they run to hundreds of GB).

### Corrections to the authors' code

| file | what was wrong |
|---|---|
| `infer/bagel/inferencer.py` | Bagel inference fixes across `InterleaveInferencer` (7 hunks). The `think=` / `max_think_tokens=` CoT parameters were present but unreachable — see below. |
| `infer/bagel/modeling/bagel/bagel.py` | stray import removed |
| `infer/infer_bagel.py` | output-resolution fix, a reworked `MazePromptImageDataset`, and substantial changes to the `eval()` loop (batched inference, image/prompt alignment) |
| `infer/infer_janus.py` | import fix, plus the CoT pass below |
| `sft/bagel/sft.py`, `sft/janus/sft.py` | fine-tuning script fixes |

### Chain-of-thought (CoT) sampling

The paper describes a `<think>` planning pass before image generation. It was
not usable as shipped:

- **Bagel** — `InterleaveInferencer` implemented `think=` and
  `max_think_tokens=` together with `GEN_THINK_SYSTEM_PROMPT`, but
  `infer_bagel.py` never passed them, so the feature was dead code. It is now
  forwarded from the `THINK` environment variable.
- **Janus** — no CoT support of any kind existed. Janus-Pro cannot emit text and
  an image in one pass, so CoT is implemented as two passes: the conversation is
  built *without* the trailing `image_start_tag` (leaving the assistant turn open
  for text), the puzzle's VQ embeddings are injected exactly as in the image
  path, a plan is generated, and the prompt is rewritten to carry it before the
  normal image pass runs. A failed planning pass logs an error and falls back to
  plain generation rather than losing the batch.

Enable with `THINK=1` (optionally `MAX_THINK_TOKENS=...`). CoT is inference-time
only — neither model was fine-tuned with it — so a CoT run is a full
re-generation, not a re-score. CoT output is written to a separate
`<task>_cot` / `*_cot` directory so it can never overwrite the plain generations.

### Metric corrections (`eval/amaze_eval.py`)

- **`Pass` now follows the paper**: `max(0, Coverage - Violation)`, a continuous
  score, for both maze and queens. It was previously implemented as a binary
  exact-match. Note the paper's `Violation` is normalised by the *generated*
  cell count, which is what we use; the authors' released reward code normalises
  by the total background cell count instead. The two disagree.
- **`Exact` added** as a separate metric — strict set equality between the
  predicted and ground-truth cell sets. This is the stricter criterion the old
  `Pass` implemented, kept because it is not gameable: under the paper's `Pass`,
  painting every cell of the maze blue scores 0.73–0.82, since the solution path
  covers most cells.
- **`Pass@K` is the mean over the K attempts** (the paper evaluates each model
  five times and reports the average); **`Exact@K` is best-of-K**, the usual
  pass@k reading.
- **Start and goal cells are treated as given.** They are drawn into the input as
  a red dot and cross, so requiring the model to *also* paint them blue measures
  obedience to a rendering convention rather than planning. One model left them
  at exactly `0.000` blue fill and lost every triangle board to it while drawing
  the route correctly; another painted through them. The terminal cells are
  intersected with the ground truth, so this can never invent a cell or lower
  `Violation`, and it is a no-op for a model that does paint them.

A ground-truth self-test is available: `sbatch slurm_scripts/test_metrics.sh`
scores the reference solution image for every geometry and size and asserts
`Coverage=1, Violation=0, Pass=1, Exact=1`.

### Scoring resolution

Metrics are computed at each puzzle's **native** resolution — the `cell_map` is
never resized, and the generated image is upscaled to match it. The `IMAGE_SIZE`
normalisation applied before scoring (default 144) is therefore a bottleneck only
for models that generate larger images. Pass `IMAGE_SIZE=` to match the
generation resolution when scoring Janus (384) or Bagel (640), otherwise detail
is discarded; the scorer warns when it is downscaling.

### Collecting results

`collect_final_results.py` gathers per-model metrics into `final_results/`. It
merges the per-shape maze JSONs (the scorer writes one file per geometry inside
the generation directory), strips OOD sections, and refuses to ship a result
that mixes freshly-scored and stale shards — recording each shard's timestamp so
a partial re-run is visible rather than silent.
