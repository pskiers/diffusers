# Solving Sokoban using only diffusion with TRM mechanism

## Get data

Data available on google drive https://drive.google.com/drive/folders/1Ac9MwbjnS9nFCEOoxBXNGkDYlXoJL_Xe. Download it and extract to `sokoban/data/raw` directory

## Run TRM training

Login to wandb. Metrics will be displayed there.

```bash
wandb login
```

There are 12 possible combinations for sokoban task.

- There are 4 model variations: Unet standard, Unet TRM, ViT standard and ViT TRM
- There are 3 tasks varations: unconditional, conditional with single k and conditional with multiple k.

All different versions of sokoban boards generation are defined in `configs/models/sokoban_*`, `configs/tasks/*`, `configs/experiment/sokoban.yaml` and `configs/dataset/sokoban.yaml`

```bash
python3 train.py experiemnt=sokoban model= task=

```

## Run TRM sampling

```bash
python3 sample_trm.py \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  arch.L_layers=2 \
  arch.halt_max_steps=16 \
  +checkpoint_path="../outputs/checkpoints/step_XXXXX" \
  +output_dir="../outputs/samples" \
  +num_samples=10
```
