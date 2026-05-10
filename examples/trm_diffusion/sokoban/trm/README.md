# Solving sokoban using official TRM implementation

https://github.com/SamsungSAILMontreal/TinyRecursiveModels

## Get data

Data available on google drive https://drive.google.com/drive/folders/1Ac9MwbjnS9nFCEOoxBXNGkDYlXoJL_Xe. Download it and extract to `sokoban/data/raw` directory

## Create dataset

Run

```bash
cd sokoban/trm
python3 -m build_sokoban_dataset_trm
```

All nessesary files will be in `sokoban/data/` directory (train and test datasets)

## Run TRM training

Login to wandb. Metrics will be displayed there.

```bash
wandb login
```

Run

```bash
python3 -m sokoban.trm.trm_paper_utils.pretrain \
  arch=trm \
  data_paths="[sokoban/data]" \
  evaluators="[]" \
  epochs=1000 \
  eval_interval=100 \
  global_batch_size=128 \
  lr=1e-4 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  arch.L_layers=2 \
  arch.halt_max_steps=16 \
  +eval_save_outputs=[logits] \
  +checkpoint_path="/net/tscratch/people/plgmgrzanka/training-outputs/trm/checkpoints" \
  ema=True \
  +run_name="trm_sokoban_baseline" \
  +project_name="Sokoban-TRM"
```

## Run TRM sampling

```bash
python3 trm/sample_trm.py \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  arch.L_layers=2 \
  arch.halt_max_steps=16 \
  +checkpoint_path="/net/tscratch/people/plgmgrzanka/training-outputs/trm/checkpoints/step_XXXXX" \
  +output_dir="/net/tscratch/people/plgmgrzanka/training-outputs/trm/samples" \
  +num_samples=10
```
