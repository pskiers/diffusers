# Queen Generator Usage Guide

This folder contains tools for generating Queens puzzle samples and converting
the generated samples into parquet datasets compatible with the maze dataset
loader.

## Files

- `generate_queens_puzzle.py` generates Queens puzzle levels.
- `convert_queen_to_parquet.py` converts generated levels into train/test parquet files.
- `batch.sh` is an example Bash script for large batch generation.

## Requirements

Use Python 3.7 or newer. On Windows, prefer `py -3` if the default `python`
command points to an older interpreter.

Install the common dependencies:

```bash
pip install pillow numpy pandas pyarrow
```

Notes:

- `Pillow` is required for PNG output and for parquet conversion.
- If `Pillow` is not installed, `generate_queens_puzzle.py --image-format auto`
  falls back to SVG output.
- `convert_queen_to_parquet.py` requires PNG inputs, so generate with
  `--image-format png` before converting to parquet.

## Generate Puzzle Levels

Example:

```bash
python generate_queens_puzzle.py --n 7 --count 100 --outdir ./output_queens_n7 --seed 1 --cell-size 64 --queen-radius 16 --image-format png
```

### Generator Options

- `--n`: Grid size for an `N x N` board. Must be at least `4`.
- `--count`: Number of levels to generate. Must be positive.
- `--outdir`: Output directory. Default: `./output_queens`.
- `--seed`: Random seed. Default: `0`.
- `--max-attempts`: Maximum uniqueness-search attempts per level. Default: `2000`.
- `--cell-size`: Cell size in pixels. Default: `64`.
- `--queen-radius`: Queen circle radius in pixels. Default: 25 percent of cell size.
- `--image-format`: `auto`, `png`, or `svg`. Default: `auto`.

If generation fails with a uniqueness-search error, increase `--max-attempts` or
try a different `--seed`.

## Output Directory Layout

After generation, the output directory contains:

```text
output_queens_n7/
  puzzle/
    level_7_0000.png
    ...
  gt/
    level_7_0000.png
    ...
  json/
    level_7_0000.json
    ...
  cell_map/
    level_7_0000.pgm
    ...
```

Directory meanings:

- `puzzle/`: Unsolved puzzle images without queens.
- `gt/`: Ground-truth solved images with queens drawn as black circles.
- `json/`: Metadata for each level, including board size, image size, queen
  positions, region IDs, and cell ID rules.
- `cell_map/`: PGM maps where each pixel value is the cell ID
  `row * n + col`.

Level file names use this pattern:

```text
level_<n>_<index>
```

For example, `level_7_0000.png` is the first generated 7x7 level.

## Batch Generation

`batch.sh` is an example batch runner:

```bash
bash batch.sh
```

Before running it, edit these variables as needed:

- `N_LIST`: Board sizes to generate.
- `COUNT_PER_N`: Number of levels per board size.
- `CELL_SIZE`: Pixel size for each cell.
- `QUEEN_RADIUS`: Radius of the queen marker in pixels.
- `BASE_OUTDIR`: Output directory.
- `SEED`: Random seed.
- `MAX_ATTEMPTS`: Maximum attempts per generated level.

The current script is written for a Unix-like shell. On Windows, run it from WSL,
Git Bash, or adapt the command to PowerShell.

## Convert Generated Levels to Parquet

Example:

```bash
python convert_queen_to_parquet.py --queen-outdir ./output_queens_n7 --dataset-outdir ./queen_dataset_n7 --test-ratio 0.2 --seed 42
```

Windows PowerShell example:

```powershell
py -3 .\convert_queen_to_parquet.py --queen-outdir .\output_queens_n7 --dataset-outdir .\queen_dataset_n7 --test-ratio 0.2 --seed 42
```

The input directory must contain all four generated subdirectories:

- `puzzle/`
- `gt/`
- `json/`
- `cell_map/`

The converter writes:

```text
queen_dataset_n7/
  maze_dataset_train.parquet
  maze_dataset_test.parquet
```

### Converter Options

- `--queen-outdir`: Required. Directory created by `generate_queens_puzzle.py`.
- `--dataset-outdir`: Required. Directory for output parquet files.
- `--test-ratio`: Fraction of rows assigned to the test split. Default: `0.2`.
- `--seed`: Random seed for train/test splitting. Default: `42`.
- `--instruction-template`: Instruction text stored with each sample.

### Parquet Schema

The parquet rows include:

- `id`: Level ID, such as `level_7_0000`.
- `instruction`: Prompt/instruction text for the sample.
- `original_img`: Base64-encoded PNG bytes for the unsolved puzzle image.
- `m_original_img`: Same image as `original_img`.
- `sol_img`: Base64-encoded PNG bytes for the solved image.
- `cell_map`: Base64-encoded PNG bytes for the RGB-encoded cell map.
- `sample_json`: Full JSON metadata from the generator.
- `n`: Board size, when available in the JSON metadata.
- `width`: Image width, when available in the JSON metadata.
- `height`: Image height, when available in the JSON metadata.

For the RGB cell map, the original cell ID is encoded as:

```text
R = id & 255
G = (id >> 8) & 255
B = (id >> 16) & 255
```

## Complete Example

```bash
python generate_queens_puzzle.py --n 7 --count 6400 --outdir ./train_7_6400 --seed 1 --max-attempts 300 --cell-size 64 --queen-radius 16 --image-format png
python convert_queen_to_parquet.py --queen-outdir ./train_7_6400 --dataset-outdir ./queen_train_7 --test-ratio 0.5 --seed 42
```