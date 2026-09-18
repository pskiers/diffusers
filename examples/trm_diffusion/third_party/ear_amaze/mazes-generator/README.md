# Batch Maze Generator

This directory contains a Node.js maze generator and Python utilities for
building image/parquet datasets from the generated mazes.

The main workflow is:

1. Generate maze images with `batch-maze-generator.js`.
2. Use the generated metadata, masks, and solution images to build parquet
   datasets with `process_maze_into_parquet.py`.
3. Optionally inspect cell maps with `visualize_cell_map.py`.

Run all commands below from this directory.

## Directory Contents

- `batch-maze-generator.js`: Main Node.js generator.
- `maze-config.json`: Example JSON configuration for batch generation.
- `process_maze_into_parquet.py`: Recommended parquet converter. It preserves
  original images, marked images, solution images, path masks, cell maps, and
  metadata.
- `create_parquet_dataset.py`: Legacy/simple converter for maze/solution image
  pairs only.
- `visualize_cell_map.py`: Debug/inspection tool for generated cell maps.
- `generate_batch.sh`: Bash batch helper. The checked-in version still contains
  old test-file references, so review and edit it before running.
- `js/lib/`: Maze shapes, algorithms, drawing logic, random utilities, and mask
  generation logic.
- `css/`, `js/`: Browser-side files from the original maze project.

## Requirements

### Node.js

Use a modern Node.js version with ES module support. Node 18+ is recommended.

The generator imports `jsdom` and `sharp`:

```bash
npm init -y
npm pkg set type=module
npm install jsdom sharp
```

If a valid `package.json` already exists in your checkout, you can usually run
only:

```bash
npm install jsdom sharp
```

### Python

Use Python 3.8+ for the dataset utilities.

For parquet conversion:

```bash
python -m pip install pillow pandas pyarrow tqdm
```

For cell-map visualization:

```bash
python -m pip install opencv-python numpy matplotlib
```

## Generate Mazes

### List Available Options

```bash
node batch-maze-generator.js list
```

This prints supported shapes, algorithms, and exit configurations.

### Generate Preset Samples

```bash
node batch-maze-generator.js samples
```

This generates several sample mazes across different shapes and algorithms.

### Generate One Maze

Square maze:

```bash
node batch-maze-generator.js single --shape square --width 20 --height 20 --algorithm recursiveBacktrack --exitConfig vertical --seed 123 --filename square_20x20_recursive.png
```

Circle maze:

```bash
node batch-maze-generator.js single --shape circle --layers 10 --algorithm wilson --exitConfig hardest --seed 123 --filename circle_10_wilson.png
```

Hexagon maze:

```bash
node batch-maze-generator.js single --shape hexagon --width 12 --height 10 --algorithm truePrims --exitConfig horizontal --seed 123 --filename hexagon_12x10_true_prims.png
```

Triangle maze:

```bash
node batch-maze-generator.js single --shape triangle --width 12 --height 10 --algorithm huntAndKill --exitConfig vertical --seed 123 --filename triangle_12x10_hunt_kill.png
```

Important: the current CLI parser uses `--exitConfig`, not `--exits`. If you
pass `--exits`, it is parsed but ignored by `generateMaze`.

## Generate From a Config File

```bash
node batch-maze-generator.js config maze-config.json
```

The config file can contain either one maze object or an object with a `mazes`
array:

```json
{
  "mazes": [
    {
      "shape": "square",
      "width": 20,
      "height": 20,
      "algorithm": "recursiveBacktrack",
      "exitConfig": "vertical",
      "seed": 123,
      "filename": "square_20x20_recursive.png"
    },
    {
      "shape": "circle",
      "layers": 10,
      "algorithm": "wilson",
      "exitConfig": "hardest",
      "seed": 456,
      "filename": "circle_10_wilson.png"
    }
  ]
}
```

### Config Fields

| Field | Required | Applies To | Description |
| --- | --- | --- | --- |
| `shape` | No | All | `square`, `triangle`, `hexagon`, or `circle`. Default: `square`. |
| `width` | No | Square, triangle, hexagon | Grid width. Default: `10`. |
| `height` | No | Square, triangle, hexagon | Grid height. Default: `10`. |
| `layers` | No | Circle | Number of circular layers. Default: `10`. |
| `algorithm` | No | All | Generation algorithm. Default: `recursiveBacktrack`. |
| `exitConfig` | No | All | `vertical`, `horizontal`, `hardest`, or `no exits`. Default: `vertical`. |
| `seed` | No | All | Random seed. If omitted, the current timestamp is used. |
| `filename` | No | All | Output filename. The extension is normalized to `.png`. |

## Supported Shapes and Algorithms

| Algorithm | Square | Triangle | Hexagon | Circle |
| --- | --- | --- | --- | --- |
| `recursiveBacktrack` | Yes | Yes | Yes | Yes |
| `simplifiedPrims` | Yes | Yes | Yes | Yes |
| `truePrims` | Yes | Yes | Yes | Yes |
| `wilson` | Yes | Yes | Yes | Yes |
| `aldousBroder` | Yes | Yes | Yes | Yes |
| `huntAndKill` | Yes | Yes | Yes | Yes |
| `kruskal` | Yes | No | No | No |
| `binaryTree` | Yes | No | No | No |
| `sidewinder` | Yes | No | No | No |
| `ellers` | Yes | No | No | No |

The code also defines `none`, which renders the grid without carving a maze.

## Output Files

The generator creates these directories automatically:

```text
generated_mazes/
generated_mazes_no_markers/
generated_solutions/
generated_metadata/
```

Directory meanings:

- `generated_mazes/`: Maze image with entrance/exit labels.
- `generated_mazes_no_markers/`: Maze image without entrance/exit labels. This
  is the recommended `original_img` source for training.
- `generated_solutions/`: Solution image with the path drawn.
- `generated_metadata/`: JSON metadata, path masks, and cell maps.

## Metadata

Each metadata JSON file contains:

- `path_coordinates`: Raw maze coordinates along the solution path.
- `path_cell_ids`: Cell IDs along the solution path.
- `start_cell`: Start cell coordinates.
- `end_cell`: End cell coordinates.
- `maze_config`: Shape, size, algorithm, and seed.
- `image_size`: Output image size, cell size, and wall width.
- `difficulty`: Turn-complexity statistics for the solution path.

The corresponding metadata images are:

- `<base>_path_mask.png`: Binary path mask. Path pixels are white and
  non-path pixels are black.
- `<base>_cell_map.png`: Per-pixel cell ID map encoded into RGB channels for
  OpenCV-based decoding. Use `visualize_cell_map.py` as the reference decoder.

## Convert to Parquet

Use `process_maze_into_parquet.py` for the full dataset format:

```bash
python process_maze_into_parquet.py --maze-dir ./generated_mazes --no-markers-dir ./generated_mazes_no_markers --solution-dir ./generated_solutions --metadata-dir ./generated_metadata --output ./maze-dataset/maze_dataset.parquet --train-ratio 0.9 --seed 42
```

Set `--train-ratio` explicitly. The current code default is `1.0`, which puts
all matched samples in the train split.

### Full Parquet Schema

Rows produced by `process_maze_into_parquet.py` include:

- `id`: Random UUID.
- `original_img`: Base64-encoded PNG from `generated_mazes_no_markers/`.
- `m_original_img`: Base64-encoded PNG from `generated_mazes/`.
- `instruction`: Instruction text.
- `sol_img`: Base64-encoded PNG from `generated_solutions/`.
- `mask_img`: Base64-encoded PNG path mask.
- `cell_map`: Base64-encoded PNG cell map.
- `metadata`: Serialized JSON metadata.

To limit conversion for a quick test:

```bash
python process_maze_into_parquet.py -n 20 --train-ratio 0.8
```

## Legacy Simple Parquet Converter

`create_parquet_dataset.py` only matches images from `generated_mazes/` and
`generated_solutions/`. It does not include no-marker images, masks, cell maps,
or JSON metadata.

```bash
python create_parquet_dataset.py --maze-dir ./generated_mazes --solution-dir ./generated_solutions --output ./maze_dataset.parquet
```

Prefer `process_maze_into_parquet.py` for training datasets that need masks and
cell maps.

## Visualize Cell Maps

After generating mazes, run:

```bash
python visualize_cell_map.py --metadata-dir ./generated_metadata --maze-dir ./generated_mazes --output-dir ./visualizations --colormap tab20b
```

Windows PowerShell:

```powershell
py -3 .\visualize_cell_map.py --metadata-dir .\generated_metadata --maze-dir .\generated_mazes --output-dir .\visualizations --colormap tab20b
```

This creates comparison/debug images under `visualizations/`.

## Batch Shell Script

`generate_batch.sh` is intended for Unix-like shells such as Linux, WSL, or Git
Bash. Before using it, review these values near the top of the script:

- `BATCH_SIZE`
- `MIN_MAZE_SIZE`
- `MAX_MAZE_SIZE`
- `OUTPUT_DIR`
- `SOLUTION_DIR`
- `NO_MARKER_DIR`
- `MASK_DIR`

Also update old references to `batch-maze-generator_test.js` so they point to
the current generator:

```bash
node batch-maze-generator.js ...
```
