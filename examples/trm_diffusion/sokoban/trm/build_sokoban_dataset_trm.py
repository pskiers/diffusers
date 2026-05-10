"""
Generate sokoban dataset to train TRM and TRM-PT Thinker.
One sokoban board is 12x12
Data is uploaded from google drive. Change this path to your own if you want to check other variations of sokoban boards.

Dataset consist of:
1. Input x vector - flattened 1D board with only floors (empty board)
2. Label y vector - flattened 1D board with the actual state
"""
import os
import json
import numpy as np
import joblib
from glob import glob
from tqdm import tqdm
from argdantic import ArgParser
from pydantic import BaseModel
from typing import Literal

from ..fields_states import FieldStates


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_dir: str = "/home/gosia/STUDIA/DYPLOM/reasoning-diffusion/trm/diffusers/examples/trm_diffusion/sokoban/data/raw/12-12-4"
    output_dir: str = "./data/"
    max_training_boards: int | None = 100000
    max_validation_boards: int | None = 20000
    seed: int = 42


def create_masked_input(board_np: np.ndarray) -> np.ndarray:
    """Change board to empty board (only floors)"""
    input_board = board_np.copy()
    mask = (input_board != FieldStates.FLOOR.id)
    input_board[mask] = FieldStates.FLOOR.id
    return input_board


def build_sokoban_trm(split_name: Literal["train", "test"], config: DataProcessConfig, data_dir_files: list[str]):
    max_boards = config.max_training_boards if split_name == "train" else config.max_validation_boards

    save_dir = os.path.join(config.output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    if len(data_dir_files) == 0:
        raise ValueError(f"Brak plików dla zbioru {split_name}")

    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0]
    }

    current_idx = 0
    seq_len = None
    total_groups = 0

    for f in tqdm(data_dir_files, total=len(data_dir_files), desc=f"Creating {split_name} dataset"):
        try:
            data = joblib.load(f)
        except Exception as e:
            print(f"Error loading file {f}: {e}")
            continue

        for trajectory in data.values():
            # (game sequence length, H, W)
            trajectory_np = np.argmax(trajectory, axis=3).astype(np.uint8)
            trajectory_np = trajectory_np + 1

            n_frames = trajectory_np.shape[0]   # it is better to have more shorter sequences
            sample_size = min(50, n_frames)
            random_indices = np.random.choice(n_frames, size=sample_size, replace=False)
            trajectory_np = trajectory_np[random_indices]

            for board in trajectory_np:
                if seq_len is None:
                    seq_len = board.shape[0] * board.shape[1]

                input_board = create_masked_input(board)

                results["inputs"].append(input_board.flatten())
                results["labels"].append(board.flatten())

                current_idx += 1

                results["puzzle_indices"].append(current_idx)
                results["puzzle_identifiers"].append(0)

                if max_boards and current_idx >= max_boards:
                    break

            results["group_indices"].append(current_idx)
            total_groups += 1

            if max_boards and current_idx >= max_boards:
                break

        if max_boards and current_idx >= max_boards:
            break

    for k in ["inputs", "labels"]:
        results[k] = np.stack(results[k], axis=0).astype(np.uint8)
    for k in ["puzzle_indices", "group_indices", "puzzle_identifiers"]:
        results[k] = np.array(results[k], dtype=np.int32)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    metadata = {
        "seq_len": seq_len,
        "vocab_size": 8,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": total_groups,
        "mean_puzzle_examples": current_idx / max(1, total_groups),
        "total_puzzles": current_idx,
        "sets": ["all"]
    }

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nCreated {current_idx} boards grouped into {total_groups} trajectories for {split_name}.")


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    print(f"Reading dataset from local directory: {config.input_dir}...")

    all_files = glob(os.path.join(config.input_dir, "*"))
    if len(all_files) == 0:
        raise ValueError(f"No files in {config.input_dir} directory.")

    np.random.seed(config.seed)
    np.random.shuffle(all_files)
    split_index = int(len(all_files) * 0.8)

    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    print(f"Found {len(all_files)} files. Split: {len(train_files)} train, {len(test_files)} test.")

    build_sokoban_trm("train", config, train_files)
    build_sokoban_trm("test", config, test_files)


if __name__ == "__main__":
    cli()
