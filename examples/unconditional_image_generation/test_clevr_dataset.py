import os
import math
import pytest
import torch
from clevr_dataset import (
    sample_random_scene,
    make_tensor_from_scene,
    CLEVRHybridDataset,
    COLORS, MATERIALS, SHAPES, SIZES
)
from config import parse_args
from data_factory import get_dataloaders


@pytest.mark.parametrize("num_objects", [3, 5, 10, None])
def test_sample_random_scene(num_objects):
    """Test that scenes are generated with correct bounds, keys, and sizes."""
    scene = sample_random_scene(num_objects=num_objects, mode="absolute")

    # Check basic structure
    assert scene["mode"] == "absolute"
    assert "objects" in scene
    assert "relationships" in scene

    # If None was passed, it should pick a random number between 3 and 10
    actual_num_objects = len(scene["objects"])
    if num_objects is not None:
        assert actual_num_objects == num_objects
    else:
        assert 3 <= actual_num_objects <= 10

    # Check object constraints
    for obj in scene["objects"]:
        assert obj["color"] in COLORS
        assert obj["material"] in MATERIALS
        assert obj["shape"] in SHAPES
        assert obj["size"] in SIZES
        assert 0.0 <= obj["rotation"] <= 360.0
        assert len(obj["3d_coords"]) == 3
        assert len(obj["pixel_coords"]) == 3

    # Check relationships structure
    for rel in ['left', 'right', 'front', 'behind']:
        assert len(scene["relationships"][rel]) == actual_num_objects


def test_make_tensor_from_scene_absolute():
    """Verify the exact mathematical mapping for absolute mode."""
    test_scene = {
        "mode": "absolute",
        "objects": [
            {
                "color": "red",         # Index 1 in COLORS
                "material": "metal",    # Index 1 in MATERIALS
                "shape": "cube",        # Index 0 in SHAPES
                "size": "large",        # Index 1 in SIZES
                "rotation": 90.0,       # sin(90)=1, cos(90)=0
                "3d_coords": [2.5, -2.5, 0.0],
                "pixel_coords": [240.0, 160.0, 10.0]
            }
        ],
        "relationships": {"left": [[]], "right": [[]], "front": [[]], "behind": [[]]}
    }

    cond, mask = make_tensor_from_scene(test_scene)

    # Check shapes
    assert cond.shape == (1, 10, 21)
    assert mask.shape == (1, 10)

    # Check mask logic (1 object valid, 9 padded)
    assert mask[0, 0] == 1.0
    assert mask[0, 1:].sum() == 0.0

    # Reconstruct the expected 21-dim vector mathematically
    # [sin(90), cos(90)]
    rot_vec = [math.sin(math.radians(90)), math.cos(math.radians(90))]
    sz_vec = [1.0] # large
    mat_vec = [1.0] # metal
    sh_vec = [1.0, 0.0, 0.0] # cube (one-hot out of 3)
    col_vec = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # red (one-hot out of 8)

    # [px/480, py/320, pz/20] -> [240/480, 160/320, 10/20]
    p_vec = [0.5, 0.5, 0.5]
    # [x3/5, y3/5, z3/5] -> [2.5/5, -2.5/5, 0/5]
    c3_vec = [0.5, -0.5, 0.0]

    expected_vec = torch.tensor(rot_vec + sz_vec + mat_vec + sh_vec + col_vec + p_vec + c3_vec)

    # Assert values match exactly (using assert_close for float precision safety)
    torch.testing.assert_close(cond[0, 0, :], expected_vec, rtol=1e-4, atol=1e-4)
    # The padded elements should be purely zero
    assert cond[0, 1:, :].sum() == 0.0


def test_make_tensor_from_scene_relative():
    """Verify the exact mathematical mapping for relative mode."""
    test_scene = {
        "mode": "relative",
        "objects": [
            {
                "color": "gray", "material": "rubber", "shape": "sphere", "size": "small",
                "rotation": 0.0, "3d_coords": [0,0,0], "pixel_coords": [0,0,0]
            },
            {
                "color": "blue", "material": "metal", "shape": "cylinder", "size": "large",
                "rotation": 0.0, "3d_coords": [0,0,0], "pixel_coords": [0,0,0]
            }
        ],
        "relationships": {
            "left": [[1], []],  # Object 1 is to the left of Object 0
            "right": [[], [0]], # Object 0 is to the right of Object 1
            "front": [[], []],
            "behind": [[], []]
        }
    }

    cond, mask = make_tensor_from_scene(test_scene)

    assert cond.shape == (1, 10, 55)
    assert mask[0, :2].sum() == 2.0
    assert mask[0, 2:].sum() == 0.0

    # Let's verify the 40-dim relationship grid for the first object
    # Indices 15 through 54 represent the flattened 4x10 grid.
    rel_grid = cond[0, 0, 15:].view(4, 10)

    # Row 0 = 'left'. Object 0 has Object 1 to its left.
    assert rel_grid[0, 1] == 1.0
    assert rel_grid[0, 0] == 0.0

    # Row 1 = 'right'. Object 0 has nothing to its right.
    assert rel_grid[1, :].sum() == 0.0


def test_clevr_hybrid_dataset(tmp_path):
    """Test dataset behavior without triggering a download."""

    cache_dir = "cache_dir"
    clevr_path = os.path.join(cache_dir, "CLEVR_v1.0", "scenes", "CLEVR_train_scenes.json")

    if not os.path.exists(clevr_path):
        pytest.skip(f"CLEVR dataset not found in {cache_dir}. Skipping iteration test to prevent download.")

    dataset = CLEVRHybridDataset(
        root_dir=cache_dir,
        split="train",
        mode="absolute",
        image_size=64,
        download=False
    )

    assert len(dataset) > 0, "Dataset loaded but contains no scenes!"

    sample = dataset[0]

    assert "images" in sample
    assert "conditions" in sample
    assert "masks" in sample

    # Image should be transformed to (C, H, W) -> (3, 64, 64)
    assert sample["images"].shape == (3, 64, 64)
    # Tensors should not have a batch dimension here
    assert sample["conditions"].shape == (10, 21)
    assert sample["masks"].shape == (10,)


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_clevr_hybrid_dataset_exact_values(mode):
    """
    Test dataset iteration and verify that the Dataset's internal math
    exactly matches the standalone make_tensor_from_scene function for the first n items.
    """
    # Point this to your actual cache directory during testing
    cache_dir = "cache_dir"
    clevr_path = os.path.join(cache_dir, "CLEVR_v1.0", "scenes", "CLEVR_train_scenes.json")

    if not os.path.exists(clevr_path):
        pytest.skip(f"CLEVR dataset not found in {cache_dir}. Skipping iteration test to prevent download.")

    dataset = CLEVRHybridDataset(
        root_dir=cache_dir,
        split="train",
        mode=mode,
        image_size=64,
        download=False
    )

    assert len(dataset) > 0, "Dataset loaded but contains no scenes!"

    # Check the first N items to ensure stability
    n_items_to_check = 5

    print(f"\n--- Checking first {n_items_to_check} items in {mode} mode ---")

    for i in range(n_items_to_check):
        # 1. Get the output directly from the Dataset class
        sample = dataset[i]
        ds_features = sample["conditions"]
        ds_mask = sample["masks"]

        # Check basic shapes
        assert sample["images"].shape == (3, 64, 64)
        assert ds_mask.shape == (10,)
        assert ds_features.shape == (10, 21 if mode == "absolute" else 55)

        # 2. Get the raw scene dictionary and inject the 'mode' key
        raw_scene = dataset.scenes[i]
        raw_scene["mode"] = mode

        # 3. Get the output from the standalone function
        func_features, func_mask = make_tensor_from_scene(raw_scene)

        # make_tensor_from_scene adds a batch dimension of 1, so we strip it with [0]
        func_features = func_features[0]
        func_mask = func_mask[0]

        # 4. Assert they are mathematically identical
        torch.testing.assert_close(ds_features, func_features, msg=f"Features mismatch at index {i} for {mode} mode!")
        torch.testing.assert_close(ds_mask, func_mask, msg=f"Mask mismatch at index {i} for {mode} mode!")

        # Print the first object's features of the first scene just so you can visually inspect it
        if i == 0:
            print(f"\nScene 0, Object 0 features ({mode}):\n{ds_features[0].tolist()}")


def test_data_factory_standardization(monkeypatch):
    """Ensure the factory yields strictly standardized dictionary keys for all datasets."""

    # 1. Test Unconditional HF Dataset
    monkeypatch.setattr("sys.argv", ["train.py", "--dataset_name", "uoft-cs/cifar100", "--epoch_max_batches_train", "1", "--train_batch_size", "2"])
    args_uncond = parse_args()
    args_uncond.num_classes = 0 # Force unconditional
    train_dl, _ = get_dataloaders(args_uncond)
    batch = next(iter(train_dl))

    assert "images" in batch and batch["images"].shape == (2, 3, 64, 64)
    assert batch["conditions"] is None
    assert batch["masks"] is None

    # 2. Test Conditional HF Dataset
    monkeypatch.setattr("sys.argv", ["train.py", "--dataset_name", "uoft-cs/cifar100", "--epoch_max_batches_train", "1", "--train_batch_size", "2", "--num_classes", "100"])
    args_cond = parse_args()
    train_dl, _ = get_dataloaders(args_cond)
    batch = next(iter(train_dl))

    assert "images" in batch
    assert batch["conditions"] is not None and batch["conditions"].shape == (2,)
    assert batch["masks"] is None

    # 3. Test CLEVR Dataset
    monkeypatch.setattr("sys.argv", ["train.py", "--train_data_dir", "cache_dir", "--output_dir", "test-clevr", "--epoch_max_batches_train", "1", "--train_batch_size", "2"])
    args_clevr = parse_args()
    # Skip if CLEVR isn't downloaded locally to avoid breaking the test
    import os
    if os.path.exists(os.path.join("cache_dir", "CLEVR_v1.0", "scenes", "CLEVR_train_scenes.json")):
        train_dl, _ = get_dataloaders(args_clevr)
        batch = next(iter(train_dl))

        assert "images" in batch
        assert batch["conditions"] is not None and batch["conditions"].shape == (2, 10, 55) # default relative mode
        assert batch["masks"] is not None and batch["masks"].shape == (2, 10)
