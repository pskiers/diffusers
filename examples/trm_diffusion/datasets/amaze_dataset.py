from __future__ import annotations

import base64
import dataclasses
import io
import os
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.data_sample import DataSample, collate_data_samples


class AmazeDataset(Dataset):
    """TRM-compatible loader for local Amaze parquet datasets.

    Exposes marked puzzle input images and solved puzzle images as tensors for TRM training.
        :marked image: source: `m_original_img`, exposed as `spatial_conditions`
        :solved image: source: `sol_img`, exposed as `images`
    """
    collate_fn = staticmethod(collate_data_samples)

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        image_size: int = 256,
        condition_field: Optional[str] = "m_original_img",
        target_field: str = "sol_img",
        num_channels: int = 3,
        include_metadata: Optional[bool] = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split

        self.image_size = image_size
        self.condition_field = condition_field
        self.target_field = target_field
        self.include_eval_metadata = \
            (split in ("test", "val")) if include_metadata is None else include_metadata

        if num_channels not in (1, 3):
            raise ValueError("AmazeDataset num_channels must be 1 or 3")
        self.num_channels = num_channels

        # dataset_path may be a directory (loads the split parquet from it) OR a
        # direct .parquet FILE (flat layout, e.g. test_maze/hex/n3_hex_test.parquet
        # — the split is then ignored).
        if str(dataset_path).endswith(".parquet") and os.path.isfile(dataset_path):
            file_path = str(dataset_path)
        else:
            # New layout uses train.parquet / validate.parquet; keep the legacy
            # maze_dataset_*.parquet names as a fallback for older datasets.
            candidates = {
                "train": ("train.parquet", "maze_dataset_train.parquet"),
                "val": ("validate.parquet", "maze_dataset_val.parquet"),
                "test": ("test.parquet", "maze_dataset_test.parquet"),
            }.get(split)
            if candidates is None:
                raise ValueError("AmazeDataset split must be 'train', 'val', or 'test'")
            file_path = next(
                (os.path.join(dataset_path, name)
                 for name in candidates
                 if os.path.exists(os.path.join(dataset_path, name))),
                None,
            )
            if file_path is None:
                raise FileNotFoundError(
                    f"Amaze dataset file not found in {dataset_path} "
                    f"(looked for {' or '.join(candidates)})"
                )

        self.data = pd.read_parquet(file_path)
        if self.data.empty:
            raise ValueError(f"Amaze dataset at {file_path} is empty")

        steps: list = [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        ]
        if num_channels == 1:
            steps.append(transforms.Grayscale(num_output_channels=1))
        steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.data)

    def _decode_image(self, raw_image) -> Optional[Image.Image]:
        if raw_image is None or pd.isna(raw_image):
            return None

        if isinstance(raw_image, Image.Image):
            return raw_image.convert("RGB")

        if isinstance(raw_image, (bytes, bytearray)):
            raw_image = bytes(raw_image)
            return Image.open(io.BytesIO(raw_image)).convert("RGB")

        if isinstance(raw_image, str):
            if raw_image.startswith("data:"):
                raw_image = raw_image.split(",", 1)[1]
            try:
                decoded = base64.b64decode(raw_image)
                return Image.open(io.BytesIO(decoded)).convert("RGB")
            except Exception:
                if os.path.exists(raw_image):
                    return Image.open(raw_image).convert("RGB")
                raise

        raise ValueError(f"Unsupported Amaze image type: {type(raw_image)}")

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        return self.transform(image)    # type: ignore

    def __getitem__(self, idx: int) -> DataSample:
        row = self.data.iloc[idx]

        target_image = self._decode_image(row.get(self.target_field))
        if target_image is None:
            raise ValueError(
                f"Amaze sample {idx} is missing target field '{self.target_field}'"
            )
        image = self._image_to_tensor(target_image)

        spatial_condition = None
        if self.condition_field is not None:
            condition_image = self._decode_image(
                row.get(self.condition_field)
            )
            if condition_image is not None:
                spatial_condition = self._image_to_tensor(condition_image)

        metadata = None
        if self.include_eval_metadata:
            metadata = {
                'id': row.get('id'),
                'metadata': row.get('metadata'),
                'sample_json': row.get('sample_json'),
                'original_img': self._decode_image(row.get('original_img')),
                'm_original_img': self._decode_image(row.get('m_original_img')),
                'sol_img': self._decode_image(row.get('sol_img')),
                'mask_img': self._decode_image(row.get('mask_img')),
                'cell_map': self._decode_image(row.get('cell_map')),
            }

        return DataSample(
            images=image,
            spatial_conditions=spatial_condition,
            metadata=metadata,
            prompt=row.get('instruction') or row.get('text') or None,
            puzzle_id=torch.tensor(idx, dtype=torch.long),
        )


def scale_and_paste(
    image: torch.Tensor, scale: float, off_y: int, off_x: int, canvas_color=(0.0, 0.0, 0.0)
) -> torch.Tensor:
    """Shrink a (C, H, W) image by `scale` and paste it at (off_y, off_x) on a
    same-size canvas of `canvas_color`."""
    C, H, W = image.shape
    new_h, new_w = max(1, round(H * scale)), max(1, round(W * scale))
    resized = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False
    )[0].clamp(0.0, 1.0)
    canvas = torch.tensor(canvas_color, dtype=image.dtype).view(-1, 1, 1).expand(C, H, W).clone()
    canvas[:, off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas


def sample_scale_and_offset(size: int, scale_min: float, scale_max: float) -> tuple[float, int, int]:
    """Random (scale, off_y, off_x) for a square `size` image. Uses torch's RNG,
    which the DataLoader seeds differently per worker."""
    scale = scale_min + (scale_max - scale_min) * torch.rand(()).item()
    new = max(1, round(size * scale))
    off_y = int(torch.randint(0, size - new + 1, ()).item())
    off_x = int(torch.randint(0, size - new + 1, ()).item())
    return scale, off_y, off_x


class AmazeScaledDataset(AmazeDataset):
    """AmazeDataset whose diffusion TARGET (`images`, the solved puzzle) is shrunk
    to a random fraction of its size and pasted at a random offset on a plain
    canvas, while the condition (`spatial_conditions`, the unsolved puzzle) is
    left untransformed — the Amaze counterpart of MNISTSudokuScaledDataset.

    The model therefore has to paint the solution at an unknown scale and
    position, and the correspondence between condition positions and target
    positions is broken. Generated images are cropped back to their content
    bounding box before scoring (eval.amaze_eval.crop_to_content, enabled by the
    *_scaled eval-callback configs and +crop_to_content=true in the metrics
    script). Metadata (the GT used for scoring) is untouched.

    Args (in addition to AmazeDataset's):
        scale_min, scale_max: target scale drawn uniformly from this range.
        canvas_color: RGB in [0, 1]; must not occur along the puzzle image's own
            border, so the bounding box recovers the pasted image exactly
            (black works for queens, whose board has a dark-grey frame; mazes
            have a black background and need another colour).
        transform_condition: also apply the SAME transform to the condition
            (the aligned control: scale/position variety without mismatch).
    """

    def __init__(
        self,
        *args,
        scale_min: float = 0.8,
        scale_max: float = 0.9,
        canvas_color=(0.0, 0.0, 0.0),
        transform_condition: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.canvas_color = tuple(float(c) for c in canvas_color)
        self.transform_condition = bool(transform_condition)

    def __getitem__(self, idx: int) -> DataSample:
        sample = super().__getitem__(idx)
        scale, off_y, off_x = sample_scale_and_offset(sample.images.shape[-1], self.scale_min, self.scale_max)
        updates = {"images": scale_and_paste(sample.images, scale, off_y, off_x, self.canvas_color)}
        if self.transform_condition and sample.spatial_conditions is not None:
            updates["spatial_conditions"] = scale_and_paste(
                sample.spatial_conditions, scale, off_y, off_x, self.canvas_color
            )
        return dataclasses.replace(sample, **updates)


def test_dataset():
    from pathlib import Path

    trm_root = Path(__file__).resolve().parents[1]
    dataset_path = trm_root / 'data' / 'amaze' / 'queens_7x7_debug'

    dataset = AmazeDataset(str(dataset_path), split='train', include_metadata=True)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")

    sample = dataset[0]
    assert sample.images is not None and sample.spatial_conditions is not None

    try:
        img = Image.fromarray((sample.images.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        img.show()
        img1 = Image.fromarray((sample.spatial_conditions.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        img1.show()

    except FileNotFoundError:
        print("Error: The specified image file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    test_dataset()
