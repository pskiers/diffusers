from __future__ import annotations

import base64
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

        # dataset_path may be a directory (loads maze_dataset_<split>.parquet
        # from it) OR a direct .parquet FILE (flat layout, e.g.
        # test_maze/square_3.parquet — the split is then ignored).
        if str(dataset_path).endswith(".parquet") and os.path.isfile(dataset_path):
            file_path = str(dataset_path)
        else:
            if split == "train":
                file_name = "maze_dataset_train.parquet"
            elif split == "val":
                file_name = "maze_dataset_val.parquet"
            elif split == "test":
                file_name = "maze_dataset_test.parquet"
            else:
                raise ValueError("AmazeDataset split must be 'train', 'val', or 'test'")
            file_path = os.path.join(dataset_path, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Amaze dataset file not found: {file_path}")

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
