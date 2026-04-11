from sokoban.sokoban_dataset import SokobanBitDataset, SokobanDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from data_utils import LimitedLoader
from clevr_dataset import CLEVRHybridDataset


def get_dataloaders(args):
    """
    Factory function to return standardized dataloaders.
    Always yields batches with keys: 'images', 'conditions', 'masks'.
    Values are None if the dataset doesn't use them.
    """
    # 1. Define standard image augmentations
    train_augmentations = transforms.Compose(
        [
            transforms.Resize(
                (args.dataset.resolution, args.dataset.resolution), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            (
                transforms.CenterCrop(args.dataset.resolution)
                if args.dataset.center_crop
                else (
                    transforms.RandomCrop(args.dataset.resolution)
                    if not args.dataset.center_crop
                    else transforms.Lambda(lambda x: x)
                )
            ),
            transforms.RandomHorizontalFlip() if args.dataset.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    eval_augmentations = transforms.Compose(
        [
            transforms.Resize(
                (args.dataset.resolution, args.dataset.resolution), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # 2. Determine Dataset Type
    if args.dataset.dataset_type == "hf":
        train_ds = load_dataset(
            args.dataset.dataset_name, args.dataset.dataset_config_name, cache_dir=args.cache_dir, split="train"
        )
        eval_ds = load_dataset(
            args.dataset.dataset_name,
            args.dataset.dataset_config_name,
            cache_dir=args.cache_dir,
            split=args.dataset.test_split_name,
        )

        is_conditional = args.dataset.num_classes is not None and args.dataset.num_classes > 0

        def hf_train_transform(examples):
            images = [train_augmentations(image.convert("RGB")) for image in examples[args.dataset.image_key]]
            out = {"images": images}
            if is_conditional:
                out["conditions"] = examples[args.dataset.class_key]
            return out

        def hf_eval_transform(examples):
            images = [eval_augmentations(image.convert("RGB")) for image in examples[args.dataset.image_key]]
            out = {"images": images}
            if is_conditional:
                out["conditions"] = examples[args.dataset.class_key]
            return out

        train_ds.set_transform(hf_train_transform)
        eval_ds.set_transform(hf_eval_transform)

    elif args.dataset.dataset_type == "clevr":
        train_ds = CLEVRHybridDataset(
            root_dir=args.dataset.train_data_dir,
            split="train",
            mode=args.dataset.dataset_mode,
            image_size=args.dataset.resolution,
            download=True,
        )
        eval_ds = CLEVRHybridDataset(
            root_dir=args.dataset.train_data_dir,
            split="validation",
            mode=args.dataset.dataset_mode,
            image_size=args.dataset.resolution,
            download=True,
        )
        train_ds.set_transform(train_augmentations)
        eval_ds.set_transform(eval_augmentations)

    elif args.dataset.dataset_type == "sokoban":
        train_base_ds = SokobanDataset(
            data_path=args.dataset.train_data_dir,
            k=args.dataset.k,
            max_trajectories=args.dataset.max_trajectories
        )
        eval_base_ds = SokobanDataset(data_path=args.dataset.eval_data_dir, k=args.dataset.k)

        num_bits = args.dataset.input_channels
        clip_range = getattr(args, "clip_sample_range", 1.0)

        train_ds = SokobanBitDataset(train_base_ds, num_bits=num_bits, clip_sample_range=clip_range)
        eval_ds = SokobanBitDataset(eval_base_ds, num_bits=num_bits, clip_sample_range=clip_range)

    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset.dataset_type}")

    # 3. Custom Collate Function to handle missing keys cleanly
    def collate_fn(examples):
        batch = {"images": torch.stack([ex["images"] for ex in examples])}

        if "conditions" in examples[0]:
            if isinstance(examples[0]["conditions"], torch.Tensor):
                batch["conditions"] = torch.stack([ex["conditions"] for ex in examples])
            else:
                batch["conditions"] = torch.tensor([ex["conditions"] for ex in examples], dtype=torch.long)
        else:
            batch["conditions"] = None

        if "masks" in examples[0]:
            batch["masks"] = torch.stack([ex["masks"] for ex in examples])
        else:
            batch["masks"] = None

        if "class_labels" in examples[0]:
            if isinstance(examples[0]["class_labels"], torch.Tensor):
                batch["class_labels"] = torch.stack([ex["class_labels"] for ex in examples])
            else:
                batch["class_labels"] = torch.tensor([ex["class_labels"] for ex in examples], dtype=torch.long)
        else:
            batch["class_labels"] = None

        return batch

    # 4. Create standard DataLoaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # 5. Wrap with LimitedLoader
    train_dl = LimitedLoader(train_dl, limit_batches=args.epoch_max_batches_train)
    eval_dl = LimitedLoader(eval_dl, limit_batches=args.epoch_max_batches_eval)

    return train_dl, eval_dl
