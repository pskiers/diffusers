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
    train_augmentations = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution) if not args.center_crop else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    eval_augmentations = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 2. Determine Dataset Type
    if args.dataset_type == "hf":
        train_ds = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, split="train")
        eval_ds = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, split=args.test_split_name)

        is_conditional = args.num_classes is not None and args.num_classes > 0

        def hf_train_transform(examples):
            images = [train_augmentations(image.convert("RGB")) for image in examples[args.img_key]]
            out = {"images": images}
            if is_conditional:
                out["conditions"] = examples[args.class_key]
            return out

        def hf_eval_transform(examples):
            images = [eval_augmentations(image.convert("RGB")) for image in examples[args.img_key]]
            out = {"images": images}
            if is_conditional:
                out["conditions"] = examples[args.class_key]
            return out

        train_ds.set_transform(hf_train_transform)
        eval_ds.set_transform(hf_eval_transform)

    elif args.dataset_type == "clevr":
        train_ds = CLEVRHybridDataset(root_dir=args.train_data_dir, split="train", mode=args.dataset_mode, image_size=args.resolution, download=False)
        eval_ds = CLEVRHybridDataset(root_dir=args.train_data_dir, split="validation", mode=args.dataset_mode, image_size=args.resolution, download=False)
        train_ds.set_transform(train_augmentations)
        eval_ds.set_transform(eval_augmentations)

    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

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

        return batch

    # 4. Create standard DataLoaders
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True, collate_fn=collate_fn)
    eval_dl = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers, drop_last=True, collate_fn=collate_fn)

    # 5. Wrap with LimitedLoader
    train_dl = LimitedLoader(train_dl, limit_batches=args.epoch_max_batches_train)
    eval_dl = LimitedLoader(eval_dl, limit_batches=args.epoch_max_batches_eval)

    return train_dl, eval_dl
