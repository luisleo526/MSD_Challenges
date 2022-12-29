import torch
from monai.apps import DecathlonDataset
from monai.data import ThreadDataLoader

from transform import get_transforms


def get_dataloaders(args):

    train_transform = get_transforms("train", args)

    validation_transform = get_transforms("validation", args)

    train_ds = DecathlonDataset(
        root_dir=args.GENERAL.root_dir,
        task=args.GENERAL.task,
        section="training",
        transform=train_transform,
        download=args.GENERAL.download,
        seed=args.GENERAL.seed,
        val_frac=1-args.GENERAL.split,
        cache_rate=args.GENERAL.cache_rate,
        num_workers=args.GENERAL.num_workers,
    )

    val_ds = DecathlonDataset(
        root_dir=args.GENERAL.root_dir,
        task=args.GENERAL.task,
        section="validation",
        transform=validation_transform,
        download=False,
        seed=args.GENERAL.seed,
        val_frac=1-args.GENERAL.split,
        cache_rate=args.GENERAL.cache_rate,
        num_workers=args.GENERAL.num_workers,
    )

    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=args.TRAIN.batch_size,
        num_workers=args.GENERAL.num_workers,
        pin_memory=torch.cuda.is_available(),
        use_thread_workers=True,
        buffer_size=2
    )

    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=1,
        num_workers=args.GENERAL.num_workers,
        pin_memory=torch.cuda.is_available(),
        use_thread_workers=True,
        buffer_size=2
    )

    return train_loader, val_loader

