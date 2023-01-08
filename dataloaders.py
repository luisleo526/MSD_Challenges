import os

import torch
from accelerate import Accelerator
from monai.data import ThreadDataLoader, load_decathlon_datalist, CacheDataset
from monai.data.utils import partition_dataset

from transform import get_transforms


def get_dataloaders(args, accelerator: Accelerator, debug=False):
    train_transform = get_transforms("train", args)
    validation_transform = get_transforms("validation", args)

    datalist = load_decathlon_datalist(os.path.join(args.GENERAL.root_dir, args.GENERAL.task, "dataset.json"))
    train_files, val_files = partition_dataset(datalist, seed=args.GENERAL.seed,
                                               ratios=[args.GENERAL.split, 1 - args.GENERAL.split])

    train_files = partition_dataset(train_files,
                                    num_partitions=accelerator.num_processes,
                                    even_divisible=True,
                                    shuffle=True,
                                    seed=args.GENERAL.seed)[accelerator.process_index]

    val_files = partition_dataset(val_files,
                                  num_partitions=accelerator.num_processes,
                                  even_divisible=True,
                                  shuffle=True,
                                  seed=args.GENERAL.seed)[accelerator.process_index]

    if debug:
        train_files = train_files[:5]
        val_files = val_files[:2]

    accelerator.print("Loading dataset...")
    train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=args.GENERAL.cache_rate,
                            num_workers=args.GENERAL.num_workers, progress=accelerator.is_main_process)

    val_ds = CacheDataset(data=val_files, transform=validation_transform, cache_rate=args.GENERAL.cache_rate,
                          num_workers=args.GENERAL.num_workers, progress=accelerator.is_main_process)
    accelerator.wait_for_everyone()
    accelerator.print("Finish loading dataset...")

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
