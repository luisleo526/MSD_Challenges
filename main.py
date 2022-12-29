import logging
from argparse import ArgumentParser
from datetime import datetime

import math
import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from munch import DefaultMunch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataloaders import get_dataloaders
from model_wrapper import SegmentationModel
from utils import get_class, get_MSD_dataset_properties

logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--yaml", type=str, default="default.yaml")
    args = parser.parse_args()
    return args


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.TRAIN.gradient_accumulation_steps)
    device = accelerator.device
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    train_loader, val_loader = get_dataloaders(args)
    model = SegmentationModel(args).to(device)
    optimizer = get_class(args.TRAIN.optimizer.type)(model.parameters(), **args.TRAIN.optimizer.params)
    scheduler = get_class(args.TRAIN.scheduler.type)(optimizer, **args.TRAIN.scheduler.params)
    if accelerator.is_main_process:
        writer = SummaryWriter(f"./logs/{args.GENERAL.task}/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    else:
        writer = None

    # for metrics
    properties = get_MSD_dataset_properties(args)
    labels = properties["labels"]
    post_pred = AsDiscrete(to_onehot=len(labels), argmax=True)
    post_label = AsDiscrete(to_onehot=len(labels), argmax=False)
    metrics = DiceMetric(include_background=False, reduction='mean')

    model = accelerator.prepare_model(model)
    optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        optimizer, scheduler, train_loader, val_loader
    )

    max_train_steps = args.TRAIN.max_epochs * math.ceil(len(train_loader) / args.TRAIN.gradient_accumulation_steps)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    step = 0
    for epoch in range(args.TRAIN.max_epochs):
        step = step + 1

        results = dict(total_loss=dict(train=0, test=0))
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss, pred = model(batch)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            results['total_loss']['train'] += accelerator.gather(loss.detach().float()).item()
            pred = accelerator.gather(pred.contiguous())
            target = accelerator.gather(batch["label"].contiguous())
            pred = [post_pred(i) for i in pred]
            target = [post_label(i) for i in target]
            metrics(pred, target)

        for i, score in list(metrics.aggregate(reduction='mean_batch').cpu().numpy()):
            results[labels[str(i + 1)]] = dict(train=score, test=0)
        metrics.reset()

        model.eval()
        for batch in val_loader:
            with torch.no_grad:
                loss, pred = sliding_window_inference(inputs=batch, roi_size=args.TRANSFORM.patch_size,
                                                      sw_batch_size=args.TRAIN.batch_size * args.TRANSFORM.num_samples,
                                                      predictor=model)

                results['total_loss']['test'] += accelerator.gather(loss.detach().float()).item()
                pred = accelerator.gather(pred.contiguous())
                target = accelerator.gather(batch["label"].contiguous())
                pred = [post_pred(i) for i in pred]
                target = [post_label(i) for i in target]
                metrics(pred, target)

        for i, score in list(metrics.aggregate(reduction='mean_batch').cpu().numpy()):
            results[labels[str(i + 1)]]['test'] = score

        if accelerator.is_main_process:
            for key, value in results.items():
                writer.add_scalars(key, value, global_step=step)

        if accelerator.sync_gradients:
            progress_bar.update(1)


if __name__ == '__main__':
    with open(parse_args().yaml, "r") as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    args = DefaultMunch.fromDict(data)
    main(args)
