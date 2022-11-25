import logging
import math
from argparse import ArgumentParser

import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from munch import DefaultMunch
from tqdm.auto import tqdm

from create_dynunet import get_network
from dataloaders import get_dataloaders
from loss_fn import loss_fn
from utils import get_class

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
    model = get_network(args).to(device)
    optimizer = get_class(args.TRAIN.optimizer.type)(model.parameters(), **args.TRAIN.optimizer.params)
    scheduler = get_class(args.TRAIN.scheduler.type)(optimizer, **args.TRAIN.scheduler.params)

    # for metrics
    post_pred = AsDiscrete(to_onehot=model.out_channels, argmax=True)
    post_label = AsDiscrete(to_onehot=model.out_channels, argmax=False)
    metrics = DiceMetric(include_background=False, reduction='mean')

    model = accelerator.prepare_model(model)
    optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        optimizer, scheduler, train_loader, val_loader
    )

    max_train_steps = args.TRAIN.max_epochs * math.ceil(len(train_loader) / args.TRAIN.gradient_accumulation_steps)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.TRAIN.max_epochs):
        model.train()
        for batch in train_loader:

            with accelerator.accumulate(model):
                pred = model(batch["image"])
                loss, pred = loss_fn(pred, batch["label"])
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pred, target = accelerator.gather_for_metrics((pred, batch["label"]))
            pred = [post_pred(i) for i in pred]
            target = [post_label(i) for i in target]
            metrics(pred, target)

            if accelerator.sync_gradients:
                progress_bar.update(1)

        logger.info(f"MeanDice Score: {list(metrics.aggregate(reduction='mean_batch').cpu().numpy())}")


if __name__ == '__main__':
    with open(parse_args().yaml, "r") as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    args = DefaultMunch.fromDict(data)
    main(args)