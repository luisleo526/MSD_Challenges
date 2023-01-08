import logging
import math
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from munch import DefaultMunch
from tqdm.auto import tqdm

import wandb
from dataloaders import get_dataloaders
from model_wrapper import SegmentationModel
from scheduler import Scheduler
from utils import get_class, get_MSD_dataset_properties

logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--yaml", type=str, default="default.yaml")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    return args


def main():
    opt = parse_args()

    with open(opt.yaml, "r") as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    args = DefaultMunch.fromDict(data)

    accelerator = Accelerator(gradient_accumulation_steps=args.TRAIN.gradient_accumulation_steps,
                              step_scheduler_with_optimizer=False)
    device = accelerator.device
    set_seed(args.GENERAL.seed, False)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    train_loader, val_loader = get_dataloaders(args, accelerator, debug=opt.debug)
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    max_train_steps = args.TRAIN.max_epochs * math.ceil(len(train_loader) / args.TRAIN.gradient_accumulation_steps)

    model = SegmentationModel(args).to(device)
    optimizer = get_class(args.TRAIN.optimizer.type)(model.parameters(), **args.TRAIN.optimizer.params)
    scheduler = Scheduler(optimizer, accelerator, args)
    optimizer, model = accelerator.prepare(optimizer, model)

    if accelerator.is_main_process:
        name = args.GENERAL.task
        if opt.debug:
            name += "-debug"
        wandb.init(project=name, entity="luisleo", config=args)
        wandb.define_metric("dice/*", summary="max")

    # for metrics
    properties = get_MSD_dataset_properties(args)
    labels = properties["labels"]
    labels = {int(key): value for key, value in labels.items()}
    post_pred = AsDiscrete(to_onehot=len(labels), argmax=True)
    post_label = AsDiscrete(to_onehot=len(labels), argmax=False)
    metrics = DiceMetric(include_background=False, reduction='mean')

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.TRAIN.max_epochs):

        sample_id = np.random.randint(0, len(val_loader))

        if opt.debug:
            logger.info(" *** training *** ")

        results = {"loss/train": 0.0}
        model.train()
        for batch_id, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                loss, pred = model(batch)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            results['loss/train'] += accelerator.gather(loss.detach().float()).item()
            pred = accelerator.gather(pred.contiguous())
            target = accelerator.gather(batch["label"].contiguous())
            pred = [post_pred(i) for i in pred]
            target = [post_label(i) for i in target]
            metrics(pred, target)

            if accelerator.sync_gradients:
                progress_bar.update(1)

        scheduler.step(epoch=epoch)

        for i, score in enumerate(list(metrics.aggregate(reduction='mean_batch').cpu().numpy())):
            results[f"dice/{labels[i + 1]}/train"] = score
        metrics.reset()

        if opt.debug:
            logger.info(" *** testing *** ")

        if epoch % 5 == 0 or opt.debug:
            results['loss/test'] = 0
            model.eval()
            for batch_id, batch in enumerate(val_loader):
                with torch.no_grad():
                    pred = sliding_window_inference(inputs=batch['image'], roi_size=args.TRANSFORM.patch_size,
                                                    sw_batch_size=args.TRAIN.batch_size * args.TRANSFORM.num_samples,
                                                    predictor=model)
                    loss = model(pred, batch['label'])

                    if accelerator.is_main_process and batch_id == sample_id:
                        prediction = pred.argmax(1)
                        accelerator.print(batch['label'].shape, prediction.shape)
                        size = pred.shape[-1]
                        total_slices = min(args.GENERAL.num_slices_to_show, size)
                        images = []
                        for slice_num in range(total_slices):
                            slice_pos = (size // total_slices) * slice_num
                            image = batch['image'][0, 0, :, :, slice_pos].permute(1, 0).cpu().numpy()
                            label = batch['label'][0, 0, :, :, slice_pos].permute(1, 0).cpu().numpy()
                            plabel = prediction[0, :, :, slice_pos].permute(1, 0).cpu().numpy()
                            mask_img = wandb.Image(image, caption=f"image @ {slice_pos} / {size}",
                                                   masks={"ground_truth": {"mask_data": label, "class_labels": labels},
                                                          "prediction": {"mask_data": plabel, "class_labels": labels}})
                            images.append(mask_img)
                        results["samples"] = images

                    results['loss/test'] += accelerator.gather(loss.detach().float()).item()
                    pred = accelerator.gather(pred.contiguous())
                    target = accelerator.gather(batch["label"].contiguous())
                    pred = [post_pred(i) for i in pred]
                    target = [post_label(i) for i in target]
                    metrics(pred, target)

            for i, score in enumerate(list(metrics.aggregate(reduction='mean_batch').cpu().numpy())):
                results[f"dice/{labels[i + 1]}/test"] = score
            metrics.reset()

        if accelerator.is_main_process:
            wandb.log(results)


if __name__ == '__main__':
    main()
