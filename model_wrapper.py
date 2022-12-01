import torch.nn as nn

from utils import get_class


class SegmentationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = get_class(args.TRAIN.model.generator)(args)
        self.loss_fn = get_class(args.TRAIN.model.loss_fn)

    def forward(self, batch):
        loss, pred = self.loss_fn(self.model(batch["image"]), batch["label"])

        return loss, pred
