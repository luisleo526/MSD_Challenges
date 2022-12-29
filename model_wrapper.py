import torch.nn as nn
import torch
from utils import get_class


class SegmentationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = get_class(args.TRAIN.model.generator)(args)
        self.loss_fn = get_class(args.TRAIN.model.loss_fn)

    def forward(self, input, label=None):
        if type(input) is dict:
            loss, pred = self.loss_fn(self.model(input["image"]), input["label"])
        elif torch.is_tensor(input) and torch.is_tensor(label):
            loss, pred = self.loss_fn(self.model(input), label)
        else:
            raise NotImplementedError("Invalid input.")
        return loss, pred
