import torch
import torch.nn as nn
from monai.losses import DiceCELoss

from utils import get_class


class SegmentationModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = get_class(args.TRAIN.model.generator)(args)
        self.loss_fn = get_class(args.TRAIN.model.loss_fn)

    def forward(self, input, label=None):
        if type(input) is dict:
            return self.loss_fn(self.model(input["image"]), input["label"])
        elif torch.is_tensor(input) and label is None:
            return torch.unbind(self.model(input), dim=1)[0]
        elif torch.is_tensor(input) and torch.is_tensor(label):
            return DiceCELoss(include_background=False, softmax=True, reduction='sum', to_onehot_y=True)(input, label)
        else:
            raise NotImplementedError("Invalid input.")
