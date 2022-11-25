import torch
from monai.losses import DiceCELoss


def loss_fn(pred, target):
    loss_fn = DiceCELoss(include_background=False, softmax=True, reduction='sum', to_onehot_y=True)
    pred = torch.unbind(pred, dim=1)
    loss = sum([0.5 ** i * loss_fn(p, target) for i, p in enumerate(pred)])
    return loss, pred[0]
