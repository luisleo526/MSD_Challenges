from accelerate import Accelerator
from torch.optim import Optimizer
from utils import get_class


class Scheduler:

    def __init__(self, optimizer: Optimizer, accelerator: Accelerator, args):
        schedulers = [get_class(scheduler.type)(optimizer, **scheduler.params) for scheduler in args.TRAIN.scheduler]
        schedulers = [accelerator.prepare_scheduler(x) for x in schedulers]
        milestones = [scheduler.end * args.TRAIN.max_epochs for scheduler in args.TRAIN.scheduler]

        self.schedulers = dict(zip(milestones, schedulers))

    def step(self, epoch, **kwargs):

        for milestone, scheduler in self.schedulers.items():
            if epoch <= milestone:
                scheduler.step(**kwargs)
                break
