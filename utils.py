import os
from importlib import import_module

from monai.data import load_decathlon_properties


def get_class(x):
    module = x[:x.rfind(".")]
    obj = x[x.rfind(".") + 1:]
    return getattr(import_module(module), obj)


def get_MSD_dataset_properties(args):
    property_keys = [
        "name",
        "description",
        "reference",
        "licence",
        "tensorImageSize",
        "modality",
        "labels",
        "numTraining",
        "numTest",
    ]

    directory = os.path.join(args.GENERAL.root_dir, args.GENERAL.task, "dataset.json")
    properties = load_decathlon_properties(directory, property_keys)

    return properties
