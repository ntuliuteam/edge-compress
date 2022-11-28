"""Data loader."""

import os

import torch
from configs.config import cfg
from data.imagenet import ImageNet, CroppedImageNet, ImageNetForCrop, ImageNetBox, ImageNetFull, ImageNetTune, ImageNetHP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


# Supported datasets
_DATASETS = {
    "imagenet": ImageNet,
    "cropped_imagenet": CroppedImageNet,
    "imagenet_for_crop": ImageNetForCrop,
    "imagenet_box": ImageNetBox,
    "imagenet_full": ImageNetFull,
    "imagenet_tune": ImageNetTune,
    "imagenet_hp": ImageNetHP,
    "imagenet100": ImageNet,
    "imagenet100_for_crop": ImageNetForCrop,
    "imagenet100_box": ImageNetBox,
    "imagenet100_full": ImageNetFull,
    "imagenet100_hp": ImageNetHP,
}

# Relative data paths to default data directory
_PATHS = {
    "imagenet": "imagenet",
    "cropped_imagenet": "cropped_imagenet/resnet50",
    "imagenet_for_crop": "imagenet",
    "imagenet_box": "imagenet",
    "imagenet_full": "imagenet",
    "imagenet_tune": "imagenet",
    "imagenet_hp": "imagenet",
    "imagenet100": "imagenet_100",
    "imagenet100_for_crop": "imagenet_100",
    "imagenet100_box": "imagenet_100",
    "imagenet100_full": "imagenet_100",
    "imagenet100_hp": "imagenet_100"
}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    # Retrieve the data path for the dataset
    data_path = os.path.join(cfg.DATASET.PATH, _PATHS[dataset_name])
    # Construct the dataset
    dataset = _DATASETS[dataset_name](data_path, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset, shuffle=shuffle) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.DATASET.NAME,
        split='train',
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True,
    )


def construct_crop_train_loader():
    """Train loader for cropping"""
    return _construct_loader(
        dataset_name=cfg.DATASET.NAME,
        split='train',
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.DATASET.NAME,
        split='val',
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """ "Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)


def get_class_size(split):
    dataset_name = cfg.DATASET.NAME
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    splits = ['train', 'val']
    assert split in splits, "Split '{}' not supported.".format(split)
    split_path = os.path.join(cfg.DATASET.PATH, _PATHS[dataset_name], split)
    classes = os.listdir(split_path)
    classes.sort()
    class_sizes = []

    for class_name in classes:
        class_path = os.path.join(split_path, class_name)
        class_size = len(os.listdir(class_path))
        class_sizes.append(class_size)

    return class_sizes
