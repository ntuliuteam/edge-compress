from __future__ import print_function

import argparse
import sys

import configs.config as config
from configs.config import cfg
import core.distributed as dist
from core.main_worker import (
    eval_model,
    crop_eval,
    crop_save,
    train_model,
    crop_train,
    train_cropper,
    train_hp,
    dynamic_crop_eval_model
)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str,
                        help="The execution configuration file.")
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'eval', 'crop_train', 'crop_eval', 'crop', 'dy_eval'],
                        default='eval', help="Running mode in ['train', 'eval', 'crop_train', 'crop_eval', 'crop', 'dy_eval'].")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See configs/config.py for all options")

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    return parser.parse_args()


def main():
    """The main worker."""
    args = build_args()
    mode = args.mode
    config.load_cfg(args.config)
    cfg.merge_from_list(args.opts)
    config.assert_cfg()
    cfg.freeze()

    if mode == 'train':
        # Normal training process.
        if cfg.MODEL.TYPE == 'cropper':
            dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=train_cropper)
        elif cfg.MODEL.TYPE == 'hp':
            dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=train_hp)
        else:
            dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=train_model)
    elif mode == 'eval':
        # Normal evaluation process.
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=eval_model)
    elif mode == 'crop_train':
        # Take normal-size images and perform the pipeline of cropping and training
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=crop_train)
    elif mode == 'crop_eval':
        # Take normal-size images and perform the pipeline of cropping and evaluation.
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=crop_eval)
    elif mode == 'crop':
        # Crop the given dataset and save the boxes.
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=crop_save)
    elif mode == 'dy_eval':
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=dynamic_crop_eval_model)


if __name__ == '__main__':
    main()
