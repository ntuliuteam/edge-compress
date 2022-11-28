import csv
import os
import numpy as np
import torch

import core.distributed as dist
from configs.config import cfg
from core.crop import arrange_bboxes
from core.net import unwrap_model


def get_ckpt_dir():
    """Retrieve the saving directory for the given model."""
    return os.path.join(cfg.OUT_DIR, 'checkpoints', cfg.MODEL.NAME)


def get_ckpt_path(epoch, acc, prefix='model'):
    """Return the saving path of checkpoint"""
    fname = prefix + '_epoch_{:03d}_acc_{:.2f}.ckpt'.format(epoch, acc)
    return os.path.join(get_ckpt_dir(), fname)


def process_and_save_boxes(boxes, split):
    """Sort the boxes obtained from crop_model() and save the sorted boxes to
    the output directory.
    """
    fpath = os.path.join(cfg.OUT_DIR, cfg.CROP.TYPE + '_' + cfg.CROP.SHAPE + '_' + split + '.npy')
    if dist.is_main_proc():
        np.save(fpath, boxes)
        sorted_boxes = arrange_bboxes(boxes, split)
        np.save(fpath, sorted_boxes)


def save_ckpt(model, model_ema, optimizer, epoch, test_acc, ema_acc, model_type='backbone'):
    """Saves a checkpoint and also the best weights so far in a best checkpoint."""
    # Save checkpoints only from the main process
    if not dist.is_main_proc():
        return
    # Ensure that the checkpoint dir exists
    save_dir = get_ckpt_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "test_acc": test_acc,
        "ema_acc": ema_acc,
        "model_state": unwrap_model(model).state_dict(),
        "ema_state": unwrap_model(model_ema).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_ckpt_path(epoch + 1, max(test_acc, ema_acc), prefix=model_type)
    torch.save(checkpoint, checkpoint_file)

    return checkpoint_file


def save_confidence(scores, corrects):
    """Saves the confidence score and the classification results.
    """
    fpath = cfg.OUT_DIR + '/confidence.csv'
    with open(fpath, 'a') as csvfile:
        writter = csv.writer(csvfile)
        for idx, (score, correct) in enumerate(zip(scores, corrects)):
            writter.writerow([score, correct])


# if __name__ == '__main__':
#     split = 'train'
#     fpath = os.path.join('../tmp', cfg.CROP.SHAPE + '_' + split + '.npy')
#     boxes = np.load(fpath, allow_pickle=True)
#     sorted_boxes = arrange_bboxes(boxes, split)
#     np.save(fpath, sorted_boxes)
