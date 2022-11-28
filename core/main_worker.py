import torch
import torch.backends.cudnn as cudnn

import sys
import os
import numpy as np
import random
from copy import deepcopy

from configs import config
from configs.config import cfg
from core.models import setup_model, setup_model_list, setup_crop_model, get_model_type
from core.crop import tune_funcs
from core.dynamic import DynamicInfer
from core.utils import make_divisible
import core.distributed as dist
import core.meters as meters
import data.dataloader as dataloader
import core.logging as logging
import core.optimizer as optim
import core.checkpoint as ck
import core.net as net
import data.transforms as transforms

try:
    import torch.cuda.amp as amp
except ImportError:
    amp = None


logger = logging.get_logger(__name__)


grad_blobs = []


def grad_hook(module, input, output):
    if not hasattr(output, "requires_grad") or not output.requires_grad:
        return

    def _save_grad(grad):
        grad = torch.squeeze(grad)
        #         print('Grad shape：', grad.shape)
        grad_blobs.append(grad.data.cpu().numpy())

    output.register_hook(_save_grad)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_main_proc():
        # Ensure that the output dir exists
        outdir = cfg.OUT_DIR
        if not os.path.isdir(outdir):
            os.makedirs(cfg.OUT_DIR)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log torch, cuda, and cudnn versions
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    env = "".join([f"{key}: {value}\n" for key, value in sorted(os.environ.items())])
    logger.info(f"os.environ:\n{env}")
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg)) if cfg.VERBOSE else ()
    logger.info(logging.dump_log_data(cfg, "cfg", None))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Allocate CUDA memory
    # tmp = torch.randn(256, 1024, 30000).cuda()


@torch.no_grad()
def test_epoch(model, loader, meter, cur_epoch, device):
    """Evaluates the backbone and cropper model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()

    accs = []

    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.to(device), labels.to(device)

        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        if cfg.MODEL.TYPE == 'backbone':
            # For testing image classification model
            scores, correct = meters.topk_confidence(preds, labels)
            top1_acc, top5_acc = meters.topk_accuracy(preds, labels, [1, 5])
            ck.save_confidence(scores, correct)
        else:
            # For testing box predictor
            top1_acc = meters.get_miou(preds, labels)
            top5_acc = torch.tensor(0, dtype=torch.float32).cuda()

        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_acc, top5_acc = dist.scaled_all_reduce([top1_acc, top5_acc], cfg.NUM_GPUS)
        # Copy the errors from GPU to CPU (sync point)
        top1_acc, top5_acc = top1_acc.item(), top5_acc.item()

        accs.append(top1_acc)

        meter.iter_toc()
        # Update and log stats
        meter.update_stats(top1_acc, top5_acc, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)
    np.save('eval_accs.npy', accs)


def eval_model():
    setup_env()
    device = dist.get_device()
    model, start_epoch = setup_model(device)
    test_loader = dataloader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    test_epoch(model, test_loader, test_meter, 0, device)


def crop_test_epoch(model, crop_model, loader, meter, cur_epoch, device):
    model.eval()
    meter.reset()
    meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        cropped_inputs = crop_model(inputs, labels)

        # print(cropped_inputs.shape)

        with torch.no_grad():
            preds = model(cropped_inputs)
        acc1, acc5 = meters.topk_accuracy(preds, labels, (1, 5))
        acc1, acc5 = dist.scaled_all_reduce([acc1, acc5], cfg.NUM_GPUS)
        acc1, acc5 = acc1.item(), acc5.item()

        meter.iter_toc()
        meter.update_stats(acc1, acc5, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    meter.log_epoch_stats(cur_epoch)


def crop_eval():
    setup_env()
    device = dist.get_device()
    model, start_epoch = setup_model(device)
    crop_model = setup_crop_model(device)
    crop_test_loader = dataloader.construct_test_loader()
    test_meter = meters.TestMeter(len(crop_test_loader))
    crop_test_epoch(model, crop_model, crop_test_loader, test_meter, 0, device)


def test_cropper_epoch(model, loader, meter, epoch, device):
    """Test the box predictor for one epoch."""
    model.eval()
    meter.reset()
    meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        # cropped_inputs = cam_crop(inputs, labels)
        # input_size = inputs.size(-1)
        # bboxes = cam_crop.bboxes
        # bboxes = torch.tensor(bboxes, dtype=torch.float, device=device) / input_size

        # Inference
        output = model(inputs)

        # output = meters.combine_boxes(out1, out2)
        acc1, acc5 = meters.get_miou(output, labels), torch.tensor(0., device=device)
        # acc1, acc5 = meters.topk_accuracy(output, labels)
        acc1, acc5 = dist.scaled_all_reduce([acc1, acc5], cfg.NUM_GPUS)
        acc1, acc5 = acc1.item(), acc5.item()
        meter.iter_toc()
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        meter.update_stats(acc1, acc5, mb_size)
        meter.log_iter_stats(epoch, cur_iter)
        meter.iter_tic()

    meter.log_epoch_stats(epoch)


def dynamic_crop_test_epoch(models, crop_model, loader, meter, cur_epoch, device):
    for model in models:
        model.eval()
    meter.reset()
    meter.iter_tic()

    res_mults = cfg.DYNAMIC.RES_MULTS
    err_str = "The number of resolutions should be euqal to the number of models, " \
              "but got {} and {}, respectively.".format(len(res_mults), len(models))
    assert len(res_mults) == len(models), err_str
    # Calculate the resolutions for dynamic inference
    resolutions = list()
    for res_mult in res_mults:
        res = make_divisible(res_mult * cfg.TRAIN.IM_SIZE, cfg.SCALING.ROUND)
        resolutions.append(res)

    dynamic_inference = DynamicInfer(models, resolutions, cfg.DYNAMIC.THRESHOLD_HIGH,
                                     cfg.DYNAMIC.THRESHOLD_LOW, alpha=cfg.DYNAMIC.ALPHA,
                                     count_complexity=cfg.DYNAMIC.COUNT)

    for cur_iter, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        cropped_inputs = crop_model(inputs, labels)

        with torch.no_grad():
            preds = dynamic_inference(cropped_inputs)
        acc1, acc5 = meters.topk_accuracy(preds, labels, (1, 5))
        acc1, acc5 = dist.scaled_all_reduce([acc1, acc5], cfg.NUM_GPUS)
        acc1, acc5 = acc1.item(), acc5.item()

        meter.iter_toc()
        meter.update_stats(acc1, acc5, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    if cfg.DYNAMIC.COUNT:
        logger.info("Average FLOPs: {:.2f}".format(dynamic_inference.flops / dynamic_inference.num_imgs / 1e9 + 0.09))
        logger.info("Average Params: {:.2f}".format(dynamic_inference.params / dynamic_inference.num_imgs / 1e6 + 0.27))
        logger.info("Average Acts: {:.2f}".format(dynamic_inference.acts / dynamic_inference.num_imgs / 1e6 + 1.81))
        logger.info("Images Per Model: {}".format(dynamic_inference.num_imgs_per_model))
    meter.log_epoch_stats(cur_epoch)


def dynamic_crop_eval_model():
    setup_env()
    device = dist.get_device()
    models = setup_model_list(device)
    crop_model = setup_crop_model(device)
    crop_test_loader = dataloader.construct_test_loader()
    crop_test_meter = meters.TestMeter(len(crop_test_loader))
    dynamic_crop_test_epoch(models, crop_model, crop_test_loader, crop_test_meter, 0, device)

