"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
from yacs.config import CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ---------------------------------- Model options ----------------------------------- #
_C.MODEL = CfgNode()

# Model name
_C.MODEL.NAME = "resnet50"

# Model type. Select from ['backbone', 'cropper']
_C.MODEL.TYPE = "backbone"

# Number of weight layers
_C.MODEL.DEPTH = 0

# Number of classes
_C.MODEL.NUM_CLASSES = 1000

# Loss function (see core/optimizer.py for options)
_C.MODEL.LOSS_FUN = "soft_cross_entropy"

# Activation function (relu or silu/swish)
_C.MODEL.ACTIVATION_FUN = "relu"

# The temperature hyperparameter of Gumbel-softmax trix
_C.MODEL.TAU = 1.0

# Perform activation inplace if implemented
_C.MODEL.ACTIVATION_INPLACE = True

# Locally saved pretrained weights
_C.MODEL.BOX_WEIGHTS = ''
_C.MODEL.TUNE_WEIGHTS = ''
_C.MODEL.CROPNET_FEATURE_WEIGHTS = ''
_C.MODEL.WEIGHTS = ''
_C.MODEL.CAM_WEIGHTS = ''


# Download weights from cloud. Only applicable for ResNet
# Priority: MODEL.WEIGHTS > MODEL.LOAD_WEIGHTS_FROM_URL
_C.MODEL.LOAD_WEIGHTS_FROM_URL = False


# ---------------------------------- ResNet options ---------------------------------- #
_C.RESNET = CfgNode()

# Transformation function (see pycls/models/preresnet.py for options)
_C.RESNET.TRANS_FUN = "basic_transform"

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt)
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
_C.RESNET.STRIDE_1X1 = True


# ---------------------------------- AnyNet options ---------------------------------- #
_C.ANYNET = CfgNode()

# Stem type
_C.ANYNET.STEM_TYPE = "simple_stem_in"

# Stem width
_C.ANYNET.STEM_W = 32

# Block type
_C.ANYNET.BLOCK_TYPE = "res_bottleneck_block"

# Depth for each stage (number of blocks in the stage)
_C.ANYNET.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.ANYNET.WIDTHS = []

# Strides for each stage (applies to the first block of each stage)
_C.ANYNET.STRIDES = []

# Bottleneck multipliers for each stage (applies to bottleneck block)
_C.ANYNET.BOT_MULS = []

# Group widths for each stage (applies to bottleneck block)
_C.ANYNET.GROUP_WS = []

# Head width for first conv in head (if 0 conv is omitted, as is the default)
_C.ANYNET.HEAD_W = 0

# Whether SE is enabled for res_bottleneck_block
_C.ANYNET.SE_ON = False

# SE ratio
_C.ANYNET.SE_R = 0.25


# ---------------------------------- RegNet options ---------------------------------- #
_C.REGNET = CfgNode()

# Stem type
_C.REGNET.STEM_TYPE = "simple_stem_in"

# Stem width
_C.REGNET.STEM_W = 32

# Block type
_C.REGNET.BLOCK_TYPE = "res_bottleneck_linear_block"

# Stride of each stage
_C.REGNET.STRIDE = 2

# Squeeze-and-Excitation (RegNetY)
_C.REGNET.SE_ON = True
_C.REGNET.SE_R = 0.25

# Depth
_C.REGNET.DEPTH = 21

# Initial width
_C.REGNET.W0 = 16

# Slope
_C.REGNET.WA = 10.7

# Quantization
_C.REGNET.WM = 2.51

# Group width
_C.REGNET.GROUP_W = 4

# Bottleneck multiplier (bm = 1 / b from the paper)
_C.REGNET.BOT_MUL = 4.0

# Head width for first conv in head (if 0 conv is omitted, as is the default)
_C.REGNET.HEAD_W = 1024


# ------------------------------- EfficientNet options ------------------------------- #
_C.EN = CfgNode()

# Stem width
_C.EN.STEM_W = 32

# Depth for each stage (number of blocks in the stage)
_C.EN.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.EN.WIDTHS = []

# Expansion ratios for MBConv blocks in each stage
_C.EN.EXP_RATIOS = []

# Squeeze-and-Excitation (SE) ratio
_C.EN.SE_R = 0.25

# Strides for each stage (applies to the first block of each stage)
_C.EN.STRIDES = []

# Kernel sizes for each stage
_C.EN.KERNELS = []

# Head width
_C.EN.HEAD_W = 1280

# Drop connect ratio
_C.EN.DC_RATIO = 0.0

# Dropout ratio
_C.EN.DROPOUT_RATIO = 0.0


# -------------------------------- Batch norm options -------------------------------- #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 8192

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

# -------------------------------- Layer norm options -------------------------------- #
_C.LN = CfgNode()

# LN epsilon
_C.LN.EPS = 1e-5

# Use a different weight decay for LN layers
_C.LN.USE_CUSTOM_WEIGHT_DECAY = False
_C.LN.CUSTOM_WEIGHT_DECAY = 0.0

# -------------------------------- Optimizer options --------------------------------- #
_C.OPTIM = CfgNode()

# Type of optimizer. Select from ['sgd', 'adam', 'adamw']
_C.OPTIM.OPTIMIZER = 'sgd'

# Learning rate ranges from BASE_LR to MIN_LR*BASE_LR according to the LR_POLICY
# Learning rate of the classification model
_C.OPTIM.BASE_LR = 2.0
_C.OPTIM.MIN_LR = 5e-4
# Learning rate of the box tuning predictor
_C.OPTIM.TUNE_LR = 0.001

# Learning rate policy select from {'cos', 'exp', 'lin', 'steps'}
_C.OPTIM.LR_POLICY = "exp"

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = [0, 100, 200, 250]

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 100

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening. For SGD
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum. For SGD
_C.OPTIM.NESTEROV = False

# Betas (for Adam/AdamW optimizer)
_C.OPTIM.BETA1 = 0.9
_C.OPTIM.BETA2 = 0.999

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 2e-5

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# Exponential Moving Average (EMA) update value
_C.OPTIM.EMA_ALPHA = 1e-5

# Iteration frequency with which to update EMA weights
_C.OPTIM.EMA_UPDATE_PERIOD = 0

# Use a different weight decay for all biases (excluding those in BN/LN layers)
_C.OPTIM.BIAS_USE_CUSTOM_WEIGHT_DECAY = False
_C.OPTIM.BIAS_CUSTOM_WEIGHT_DECAY = 0.0

# --------------------------------- Training options --------------------------------- #
_C.TRAIN = CfgNode()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 1024

# Image size
_C.TRAIN.IM_SIZE = 224

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = False

# Weights to start training from
# _C.TRAIN.WEIGHTS = ""

# If True train using mixed precision
_C.TRAIN.MIXED_PRECISION = False

# Label smoothing value in 0 to 1 where (0 gives no smoothing)
_C.TRAIN.LABEL_SMOOTHING = 0.1

# Batch mixup regularization value in 0 to 1 (0 gives no mixup)
_C.TRAIN.MIXUP_ALPHA = 0.2

# Standard deviation for AlexNet-style PCA jitter (0 gives no PCA jitter)
_C.TRAIN.PCA_STD = 0.1

# Data augmentation to use ("", "AutoAugment", "RandAugment_N2_M0.5", etc.)
_C.TRAIN.AUGMENT = ""


# --------------------------------- Testing options ---------------------------------- #
_C.TEST = CfgNode()

_C.TEST.SPLIT = 'val'

# Total mini-batch size
_C.TEST.BATCH_SIZE = 400

_C.TEST.IM_SIZE = 256

# ----------------------------------- Scaling options -------------------------------- #
_C.SCALING = CfgNode()

# The scaling budget for the all dimensions
# Resolution scaling factor of the bounding box predictor
_C.SCALING.RES_MULT_A = 1.0

_C.SCALING.DOWN_RATIO = 4.0

# Resolution scaling factor of the classifier
_C.SCALING.RES_MULT_B = 1.0

_C.SCALING.WIDTH_MULT = 1.0

_C.SCALING.INTERPOLATE = 'bilinear'

# Make the scaled resolution divisible by a certain number for higher hardware efficiency.
_C.SCALING.ROUND = 4


# ----------------------------------- Cropping options -------------------------------- #
_C.CROP = CfgNode()

# Cropping methods. Select from ['camcrop', 'cropnet_resnet18', ...]
_C.CROP.TYPE = 'camcrop'
_C.CROP.CAM = 'gradcam'
_C.CROP.CAM_MODEL = 'resnet50'
_C.CROP.CAM_SPLIT = 0.5

# Regulate the predicted boxes into the closest predefined size
_C.CROP.REGULATE = False
_C.CROP.BOX_SIZES = [0.25, 0.5, 0.75, 1.0]

# The shape of generated box. Select from ['rect', 'square']
_C.CROP.SHAPE = 'rect'

# _C.CROP.SAVE = './tmp'

_C.CROP.SPLIT = 'train'

# The tuning model
_C.CROP.TUNE_MODEL = ''
_C.CROP.TUNE_STRIDE = 25
_C.CROP.TUNE_PROB = 0.5


# ----------------------------------- Dynamic inference options -------------------------------- #

_C.DYNAMIC = CfgNode()

# The confidence thresholds
_C.DYNAMIC.THRESHOLD_HIGH = 0.3

_C.DYNAMIC.THRESHOLD_LOW = 0.0

_C.DYNAMIC.ALPHA = 1.0

# If to count the computational complexity of dynamic inference
_C.DYNAMIC.COUNT = False

# Model name list for dynamic inference
_C.DYNAMIC.MODELS = []

# Width scaling factor list for dynamic inference
_C.DYNAMIC.WIDTH_MULTS = []

# Pretrained weight path list for dynamic inference
_C.DYNAMIC.WEIGHTS = []

# Resolution scaling factor list for dynamic inference
_C.DYNAMIC.RES_MULTS = []

# --------------------------------- Dataset options ---------------------------------- #
_C.DATASET = CfgNode()

# Dataset name ('imagenet', 'cropped_imagenet', 'imagenet_for_crop', 'imagenetbox', 'imagenetfull')
_C.DATASET.NAME = 'imagenet'

_C.DATASET.PATH = '/home/hao/dataFolder/DATA/datasets/'

# ------------------------------- Data loader options -------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------- CUDNN options ----------------------------------- #
_C.CUDNN = CfgNode()

# Perform benchmarking to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True


# ------------------------------- Precise time options ------------------------------- #
_C.PREC_TIME = CfgNode()

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ---------------------------------- Launch options ---------------------------------- #
_C.LAUNCH = CfgNode()

# The launch mode, may be 'local' or 'slurm' (or 'submitit_local' for debugging)
# The 'local' mode uses a multi-GPU setup via torch.multiprocessing.run_processes.
# The 'slurm' mode uses submitit to launch a job on a SLURM cluster and provides
# support for MULTI-NODE jobs (and is the only way to launch MULTI-NODE jobs).
# In 'slurm' mode, the LAUNCH options below can be used to control the SLURM options.
# Note that NUM_GPUS (not part of LAUNCH options) determines total GPUs requested.
_C.LAUNCH.MODE = "local"

# Launch options that are only used if LAUNCH.MODE is 'slurm'
_C.LAUNCH.MAX_RETRY = 3
_C.LAUNCH.NAME = "pycls_job"
_C.LAUNCH.COMMENT = ""
_C.LAUNCH.CPUS_PER_GPU = 10
_C.LAUNCH.MEM_PER_GPU = 60
_C.LAUNCH.PARTITION = "devlab"
_C.LAUNCH.GPU_TYPE = "volta"
_C.LAUNCH.TIME_LIMIT = 4200
_C.LAUNCH.EMAIL = ""


# ----------------------------------- Misc options ----------------------------------- #
# Optional description of a config
_C.DESC = ""

# If True output additional info to log
_C.VERBOSE = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.MAX_GPUS_PER_NODE = 1

# Output directory
_C.OUT_DIR = "./tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism is still be present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port range for multi-process groups (actual port selected randomly)
_C.HOST = "localhost"
_C.PORT_RANGE = [10000, 65000]

# Models weights referred to by URL are downloaded to this local cache
_C.DOWNLOAD_CACHE = "/tmp/pycls-download-cache"

# Allocate memory in advance
_C.TRACK = 0

# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


# --------------------------------- Deprecated keys ---------------------------------- #
_C.register_deprecated_key("MEM")
_C.register_deprecated_key("MEM.RELU_INPLACE")
_C.register_deprecated_key("OPTIM.GAMMA")
_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")
_C.register_deprecated_key("PREC_TIME.ENABLED")
_C.register_deprecated_key("PORT")
_C.register_deprecated_key("TRAIN.EVAL_PERIOD")
_C.register_deprecated_key("TRAIN.CHECKPOINT_PERIOD")


def assert_cfg():
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
    err_str = "NUM_GPUS must be divisible by or less than MAX_GPUS_PER_NODE"
    num_gpus, max_gpus_per_node = _C.NUM_GPUS, _C.MAX_GPUS_PER_NODE
    assert num_gpus <= max_gpus_per_node or num_gpus % max_gpus_per_node == 0, err_str
    err_str = "Invalid mode {}".format(_C.LAUNCH.MODE)
    assert _C.LAUNCH.MODE in ["local", "submitit_local", "slurm"], err_str


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


