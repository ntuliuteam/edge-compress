import logging
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.resnet import resnet18, resnet50, resnet34
from networks.regnet import regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf
from networks.anynet import anynet_light
from networks.efficientnet import EffNet
from networks.blocks import (
    linear,
    conv2d_cx,
    gap2d_cx,
    pool2d_cx,
    linear_cx,
    init_weights,
)
from configs.config import cfg


# try:
#     from torch.hub import load_state_dict_from_url  # noqa: 401
# except ImportError:
#     from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401


_networks = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'regnet_x_800mf': regnet_x_800mf,
    'regnet_x_1_6gf': regnet_x_1_6gf,
    'regnet_x_3_2gf': regnet_x_3_2gf,
    'effnet': EffNet,
    'anynet_light': anynet_light,
}


class BasicHead(nn.Module):
    def __init__(self, arch, inplanes, outplanes, config):
        super().__init__()
        base_config = []
        self.arch = arch
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.config = config if config else base_config
        self.layers = self._make_layers()

    def _make_layers(self):
        layers = []
        for outplanes in self.config:
            fc = linear(self.inplanes, outplanes, bias=True)
            relu = nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
            layers.extend([fc, relu])
            self.inplanes = outplanes
        layers.extend([linear(self.inplanes, self.outplanes)])

        return nn.Sequential(*layers)


class BoxHead(BasicHead):
    def __init__(self, inplanes, config=None):
        super().__init__('box', inplanes, 4, config)
        layers = list(self.layers)
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TuneHead(BasicHead):
    def __init__(self, inplanes, config=None):
        super().__init__('tune', inplanes, cfg.MODEL.NUM_CLASSES, config)
        self.box_fc1 = linear(4, 128, bias=True)
        self.relu = nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
        self.box_fc2 = linear(128, cfg.MODEL.NUM_CLASSES, bias=True)

    def forward(self, x, box=None):
        out1 = self.layers(x)

        if box is not None:
            out2 = self.box_fc1(box)
            out2 = self.relu(out2)
            out2 = self.box_fc2(out2)
            out1 += out2

        return out1


class CropNetJoint(nn.Module):
    def __init__(self, feature, head_a, head_b):
        super(CropNetJoint, self).__init__()
        self.feature = feature
        self.head_a = head_a.layers
        self.head_b = head_b.layers
        self.apply(init_weights)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        pred_a = self.head_a(x)
        pred_b = self.head_b(x)

        return pred_a, pred_b

    def complexity(self, cx):
        inplanes = 0
        # The complexity of feature extractor.
        for m in self.feature.children():
            if isinstance(m, nn.Conv2d):
                cx = conv2d_cx(cx, m.in_channels, m.out_channels, m.kernel_size[0],
                               stride=m.stride[0], groups=m.groups, bias=m.bias)
                inplanes = m.out_channels
            elif isinstance(m, nn.MaxPool2d):
                cx = pool2d_cx(cx, inplanes, m.kernel_size, stride=m.stride)
            elif isinstance(m, nn.Sequential):
                for n in m.children():
                    for k in n.children():
                        cx = k.complexity(cx)
                        inplanes = k.outplanes
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                cx = gap2d_cx(cx, inplanes)
            elif isinstance(m, nn.Linear):
                cx = linear_cx(cx, m.in_features, m.out_features, bias=(m.bias is not None))

        # The complexity of the heads.
        for m in self.box_head.children():
            if isinstance(m, nn.Linear):
                cx = linear_cx(cx, m.in_features, m.out_features, bias=(m.bias is not None))

        for m in self.tune_head.children():
            if isinstance(m, nn.Linear):
                cx = linear_cx(cx, m.in_features, m.out_features, bias=(m.bias is not None))

        return cx


class CropNetSep(nn.Module):
    def __init__(self, feature, head):
        super(CropNetSep, self).__init__()
        self.feature = feature
        self.head_arch = head.arch
        if self.head_arch == 'tune':
            self.head = head
        else:
            self.head = head.layers
        self.apply(init_weights)

    def forward(self, x, box=None):
        x = self.feature(x)
        x = torch.flatten(x, 1)

        if self.head_arch == 'tune':
            # err_str = "The box must not be None for the tune head."
            # assert box is not None, err_str
            x = self.head(x, box=box)
        else:
            # For box prediction head
            x = self.head(x)

        return x

    def complexity(self, cx):
        inplanes = 0
        # The complexity of feature extractor.
        for m in self.feature.children():
            if isinstance(m, nn.Conv2d):
                cx = conv2d_cx(cx, m.in_channels, m.out_channels, m.kernel_size[0],
                               stride=m.stride[0], groups=m.groups, bias=m.bias)
                inplanes = m.out_channels
            elif isinstance(m, nn.MaxPool2d):
                cx = pool2d_cx(cx, inplanes, m.kernel_size, stride=m.stride)
            elif isinstance(m, nn.Sequential):
                for n in m.children():
                    if isinstance(n, nn.Sequential):
                        for k in n.children():
                            cx = k.complexity(cx)
                            inplanes = k.outplanes
                    else:
                        cx = n.complexity(cx)
                        inplanes = n.outplanes
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                cx = gap2d_cx(cx, inplanes)
            elif isinstance(m, nn.Linear):
                cx = linear_cx(cx, m.in_features, m.out_features, bias=(m.bias is not None))

        # The complexity of the heads.
        for m in self.head.children():
            if isinstance(m, nn.Linear):
                cx = linear_cx(cx, m.in_features, m.out_features, bias=(m.bias is not None))

        return cx


def _cropnet_sep(
    feature: str,
    head: str,
    head_config: List[int],
    weights: str,
    **kwargs: Any,
) -> CropNetSep:
    err_message = f"Feature extact network {feature} not supported."
    assert feature in _networks.keys(), err_message
    err_message = f"Head type {head} not supported"
    assert head in ['box', 'tune'], err_message

    feature_model = _networks[feature](**kwargs)
    feature = list(feature_model.children())[:-1]
    feature_head = list(feature_model.children())[-1]
    if isinstance(feature_head, nn.Linear):
        inplanes = feature_head.in_features
    else:
        head_wo_fc = list(feature_head.children())[:-1]
        fc = list(feature_head.children())[-1]
        inplanes = fc.in_features
        feature.extend(head_wo_fc)
    feature = nn.Sequential(*feature)

    if head == 'box':
        head = BoxHead(inplanes, config=head_config)
    else:
        head = TuneHead(inplanes, config=head_config)

    cropnet_sep = CropNetSep(feature, head)

    if weights:
        ckpt = torch.load(weights, map_location='cpu')
        cropnet_sep.load_state_dict(ckpt['model_state'])

    return cropnet_sep


def _cropnet_joint(
    feature: str,
    box_head_config: List[int],
    tune_head_config: List[int],
    weights: str,
    **kwargs,
) -> CropNetJoint:
    err_message = f"Feature extract network {feature} not supported."
    assert feature in _networks.keys(), err_message

    feature_model = _networks[feature](**kwargs)
    inplanes = feature_model.inplanes
    feature = list(feature_model.children())[:-1]
    feature = nn.Sequential(*feature)

    head_a = BoxHead(inplanes, config=box_head_config)
    head_b = BoxHead(inplanes, config=box_head_config)
    cropnet_joint = CropNetJoint(feature, head_a, head_b)

    if weights:
        ckpt = torch.load(weights)
        cropnet_joint.load_state_dict(ckpt['model_state'])

    return cropnet_joint


def cropnet_anynet_light_box(**kwargs: Any) -> CropNetSep:
    """Construct a box predictor network with a light AnyNet as the feature extractor.
    """
    box_head_config = [256, 64]
    weights = cfg.MODEL.BOX_WEIGHTS
    return _cropnet_sep(
        'anynet_light',
        'box',
        box_head_config,
        weights,
        **kwargs
    )


def cropnet_anynet_light_tune(**kwargs: Any) -> CropNetSep:
    """Construct a box predictor network with a light AnyNet as the feature extractor.
    """
    box_head_config = []
    weights = cfg.MODEL.TUNE_WEIGHTS
    return _cropnet_sep(
        'anynet_light',
        'tune',
        box_head_config,
        weights,
        **kwargs
    )


def cropnet_resnet18_tune(**kwargs: Any) -> CropNetSep:
    """Construct a box predictor network with a light AnyNet as the feature extractor.
    """
    box_head_config = [128]
    weights = cfg.MODEL.TUNE_WEIGHTS
    return _cropnet_sep(
        'resnet18',
        'tune',
        box_head_config,
        weights,
        **kwargs
    )


def cropnet_resnet18_box(**kwargs: Any) -> CropNetSep:
    """Construct a box predictor network with a ResNet18 as the feature extractor.
    """
    box_head_config = [256]
    weights = cfg.MODEL.BOX_WEIGHTS
    return _cropnet_sep(
        'resnet18',
        'box',
        box_head_config,
        weights,
        **kwargs
    )


def cropnet_resnet34_box(**kwargs: Any) -> CropNetSep:
    """Construct a box predictor network with a ResNet34 as the feature extractor.
    """
    box_head_config = [256]
    weights = cfg.MODEL.BOX_WEIGHTS
    return _cropnet_sep(
        'resnet34',
        'box',
        box_head_config,
        weights,
        **kwargs
    )


def cropnet_regnetx_800mf_box(**kwargs: Any) -> CropNetSep:
    """Construct a box predictor network with a RegNet-X_800mf as the feature extractor.
    """
    box_head_config = [256]
    weights = cfg.MODEL.BOX_WEIGHTS
    return _cropnet_sep(
        'regnet_x_800mf',
        'box',
        box_head_config,
        weights,
        **kwargs
    )


def cropnet_regnetx_1_6gf_box(**kwargs: Any) -> CropNetSep:
    """Construct a box predictor network with a RegNet-X_1_6gf as the feature extractor.
    """
    box_head_config = [256]
    weights = cfg.MODEL.BOX_WEIGHTS
    return _cropnet_sep(
        'regnet_x_1_6gf',
        'box',
        box_head_config,
        weights,
        **kwargs
    )


def cropnet_effnet(**kwargs: Any) -> CropNetSep:
    """Construct a box predictor network with a EfficientNet as the feature extractor.
    """
    box_head_config = [256]
    weights = cfg.MODEL.BOX_WEIGHTS
    return _cropnet_sep(
        'effnet',
        'box',
        box_head_config,
        weights,
        **kwargs
    )


def cropnet_anynet_light_joint(**kwargs: Any) -> CropNetJoint:
    """Construct multi-branch predictor, which has a box prediction branch and a box tuning branch. The
    feature extract network is anynet_light.
    """
    box_head_config = [256]
    tune_head_config = [256, 128]
    weights = cfg.MODEL.BOX_WEIGHTS
    return _cropnet_joint(
        'anynet_light',
        box_head_config,
        tune_head_config,
        weights,
        **kwargs
    )


def cropnet_anynet_light_dual_box(**kwargs: Any) -> CropNetJoint:
    """Construct the box predictor which generates two boxes simultaneously."""
    box_config = [256]
    weights = cfg.MODEL.BOX_WEIGHTS
    return _cropnet_joint(
        'anynet_light',
        box_config,
        box_config,
        weights,
        **kwargs,
    )


if __name__ == '__main__':
    model = cropnet_effnet(width_mult=1.0)
    inputs = torch.randn(1, 3, 224, 224)
    # model, inputs = model.to('cuda'), inputs.to('cuda')

    # out = model(inputs)

    print(model)

