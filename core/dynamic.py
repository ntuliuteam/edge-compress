import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import numpy as np
from copy import deepcopy
import core.net as net
from core.crop import BaseCrop


class DynamicInfer(nn.Module):
    def __init__(self, models, resolutions, threshold_high, threshold_low, alpha=1.0, count_complexity=False):
        super(DynamicInfer, self).__init__()
        self.models = models
        self.resolutions = resolutions
        self.alpha = alpha
        self.high = [threshold_high] * len(models)
        self.low = [threshold_low] * len(models)
        self.count_complexity = count_complexity
        self.flops = 0
        self.params = 0
        self.acts = 0
        self.num_imgs = 0
        self.num_imgs_per_model = [0] * len(models)
        self.output = None

        err_str = "The number of models and resolutions must be equal, but got {} and {}, " \
                  "respectively.".format(len(models), len(resolutions))
        assert len(models) == len(resolutions), err_str

    @staticmethod
    def filter_confidence(output, high, low=0, topk=1, use_diff=False):
        """Pick out those input images whose prediction confidence score is lower than the given threshold."""
        preds = torch.softmax(output, dim=-1)
        if use_diff:
            preds = torch.topk(preds, 2, dim=-1).values
            preds = preds[:, 0] - preds[:, 1]
        else:
            preds = torch.topk(preds, topk, dim=-1).values.sum(dim=-1)
        high = torch.tensor(high).cuda()
        low = torch.tensor(low).cuda()
        # Get the index of those input images with low prediction confidence
        selected = torch.logical_and(preds < high, preds > low)
        low_conf_ids = torch.nonzero(selected)

        return torch.squeeze(low_conf_ids, dim=-1)

    @staticmethod
    def resize_tensor(inputs, res):
        new_inputs = []
        for tensor in inputs:
            h, w = tensor.shape[-2], tensor.shape[-1]
            if h == w:
                tensor = TF.resize(tensor, size=res)
            else:
                tensor = BaseCrop.scale_and_center_crop_tensor(tensor, res, res)
            new_inputs.append(tensor)

        return torch.cat(new_inputs, dim=0)

    def forward(self, x):
        self.num_imgs += len(x)
        ids_list = list()
        inputs = deepcopy(x)
        for model_idx, (model, res) in enumerate(zip(self.models, self.resolutions)):
            inputs = self.resize_tensor(inputs, res)

            bs = inputs.shape[0]
            self.num_imgs_per_model[model_idx] += bs
            if self.count_complexity:
                # Collect model statistics
                h, w = inputs.shape[-2], inputs.shape[-1]
                cx = {"h": h, "w": w, "flops": 0, "params": 0, "acts": 0}
                cx = net.unwrap_model(model).complexity(cx)
                self.flops += bs * cx["flops"]
                self.params += bs * cx["params"]
                self.acts += bs * cx["acts"]

            if model_idx == 0:
                output = model(inputs)
                # The prediction output of the first model
                self.output = output
            else:
                # Only select the input images with low confidence for the next model
                output = model(inputs)
                # Update the final prediction with the output of the latest model
                self.output[ids] += self.alpha * output

            psuedo_ids = self.filter_confidence(output, self.high[model_idx], low=self.low[model_idx], topk=1, use_diff=True)
            if len(psuedo_ids) == 0:
                # All input images obtain a highly confident prediction
                break

            # Get the real index of the selected samples
            if model_idx == 0:
                ids = psuedo_ids
            else:
                ids = ids_list[-1][psuedo_ids]

            ids_list.append(ids)
            inputs = deepcopy([x[i] for i in ids])

        return self.output

