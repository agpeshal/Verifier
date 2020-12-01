import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.autograd.set_detect_anomaly(True)


class Normalization(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        l, u = input
        return [(l - 0.1307)/0.3081, (u - 0.1307)/0.3081]


class Flatten(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.start_dim = layer.start_dim
        self.end_dim = layer.end_dim

    def forward(self, input):
        l, u = input
        return [l.flatten(self.start_dim, self.end_dim), u.flatten(self.start_dim, self.end_dim)]


class Linear(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.weight = layer.weight.data.T
        self.bias = layer.bias.data

    def forward(self, input):
        l, u = input

        # prepare mask from weight coefficients
        mask = torch.sign(self.weight)
        mask = (1 + mask)/2

        l = l.T.repeat(1, mask.shape[1])
        u = u.T.repeat(1, mask.shape[1])

        # compute lower bound
        mask_l = l * mask + u * (1 - mask)
        bound_l = torch.sum(mask_l * self.weight, dim=0) + self.bias
        bound_l = torch.unsqueeze(bound_l, 0)

        # compute upper bound
        mask_u = u * mask + l * (1 - mask)
        bound_u = torch.sum(mask_u * self.weight, dim=0) + self.bias
        bound_u = torch.unsqueeze(bound_u, 0)

        return [bound_l, bound_u]


class Conv(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.weight = layer.weight.data
        self.bias = layer.bias.data
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.kernel_size = layer.kernel_size
        self.out_channels = self.weight.shape[0]
        self.in_channels = self.weight.shape[1]

    def forward(self, input):
        l, u = input

        # get dimensions
        l_ = F.conv2d(l, self.weight, None, self.stride, self.padding)
        width = l_.shape[-2]
        height = l_.shape[-1]

        # Padding
        l = F.pad(l, pad=(1, 1, 1, 1), mode='constant', value=0)
        u = F.pad(u, pad=(1, 1, 1, 1), mode='constant', value=0)

        bound_l = torch.zeros([self.out_channels, height, width])
        bound_u = torch.zeros([self.out_channels, height, width])
        weight_mask = torch.sign(self.weight)
        weight_mask = (1 + weight_mask)/2

        for outChannel in range(self.out_channels):
            for h in range(height):
                for w in range(width):
                    for inChannel in range(self.in_channels):
                        weight = self.weight[outChannel, inChannel, :, :]
                        mask = weight_mask[outChannel, inChannel, :, :]

                        value_l = l[:, inChannel, self.stride[0]*h : self.stride[0]*(h+1)+1 , self.stride[1]*w: self.stride[1]*(w+1)+1]
                        value_u = u[:, inChannel, self.stride[0]*h : self.stride[0]*(h+1)+1 , self.stride[1]*w: self.stride[1]*(w+1)+1]

                        mask_l = value_l * mask + value_u * (1 - mask)
                        bound_l[outChannel, h, w] += torch.sum(mask_l * weight)

                        mask_u = value_u * mask + value_l * (1 - mask)
                        bound_u[outChannel, h, w] += torch.sum(mask_u * weight)

            bound_l[outChannel, :, :] += self.bias[outChannel]
            bound_u[outChannel, :, :] += self.bias[outChannel]

        bound_l = bound_l.unsqueeze(0)
        bound_u = bound_u.unsqueeze(0)

        return [bound_l, bound_u]


class ReLU(nn.Module):
    """
    Modified ReLU that tries to approximate the Zonotope region.
    Adds a new sample to batch sample, if new eps term is added to system.
    Idea:
        a) `slope` is an optional attribute. In case no crossing takes place, given ReLU is irrelevant & thus it's
           `slope` is not optimized.
        b) During the forward pass, if any crossing takes place, `slope` is intialized as `Variable`, such that
           gradients are calculated with respect to it. See `leastArea()` for implementation.
        c) `lower_bound` and `upper_bound` define the state of ReLU and hence used as attributes.
        d) `intercept` for ReLU is a function of `slope`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        l, u = input

        bound_l = torch.ones_like(l) * float('inf')
        bound_u = torch.ones_like(u) * float('inf')
        slope = torch.ones_like(l) * float('inf')

        # Strictly negative
        idx = torch.where(u <= 0)
        if len(idx[0]) > 0:
            bound_l[idx] = 0
            bound_u[idx] = 0

        # Strictly positive
        idx = torch.where(l >= 0)
        if len(idx[0]) > 0:
            bound_l[idx] = l[idx]
            bound_u[idx] = u[idx]

        # Crossing ReLU
        idx = torch.where((l < 0) & (u > 0))
        if len(idx[0]) > 0:
            # lower bound
            bound_l[idx] = 0

            # upper bound
            if not trainable:
                slope[idx] = u[idx]/ (u[idx] - l[idx])
            else:
                print('TODO')
                exit()
            bound_u[idx] = slope[idx] * u[idx] - slope[idx] * l[idx]

        return [bound_l, bound_u]


def modLayer(layer, trainable_=False):
    global trainable
    trainable = trainable_

    layer_name = layer.__class__.__name__
    modified_layers = {"Normalization": Normalization, "Flatten": Flatten, "Linear": Linear, "Conv2d": Conv, "ReLU": ReLU}

    print(layer_name)

    if layer_name not in modified_layers:
        return copy.deepcopy(layer)

    return modified_layers[layer_name](layer)