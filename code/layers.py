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
        self.kernal_size_number = self.kernel_size[0] * self.kernel_size[1]
        self.out_channels = self.weight.shape[0]

    def forward(self, input):
        l, u = input

        # get dimensions
        l_ = F.conv2d(l, self.weight, None, self.stride, self.padding)
        width = l_.shape[-2]
        height = l_.shape[-1]





        result = torch.zeros([x.shape[0] * self.out_channels, width, height], dtype=torch.float32)

        for channel in range(x.shape[1]):
            for i_convNumber in range(self.out_channels):
                xx = torch.matmul(windows[channel], self.weight[i_convNumber][channel])
                xx = xx.view(-1, width, height)
                result[i_convNumber * xx.shape[0]: (i_convNumber + 1) * xx.shape[0]] += xx

        result = result.view(x.shape[0], self.out_channels, width, height)

        print(result.shape)
        exit()

        print(torch.sum(y - result))

        exit()


        # prepare mask from weight coefficients
        mask = torch.sign(self.weight)
        mask_one = (1 + mask) / 2
        mask_one_minus = (1 - mask) / 2

        print(mask_one.shape, l.shape)
        exit()

        l = l.T.repeat(1, mask.shape[1])
        u = u.T.repeat(1, mask.shape[1])

        # compute lower bound
        mask_l = l * mask_one + u * mask_one_minus
        bound_l = torch.sum(mask_l * self.weight, dim=0) + self.bias
        bound_l = torch.unsqueeze(bound_l, 0)

        # compute upper bound
        mask_r = u * mask_one + l * mask_one_minus
        bound_u = torch.sum(mask_r * self.weight, dim=0) + self.bias
        bound_u = torch.unsqueeze(bound_u, 0)

        exit()

        return y

    def calculateWindows(self, l, u):


        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride)
        print(windows.shape)
        exit()
        windows = windows.transpose(1, 2).contiguous().view(-1, x.shape[1], self.kernal_size_number)
        windows = windows.transpose(0, 1)
        return windows


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