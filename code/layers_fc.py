import copy
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

torch.autograd.set_detect_anomaly(True)


class Normalization(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        start_time = time.time()
        lx_in, ux_in, lc_in, uc_in = input

        lx_out = lx_in * 1/0.3081
        ux_out = ux_in * 1/0.3081
        lc_out = lc_in - 0.1307
        uc_out = uc_in - 0.1307

        if is_verbose: print('Normalization: time=', round(time.time() - start_time, 4))
        return [lx_out, ux_out, lc_out, uc_out]


class Flatten(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.start_dim = layer.start_dim
        self.end_dim = layer.end_dim

    def forward(self, input):
        lx_in, ux_in, lc_in, uc_in = input

        lx_out = lx_in.flatten(self.start_dim, self.end_dim).squeeze()
        ux_out = ux_in.flatten(self.start_dim, self.end_dim).squeeze()

        lc_out = lc_in.flatten(self.start_dim, self.end_dim)
        uc_out = uc_in.flatten(self.start_dim, self.end_dim)

        if is_verbose: print('Flatten: time=', round(time.time() - start_time, 4))
        return [torch.diag(lx_out), torch.diag(ux_out), lc_out, uc_out]


class Linear(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.weight = layer.weight.data.T
        self.bias = layer.bias.data

    def forward(self, input):
        lx_in, ux_in, lc_in, uc_in = input

        # prepare signed mask from weight
        mask = torch.sign(self.weight)
        mask_pos = torch.zeros_like(mask)
        mask_neg = torch.zeros_like(mask)
        mask_pos[mask > 0] = 1.0
        mask_neg[mask < 0] = 1.0

        mask_pos = mask_pos * self.weight
        mask_neg = mask_neg * self.weight

        # compute lx_out, ux_out
        lx_out = torch.mm(lx_in, mask_pos) + torch.mm(ux_in, mask_neg)
        ux_out = torch.mm(ux_in, mask_pos) + torch.mm(lx_in, mask_neg)

        # compute lc_out, uc_out
        bias = self.bias.unsqueeze(0)

        lc_out = torch.mm(lc_in, mask_pos) + torch.mm(uc_in, mask_neg) + bias
        uc_out = torch.mm(uc_in, mask_pos) + torch.mm(lc_in, mask_neg) + bias

        if is_verbose: print('Linear: time=', round(time.time() - start_time, 4))
        return [lx_out, ux_out, lc_out, uc_out]


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
        lx_in, ux_in, lc_in, uc_in = input

        minm = x_min.T.repeat(1, lx_in.shape[1])
        maxm = x_max.T.repeat(1, ux_in.shape[1])

        ##### compute l for each neuron by inserting bounds on input x: [x_min, x_max]
        # 1. prepare signed mask from lx_in
        mask = torch.sign(lx_in)
        mask_pos = torch.zeros_like(mask)
        mask_neg = torch.zeros_like(mask)
        mask_pos[mask > 0] = 1.0
        mask_neg[mask < 0] = 1.0

        # 2. multiply input bounds with signed mask
        mask_lower = minm * mask_pos + maxm * mask_neg

        # 3. computing l using accumulated weights and coefficients
        l = torch.sum(mask_lower * lx_in, dim=0) + lc_in.squeeze(0)

        ##### compute u for each neuron by inserting bounds on input x: [x_min, x_max]
        # 1. prepare signed mask from lx_in
        mask = torch.sign(ux_in)
        mask_pos = torch.zeros_like(mask)
        mask_neg = torch.zeros_like(mask)
        mask_pos[mask > 0] = 1.0
        mask_neg[mask < 0] = 1.0

        # 2. multiply input bounds with signed mask
        mask_upper = maxm * mask_pos + minm * mask_neg

        # 3. computing u using accumulated weights and coefficients
        u = torch.sum(mask_upper * ux_in , dim=0) + uc_in.squeeze(0)

        # define lx_out, ux_out, lc_out, uc_out, slope
        lx_out = torch.ones_like(lx_in) * float('-inf')
        ux_out = torch.ones_like(ux_in) * float('inf')
        lc_out = torch.zeros_like(lc_in)
        uc_out = torch.zeros_like(uc_in)
        slope = torch.ones_like(l)

        ##### evaluate ReLU conditions
        # Strictly negative
        idx = torch.where(u <= 0)[0]
        if len(idx) > 0:
            lx_out[:, idx] = 0.0
            ux_out[:, idx] = 0.0
            lc_out[:, idx] = 0.0
            uc_out[:, idx] = 0.0

        # Strictly positive
        idx = torch.where(l >= 0)[0]
        if len(idx) > 0:
            lx_out[:, idx] = lx_in[:, idx]
            ux_out[:, idx] = ux_in[:, idx]
            lc_out[:, idx] = lc_in[:, idx]
            uc_out[:, idx] = uc_in[:, idx]

        # Crossing ReLU
        idx = torch.where((l < 0) & (u > 0))[0]
        if len(idx) > 0:
            # lower bound
            lx_out[:, idx] = 0.0
            lc_out[:, idx] = 0.0

            # upper bound
            if not hasattr(self, "slope"):
                slope[idx] = u[idx] / (u[idx] - l[idx])
                self.slope = Variable(torch.clamp(slope, 0, 1), requires_grad=True)
                self.slope.retain_grad()

            slope = torch.clamp(self.slope, 0, 1)
            ux_out[:, idx] = slope[idx] * ux_in[:, idx]
            uc_out[:, idx] = slope[idx] * uc_in[:, idx] - slope[idx] * l[idx]

        if is_verbose: print('ReLU: time=', round(time.time() - start_time, 4))
        return [lx_out, ux_out, lc_out, uc_out]


def initialize_properties(input, trainable=False, verbose=False):
    global x_min
    global x_max
    global is_trainable
    global is_verbose
    global start_time

    x_min = input[0].flatten(1, -1)
    x_max = input[1].flatten(1, -1)
    is_trainable = trainable
    is_verbose = verbose
    start_time = time.time()


def modLayer(layer):
    layer_name = layer.__class__.__name__
    modified_layers = {"Normalization": Normalization, "Flatten": Flatten, "Linear": Linear, "ReLU": ReLU}

    if layer_name not in modified_layers:
        return copy.deepcopy(layer)

    return modified_layers[layer_name](layer)