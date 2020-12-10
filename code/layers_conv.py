import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Conv2D(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.weight = layer.weight.data
        self.bias = layer.bias.data
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.kernel_size = layer.kernel_size
        self.in_channels = self.weight.shape[1]
        self.out_channels = self.weight.shape[0]
        self.kernal_size_number = self.kernel_size[0] * self.kernel_size[1]

    def get_height_out(self, h_in):
        return ((h_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)// self.stride[1]) + 1

    def get_width_out(self, w_in):
        return ((w_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)// self.stride[0]) + 1

    def get_windows(self, h_in, w_in, h_out, w_out):
        x = torch.arange(0, h_in * w_in).float() + 1
        x = x.reshape((1, 1, h_in, w_in))

        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride)
        windows = windows.transpose(1, 2).contiguous().view(-1, x.shape[1], self.kernal_size_number)
        windows = windows.transpose(0, 1)
        windows = windows.squeeze()

        mask_pos = torch.zeros_like(windows)
        mask_neg = torch.zeros_like(windows)
        mask_pos[windows > 0] = 1.0
        mask_neg[windows <= 0] = 1.0
        mask_pos = mask_pos * (windows - 1)
        mask_neg = mask_neg * (windows - 1)

        windows_h = (mask_pos) // h_in + mask_neg
        windows_w = (mask_pos) % h_in + mask_neg

        windows_h = windows_h.reshape((h_out, w_out, self.kernal_size_number))
        windows_w = windows_w.reshape((h_out, w_out, self.kernal_size_number))
        return windows_h.int(), windows_w.int()


    def get_signed_weight_mask(self):
        mask = torch.sign(self.weight)
        mask_pos = torch.zeros_like(mask)
        mask_neg = torch.zeros_like(mask)
        mask_pos[mask > 0] = 1.0
        mask_neg[mask < 0] = 1.0
        return mask_pos * self.weight, mask_neg * self.weight


    def forward(self, input):
        lx_in, ux_in, lc_in, uc_in = input
        h_in = lx_in.shape[1]
        w_in = lx_in.shape[2]

        # get new spatial dimensions
        h_out = self.get_height_out(h_in)
        w_out = self.get_width_out(w_in)

        # get convolution windows
        windows_h, windows_w = self.get_windows(h_in, w_in, h_out, w_out)  # (h_out * w_out) x (3 * 3)

        # get weight masks
        mask_pos, mask_neg = self.get_signed_weight_mask()      # dim = self.weight

        # define lx_out, ux_out, lc_out, uc_out
        lx_out = torch.zeros(size=(self.out_channels, h_out, w_out, input_size, input_size))
        ux_out = torch.zeros(size=(self.out_channels, h_out, w_out, input_size, input_size))
        lc_out = torch.zeros(size=(self.out_channels, h_out, w_out))
        uc_out = torch.zeros(size=(self.out_channels, h_out, w_out))

        mask_pos_ = mask_pos.reshape(self.out_channels, self.in_channels, self.kernal_size_number)  # 16 x 1 x 9
        mask_neg_ = mask_neg.reshape(self.out_channels, self.in_channels, self.kernal_size_number)  # 16 x 1 x 9

        # compute lx_out, ux_out, lc_out, uc_out
        for h in range(h_out):
            for w in range(w_out):
                windows_h_ = windows_h[h, w, :]
                windows_w_ = windows_w[h, w, :]
                windows_mask = (windows_h_ > -1) & (windows_w_ > -1)

                for idx, (i, j) in enumerate(zip(windows_h_, windows_w_)):
                    if windows_mask[idx]:
                        # compute lx_out
                        lx_out[:, h, w, :, :] += \
                            torch.einsum('ab,bcd->acd', mask_pos_[:, :, idx], lx_in[:, i, j, :, :]) + \
                            torch.einsum('ab,bcd->acd', mask_neg_[:, :, idx], ux_in[:, i, j, :, :])  # [16x1] x [1x28x28] -> [16,28,28]

                        # compute ux_out
                        ux_out[:, h, w, :, :] += \
                            torch.einsum('ab,bcd->acd', mask_pos_[:, :, idx], ux_in[:, i, j, :, :]) + \
                            torch.einsum('ab,bcd->acd', mask_neg_[:, :, idx], lx_in[:, i, j, :, :])  # [16x1] x [1x28x28] -> [16,28,28]

                        # compute lc_out
                        lc_out[:, h, w] += torch.einsum('ab,b->a', mask_pos_[:, :, idx], lc_in[:, i, j]) + \
                                           torch.einsum('ab,b->a', mask_neg_[:, :, idx], uc_in[:, i, j])  # [16x1] x [1] -> [16]

                        # compute uc_out
                        uc_out[:, h, w] += torch.einsum('ab,b->a', mask_pos_[:, :, idx], uc_in[:, i, j]) + \
                                           torch.einsum('ab,b->a', mask_neg_[:, :, idx], lc_in[:, i, j])  # [16x1] x [1] -> [16]

        # add bias
        for outChannel in range(self.out_channels):
            lc_out[outChannel] += self.bias[outChannel]
            uc_out[outChannel] += self.bias[outChannel]

        if is_verbose: print('Conv2D: time=', round(time.time() - start_time, 4))
        return [lx_out, ux_out, lc_out, uc_out]


class ReLU_Conv(nn.Module):
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

        in_channels = lx_in.shape[0]
        height =  lx_in.shape[1]
        width =  lx_in.shape[2]

        minm = x_min.repeat(in_channels, height, width, 1, 1)
        maxm = x_max.repeat(in_channels, height, width, 1, 1)

        ##### compute l for each neuron by inserting bounds on input x: [x_min, x_max]
        # 1. prepare signed mask from lx
        mask = torch.sign(lx_in)
        mask_pos = torch.zeros_like(mask)
        mask_neg = torch.zeros_like(mask)
        mask_pos[mask > 0] = 1.0
        mask_neg[mask < 0] = 1.0

        # 2. multiply input bounds with signed mask
        mask_lower = minm * mask_pos + maxm * mask_neg

        # 3. computing l using accumulated weights and coefficients
        l = torch.sum(mask_lower * lx_in, dim=(-2, -1)) + lc_in

        ##### compute u for each neuron by inserting bounds on input x: [x_min, x_max]
        # 1. prepare signed mask from ux
        mask = torch.sign(ux_in)
        mask_pos = torch.zeros_like(mask)
        mask_neg = torch.zeros_like(mask)
        mask_pos[mask > 0] = 1.0
        mask_neg[mask < 0] = 1.0

        # 2. multiply input bounds with signed mask
        mask_upper = maxm * mask_pos + minm * mask_neg

        # 3. computing l using accumulated weights and coefficients
        u = torch.sum(mask_upper * ux_in, dim=(-2, -1)) + uc_in

        # define lx_out, ux_out, lc_out, uc_out, slope
        lx_out = torch.zeros_like(lx_in)
        ux_out = torch.zeros_like(ux_in)
        lc_out = torch.zeros_like(lc_in)
        uc_out = torch.zeros_like(uc_in)
        slope = torch.ones_like(l)

        ##### evaluate ReLU conditions
        # Strictly negative
        idx = torch.where(u <= 0)
        if len(idx[0] > 0):
            lx_out[idx] = 0.0
            ux_out[idx] = 0.0
            lc_out[idx] = 0.0
            uc_out[idx] = 0.0

        # Strictly positive
        idx = torch.where(l >= 0)
        if len(idx[0] > 0):
            lx_out[idx] = lx_in[idx]
            ux_out[idx] = ux_in[idx]
            lc_out[idx] = lc_in[idx]
            uc_out[idx] = uc_in[idx]

        # Crossing ReLU
        idx = torch.where((l < 0) & (u > 0))
        if len(idx[0] > 0):
            # lower bound
            lx_out[idx] = 0.0
            ux_out[idx] = 0.0

            # upper bound
            if not hasattr(self, "slope"):
                slope[idx] = u[idx] / (u[idx] - l[idx])
                self.slope = Variable(slope, requires_grad=True)
                self.slope.retain_grad()

            slope = torch.clamp(self.slope, 0, 1)
            ux_out[idx] = torch.einsum('a,abc->abc', slope[idx], ux_in[idx])
            uc_out[idx] = slope[idx] * uc_in[idx] - slope[idx] * l[idx]

        if is_verbose: print('ReLU Conv: time=', round(time.time() - start_time, 4))
        return [lx_out, ux_out, lc_out, uc_out]


class ReLU_Linear(nn.Module):
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

        x_min_ = x_min.flatten(1, -1)
        x_max_ = x_max.flatten(1, -1)

        minm = x_min_.T.repeat(1, lx_in.shape[1])
        maxm = x_max_.T.repeat(1, ux_in.shape[1])

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
        u = torch.sum(mask_upper * ux_in, dim=0) + uc_in.squeeze(0)

        ##### evaluate ReLU conditions
        lx_out = torch.ones_like(lx_in) * float('-inf')
        ux_out = torch.ones_like(ux_in) * float('inf')
        lc_out = torch.zeros_like(lc_in)
        uc_out = torch.zeros_like(uc_in)
        slope = torch.ones_like(l)

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

        if is_verbose: print('ReLU Linear: time=', round(time.time() - start_time, 4))
        return [lx_out, ux_out, lc_out, uc_out]


class Flatten(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.start_dim = layer.start_dim
        self.end_dim = layer.end_dim

    def forward(self, input):
        lx_in, ux_in, lc_in, uc_in = input

        # define lx_out, ux_out, lc_out, uc_out
        lx_in = torch.flatten(lx_in, start_dim=3, end_dim=-1)
        lx_out = torch.flatten(lx_in, start_dim=0, end_dim=2).T

        ux_in = torch.flatten(ux_in, start_dim=3, end_dim=-1)
        ux_out = torch.flatten(ux_in, start_dim=0, end_dim=2).T

        lc_out = lc_in.flatten().unsqueeze(0)
        uc_out = uc_in.flatten().unsqueeze(0)

        if is_verbose: print('Flatten: time=', round(time.time() - start_time, 4))
        return [lx_out, ux_out, lc_out, uc_out]


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


def initialize_properties(input, trainable=False, verbose=False):
    global x_min
    global x_max
    global is_trainable
    global is_verbose
    global input_size
    global start_time

    x_min = input[0].squeeze(0)
    x_max = input[1].squeeze(0)
    is_trainable = trainable
    is_verbose = verbose
    input_size = input[0].shape[-1]
    start_time = time.time()


def modLayer(layer_prev, layer_cur):
    if layer_cur.__class__.__name__ == 'ReLU' and layer_prev.__class__.__name__ == 'Linear':
        layer_name = 'ReLU_Linear'
    elif layer_cur.__class__.__name__ == 'ReLU' and layer_prev.__class__.__name__ == 'Conv2d':
        layer_name = 'ReLU_Conv'
    else:
        layer_name = layer_cur.__class__.__name__

    modified_layers = {'Normalization': Normalization,
                       'Flatten': Flatten,
                       'Linear': Linear,
                       'ReLU_Linear': ReLU_Linear,
                       'ReLU_Conv': ReLU_Conv,
                       'Conv2d': Conv2D}

    if layer_name not in modified_layers:
        return copy.deepcopy(layer_cur)

    return modified_layers[layer_name](layer_cur)