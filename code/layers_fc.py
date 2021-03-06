import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)


class Normalization(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, input):
        l_in, u_in, lx_in, ux_in, lc_in, uc_in = input

        l_out = l_in[:]
        u_out = u_in[:]
        lx_out = lx_in[:]
        ux_out = ux_in[:]
        lc_out = lc_in[:]
        uc_out = uc_in[:]

        l_out[-1] = (l_in[-1] - 0.1307) / 0.3081
        u_out[-1] = (u_in[-1] - 0.1307) / 0.3081

        return [l_out, u_out, lx_out, ux_out, lc_out, uc_out]


class Flatten(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.start_dim = layer.start_dim
        self.end_dim = layer.end_dim

    def forward(self, input):
        l_in, u_in, lx_in, ux_in, lc_in, uc_in = input

        l_out = l_in[:]
        u_out = u_in[:]
        lx_out = lx_in[:]
        ux_out = ux_in[:]
        lc_out = lc_in[:]
        uc_out = uc_in[:]

        l_out[-1] = l_in[-1].flatten()
        u_out[-1] = u_in[-1].flatten()

        return [l_out, u_out, lx_out, ux_out, lc_out, uc_out]


class Linear(nn.Module):
    '''
    Abstract transformer for Fully connected layer
    '''

    def __init__(self, layer):
        super().__init__()
        self.weight = layer.weight.data.T
        self.bias = layer.bias.data

    def forward(self, input):
        l_in, u_in, lx_in, ux_in, lc_in, uc_in = input

        l_out = l_in[:]
        u_out = u_in[:]
        lx_out = lx_in[:]
        ux_out = ux_in[:]
        lc_out = lc_in[:]
        uc_out = uc_in[:]

        # lx_out, ux_out
        lx_out.append(self.weight)
        ux_out.append(self.weight)

        # lc_out, uc_out
        lc_out.append(self.bias)
        uc_out.append(self.bias)

        # l_out, u_out
        # backsubstitute to find concrete lower and upper bounds
        l_out_, u_out_ = backsubstitution(
            input=[l_out, u_out, lx_out, ux_out, lc_out, uc_out])
        l_out.append(l_out_)
        u_out.append(u_out_)

        return [l_out, u_out, lx_out, ux_out, lc_out, uc_out]


class ReLU(nn.Module):
    '''
    Abstract transformer for ReLU layer
    '''

    def __init__(self, *args):
        super().__init__()

    def forward(self, input):
        l_in, u_in, lx_in, ux_in, lc_in, uc_in = input

        l_out = l_in[:]
        u_out = u_in[:]
        lx_out = lx_in[:]
        ux_out = ux_in[:]
        lc_out = lc_in[:]
        uc_out = uc_in[:]

        n = len(l_out[-1])      # number of neurons

        lx_out_ = torch.ones(n)
        ux_out_ = torch.ones(n)
        lc_out_ = torch.zeros(n)
        uc_out_ = torch.zeros(n)

        # evaluate ReLU conditions
        l_in_ = l_in[-1]
        u_in_ = u_in[-1]

        # Strictly negative
        idx = torch.where(u_in_ <= 0)[0]
        if len(idx) > 0:
            lx_out_[idx] = 0.0
            ux_out_[idx] = 0.0
            lc_out_[idx] = 0.0
            uc_out_[idx] = 0.0

        # Strictly positive
        idx = torch.where(l_in_ >= 0)[0]
        if len(idx) > 0:
            lx_out_[idx] = 1.0
            ux_out_[idx] = 1.0
            lc_out_[idx] = 0.0
            uc_out_[idx] = 0.0

        # Crossing ReLU
        idx = torch.where((l_in_ < 0) & (u_in_ > 0))[0]
        if len(idx) > 0:
            # lower bound
            lc_out_[idx] = 0.0

            # upper bound
            if not hasattr(self, 'slope_upper'):
                slope_upper = torch.ones(n)
                slope_upper[idx] = u_in_[idx] / (u_in_[idx] - l_in_[idx])
                self.slope_upper = Variable(torch.clamp(
                    slope_upper, 0, 1), requires_grad=True)

                # DO NOT initialize with zeros (no gradient in that case)
                slope_lower = torch.ones(n)
                self.slope_lower = Variable(torch.clamp(
                    slope_lower, 0, 1), requires_grad=True)

            self.slope_upper.data.clamp_(min=0, max=1)
            ux_out_[idx] = self.slope_upper[idx]
            # Compute Hinge and intercept
            threshold = u_in_[idx] / (u_in_[idx] - l_in_[idx])
            mask_lower = self.slope_upper[idx] >= threshold
            mask_upper = self.slope_upper[idx] < threshold

            uc_out_[idx] = ((1 - self.slope_upper[idx]) * u_in_[idx]) * \
                mask_upper + (- self.slope_upper[idx] * l_in_[idx]) * mask_lower

            self.slope_lower.data.clamp_(min=0, max=1)
            lx_out_[idx] = self.slope_lower[idx]

        # lx_out, ux_out
        lx_out.append(torch.diag(lx_out_))
        ux_out.append(torch.diag(ux_out_))

        # lc_out, uc_out
        lc_out.append(lc_out_)
        uc_out.append(uc_out_)

        # Compute concrete lower and upper bound with help of values
        # from previous layer and current coefficients and intercepts
        l_out_ = torch.max(torch.zeros_like(l_in_), l_in_)
        u_out_ = u_in_ * ux_out_ + uc_out_
        l_out.append(l_out_)
        u_out.append(u_out_)

        return [l_out, u_out, lx_out, ux_out, lc_out, uc_out]


class Objective(nn.Module):
    '''
    Abstract transformer for the final verification.
    Subtracting the logits of all incorrect logits
    from the correct class logit
    '''

    def __init__(self, num_classes, true_label):
        super().__init__()
        self.num_classes = num_classes
        self.true_label = true_label

    def forward(self, input):
        l_in, u_in, lx_in, ux_in, lc_in, uc_in = input

        l_out = l_in[:]
        u_out = u_in[:]
        lx_out = lx_in[:]
        ux_out = ux_in[:]
        lc_out = lc_in[:]
        uc_out = uc_in[:]

        # lx_out, ux_out
        lx_out_ = torch.zeros(size=(self.num_classes - 1, self.num_classes))
        for i in range(self.num_classes - 1):
            lx_out_[i, self.true_label] = 1.0

            if i < self.true_label:
                lx_out_[i, i] = -1.0
            else:
                lx_out_[i, i+1] = -1.0

        lx_out.append(lx_out_.T)
        ux_out.append(lx_out_.T)  # lx_out_ and ux_out_ are the same

        # lc_out, uc_out
        lc_out_ = torch.zeros(self.num_classes - 1)
        lc_out.append(lc_out_)
        uc_out.append(lc_out_)  # lx_out_ and ux_out_ are the same

        # l_out, u_out
        # backsubstitute to find concrete lower and upper bounds
        l_out_, u_out_ = backsubstitution(
            input=[l_out, u_out, lx_out, ux_out, lc_out, uc_out])
        l_out.append(l_out_)
        u_out.append(u_out_)

        return [l_out, u_out, lx_out, ux_out, lc_out, uc_out]


def backsubstitution(input):
    '''
    Backsubstitute till the input layer
    '''

    l_in, u_in, lx_in, ux_in, lc_in, uc_in = input

    n = len(lx_in)
    n = (np.arange(n) + 1) * -1

    lx_out_ = lx_in[-1]
    ux_out_ = ux_in[-1]
    lc_out_ = lc_in[-1]
    uc_out_ = uc_in[-1]

    for i in n:
        if i != -1:
            lx_ = lx_in[i]
            ux_ = ux_in[i]
            lc_ = lc_in[i]
            uc_ = uc_in[i]

            # backsubstitute lx, lc
            mask = torch.sign(lx_out_)
            mask_pos = torch.zeros_like(mask)
            mask_neg = torch.zeros_like(mask)
            mask_pos[mask > 0] = 1
            mask_neg[mask < 0] = 1
            mask_pos = mask_pos * lx_out_
            mask_neg = mask_neg * lx_out_

            lx_out_ = torch.mm(lx_, mask_pos) + torch.mm(ux_, mask_neg)
            lc_out_ = torch.mm(lc_.unsqueeze(0), mask_pos).squeeze() + \
                torch.mm(uc_.unsqueeze(0), mask_neg).squeeze() + lc_out_

            # backsubstitute ux, uc
            mask = torch.sign(ux_out_)
            mask_pos = torch.zeros_like(mask)
            mask_neg = torch.zeros_like(mask)
            mask_pos[mask > 0] = 1
            mask_neg[mask < 0] = 1
            mask_pos = mask_pos * ux_out_
            mask_neg = mask_neg * ux_out_

            ux_out_ = torch.mm(ux_, mask_pos) + torch.mm(lx_, mask_neg)
            uc_out_ = torch.mm(uc_.unsqueeze(0), mask_pos).squeeze() + \
                torch.mm(lc_.unsqueeze(0), mask_neg).squeeze() + uc_out_

    # Insert l, u to compute l_out, u_out
    l_ = l_in[i].unsqueeze(0).T.repeat(1, lx_out_.shape[1])
    u_ = u_in[i].unsqueeze(0).T.repeat(1, lx_out_.shape[1])

    # compute l_out_
    mask = torch.sign(lx_out_)
    mask_pos = torch.zeros_like(mask)
    mask_neg = torch.zeros_like(mask)
    mask_pos[mask > 0] = 1
    mask_neg[mask < 0] = 1
    mask_pos = mask_pos * lx_out_
    mask_neg = mask_neg * lx_out_
    l_out_ = torch.sum(mask_pos * l_ + mask_neg * u_, dim=0) + lc_out_

    # compute u_out_
    mask = torch.sign(ux_out_)
    mask_pos = torch.zeros_like(mask)
    mask_neg = torch.zeros_like(mask)
    mask_pos[mask > 0] = 1
    mask_neg[mask < 0] = 1
    mask_pos = mask_pos * ux_out_
    mask_neg = mask_neg * ux_out_
    u_out_ = torch.sum(mask_pos * u_ + mask_neg * l_, dim=0) + uc_out_

    return l_out_, u_out_


def modLayer(layer):
    layer_name = layer.__class__.__name__

    modified_layers = {'Normalization': Normalization,
                       'Flatten': Flatten,
                       'Linear': Linear,
                       'ReLU': ReLU}

    if layer_name not in modified_layers:
        return copy.deepcopy(layer)

    return modified_layers[layer_name](layer)

