import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)

class Normalization(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        l_in, u_in, lx_in, ux_in, lc_in, uc_in = input

        l_out = l_in[:]
        u_out = u_in[:]
        lx_out = lx_in[:]
        ux_out = ux_in[:]
        lc_out = lc_in[:]
        uc_out = uc_in[:]

        l_out[-1] = (l_in[-1] - 0.1307)/0.3081
        u_out[-1] = (u_in[-1] - 0.1307)/0.3081

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

        l_out[-1] = l_in[-1].flatten(self.start_dim, self.end_dim).squeeze()
        u_out[-1] = u_in[-1].flatten(self.start_dim, self.end_dim).squeeze()

        return [l_out, u_out, lx_out, ux_out, lc_out, uc_out]


class Linear(nn.Module):
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

        ##### lx_out, ux_out
        lx_out.append(self.weight)
        ux_out.append(self.weight)

        ##### lc_out, uc_out
        lc_out.append(self.bias)
        uc_out.append(self.bias)

        ##### l_out, u_out
        l_out_, u_out_ = backsubstitution(input=[l_out, u_out, lx_out, ux_out, lc_out, uc_out])
        l_out.append(l_out_)
        u_out.append(u_out_)

        return [l_out, u_out, lx_out, ux_out, lc_out, uc_out]


class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
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
        slope = torch.ones(n)

        ##### evaluate ReLU conditions
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
            lx_out_[idx] = 0.0
            lc_out_[idx] = 0.0

            # upper bound
            if not hasattr(self, 'slope'):
                slope[idx] = u_in_[idx] / (u_in_[idx] - l_in_[idx])
                self.slope = Variable(torch.clamp(slope, 0, 1), requires_grad=True)
                self.slope.retain_grad()

            slope = torch.clamp(self.slope, 0, 1)
            ux_out_[idx] = slope[idx]
            uc_out_[idx] = - slope[idx] * l_in_[idx]


        ##### lx_out, ux_out
        lx_out.append(torch.diag(lx_out_))
        ux_out.append(torch.diag(ux_out_))

        ##### lc_out, uc_out
        lc_out.append(lc_out_)
        uc_out.append(uc_out_)

        ##### l_out, u_out
        l_out_, u_out_ = backsubstitution(input=[l_out, u_out, lx_out, ux_out, lc_out, uc_out])
        l_out.append(l_out_)
        u_out.append(u_out_)

        return [l_out, u_out, lx_out, ux_out, lc_out, uc_out]


def backsubstitution(input):
    l_in, u_in, lx_in, ux_in, lc_in, uc_in = input

    ##### TODO: Compute l_out_, u_out_

    '''
    # 1. prepare signed mask from weight
    mask = torch.sign(self.weight)
    mask_pos = torch.zeros_like(mask)
    mask_neg = torch.zeros_like(mask)
    mask_pos[mask > 0] = 1.0
    mask_neg[mask < 0] = 1.0
    mask_pos = mask_pos * self.weight
    mask_neg = mask_neg * self.weight

    # 2. compute l_out, u_out
    l_out_ = (torch.mm(l_in[-1].unsqueeze(0), mask_pos) + torch.mm(u_in[-1].unsqueeze(0),
                                                                   mask_neg)).squeeze() + self.bias
    u_out_ = (torch.mm(u_in[-1].unsqueeze(0), mask_pos) + torch.mm(l_in[-1].unsqueeze(0),
                                                                   mask_neg)).squeeze() + self.bias
    l_out.append(l_out_)
    u_out.append(u_out_)
    #'''

    return l_out_, u_out_


def modLayer(layer):
    layer_name = layer.__class__.__name__
    modified_layers = {'Normalization': Normalization, 'Flatten': Flatten, 'Linear': Linear, 'ReLU': ReLU}

    if layer_name not in modified_layers:
        return copy.deepcopy(layer)

    return modified_layers[layer_name](layer)