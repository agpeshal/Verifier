import logging
import torch
import torch.nn as nn

from layers import modLayer, ReLU

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

INPUT_SIZE = 28
NUM_CLASSES = 10


class Model(nn.Module):
    def __init__(self, model: nn.Module, eps: float, x: torch.Tensor, true_label: int):
        super().__init__()
        layers = [modLayer(layer) for layer in model.layers]
        self.net = nn.Sequential(*layers)

        self.true_label = true_label
        self.x_max, self.x_min = torch.clamp(x.data + eps, max=1), torch.clamp(x.data - eps, min=0)


    def forward(self):
        return self.net([self.x_min, self.x_max])


    def verify(self) -> bool:
        bound_l, bound_u = self.forward()

        minima = bound_l[:, self.true_label]
        maxima = bound_u[:, self.true_label]

        #print(self.true_label)
        #print(minima, maxima)

        ctr = 0
        for label in range(NUM_CLASSES):
            if label != self.true_label:
                minm = bound_l[:, label]
                maxm = bound_u[:, label]

                if ((maxima < minm) or (minima > maxm)):
                    ctr += 1

        if ctr == NUM_CLASSES-1:
            print('verified')
        else:
            print('not verified')

        exit()

