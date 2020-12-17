import logging
import torch
import torch.nn as nn
from layers import *

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

INPUT_SIZE = 28
NUM_CLASSES = 2


class Model(nn.Module):
    def __init__(self, model, eps, x, true_label, args):
        super().__init__()

        self.true_label = true_label
        if args.net != 'dummy':
            self.x_min = torch.clamp(x.data - eps, min=0)
            self.x_max = torch.clamp(x.data + eps, max=1)
        else:
            self.x_min = x.data - eps
            self.x_max = x.data + eps

        self.model = model

    def forward(self):
        return self.net([[self.x_min], [self.x_max], [], [], [], []])       # [[l], [u], [lx], [ux], [lc], [uc]]

    def parameters(self):
        for layer in self.net:
            if isinstance(layer, ReLU) and hasattr(layer, 'slope'):
                yield layer.slope
    
    def updateParams(self):
        # Calculates the gradient of `loss` wrt to ReLU slopes.
        loss = -self.loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def verify(self):
        iterations = 10

        # Initialize transformed network
        layers = [modLayer(layer) for layer in self.model.layers]
        layers.append(Verifier(num_classes=NUM_CLASSES, true_label=self.true_label))
        self.net = nn.Sequential(*layers)

        # Optimizer
        self.forward()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        for i in range(iterations):
            l, u, lx, ux, lc, uc = self.forward()

            # Penalize negative values (not verified)
            idx = torch.where(l[-1] < 0)[0]
            self.loss = torch.sum(l[-1][idx])

            if self.loss == 0:
                return True

            self.updateParams()

