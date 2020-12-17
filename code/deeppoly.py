import logging
import torch
import torch.nn as nn
from layers import *

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

INPUT_SIZE = 28
NUM_CLASSES = 10


class Model(nn.Module):
    def __init__(self, model, eps, x, true_label, args):
        super().__init__()
        x = x.squeeze(0)

        self.true_label = true_label
        if args.net != 'dummy' and args.net != 'dummy_norm':
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
            # if isinstance(layer, ReLU) and hasattr(layer, 'slope'):
            if isinstance(layer, ReLU_Linear) and hasattr(layer, 'slope'):
                yield layer.slope
    
    def updateParams(self):
        # Calculates the gradient of `loss` wrt to ReLU slopes.
        loss = -self.loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def get_layers(self):
        layers = []
        for i in range(len(self.model.layers)):
            if i > 1:
                layers.append(modLayer(self.model.layers[i-1], self.model.layers[i]))
            else:
                layers.append(modLayer(-1, self.model.layers[i]))
        return layers

    def verify(self):
        iterations = 20

        # Initialize transformed network
        # layers = [modLayer(layer) for layer in self.model.layers]
        layers = self.get_layers()
        layers.append(Verifier(num_classes=NUM_CLASSES, true_label=self.true_label))
        self.net = nn.Sequential(*layers)

        # Optimizer
        self.forward()
        # If not crossing ReLU in Conv -> parameters empty!
        try:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        except:
            print("Nothing to optimize")
            return False

        for i in range(iterations):
            # print("Iteration: ", i)
            l, u, lx, ux, lc, uc = self.forward()

            # Penalize negative values (not verified)
            idx = torch.where(l[-1] < 0)[0]
            self.loss = torch.sum(l[-1][idx])
            #self.loss = torch.sum(l[-1])
            
            # if not torch.any(l[-1] < 0):
            #     return True
            if self.loss == 0:
                return True
            # print("Iteration: {}, loss: {}".format(i, -self.loss.item()))
            self.updateParams()

