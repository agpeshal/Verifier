import logging
import torch
import torch.nn as nn
from configuration import Config
from layers import *

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

INPUT_SIZE = 28
NUM_CLASSES = 10


class Model(nn.Module):
    def __init__(self, model, eps, x, true_label, args):
        super().__init__()

        self.true_label = true_label
        self.x_min = torch.clamp(x.data - eps, min=0)
        self.x_max = torch.clamp(x.data + eps, max=1)

        # Initialize verifier configuration
        self.config = Config(args=args, net=model)

        # Define transformed network
        ctr = 0
        layers = []
        for layer in model.layers:
            is_backsub = False if ctr < self.config.forward_layers else True
            layers.append(modLayer(layer, is_backsub))
            ctr += 1
        self.net = nn.Sequential(*layers)



    def forward(self):
        return self.net([[self.x_min], [self.x_max], [], [], [], []])       # [[l], [u], [lx], [ux], [lc], [uc]]

    def parameters(self):
        for layer in self.net:
            if isinstance(layer, ReLU) and hasattr(layer, 'slope'):
                yield layer.slope
    
    def updateParams(self):
        # Calculates the gradient of `loss` wrt to ReLU slopes.
        loss = -self.lb.sum()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()


    def verify(self):
        iterations = 10

        for backsub in range(self.config.backsub_layers):
            self.forward()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05)

            if backsub == 0:
                l, u, lx, ux, lc, uc = self.forward()
                correct_l = l[-1][self.true_label]

                ctr = 0
                for label in range(NUM_CLASSES):
                    if label != self.true_label:
                        wrong_u = u[-1][label]

                        if correct_l > wrong_u:
                            ctr += 1

                if ctr == NUM_CLASSES - 1:
                    return True
                else:
                    continue

            for iter in range(iterations):
                l, u, lx, ux, lc, uc = self.forward()

                for bs in range(backsub):
                    correct_lx = lx[-bs-1][:, self.true_label]
                    correct_lc = lx[-bs-1][self.true_label]

                    for label in range(NUM_CLASSES):
                        if label != self.true_label:
                            wrong_ux = ux[-bs - 1][:, label]
                            wrong_uc = ux[-bs - 1][label]

                            diff_x = correct_lx - wrong_ux
                            diff_c = correct_lc - wrong_uc




                # compute lower bound of the difference
                mask = torch.sign(diff_x)
                mask_pos = torch.zeros_like(mask)
                mask_neg = torch.zeros_like(mask)
                mask_pos[mask > 0] = 1
                mask_neg[mask < 0] = 1

                mask_lower = l_prev * mask_pos + u_prev * mask_neg
                l = torch.sum(mask_lower * diff_x) + diff_c
                self.lb[label] = l





        exit()




        correct_lx = lx[:, self.true_label]
        correct_lc = lc[:, self.true_label]
        self.lb = torch.zeros(NUM_CLASSES)

        for label in range(NUM_CLASSES):
            if label != self.true_label:
                wrong_ux = ux[:, label]
                wrong_uc = uc[:, label]

                # compute difference
                diff_x = correct_lx - wrong_ux
                diff_c = correct_lc - wrong_uc

                # compute lower bound of the difference
                mask = torch.sign(diff_x)
                mask_pos = torch.zeros_like(mask)
                mask_neg = torch.zeros_like(mask)
                mask_pos[mask > 0] = 1
                mask_neg[mask < 0] = 1

                mask_lower = x_min * mask_pos + x_max * mask_neg
                l = torch.sum(mask_lower * diff_x) + diff_c
                self.lb[label] = l

        if torch.any(self.lb < 0):
            return False

        return True






