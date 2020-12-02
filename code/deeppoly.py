import logging
import torch
import torch.nn as nn

from layers import initialize_properties, modLayer, ReLU

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

INPUT_SIZE = 28
NUM_CLASSES = 10


class Model(nn.Module):
    def __init__(self, model: nn.Module, eps: float, x: torch.Tensor, true_label: int):
        super().__init__()

        self.x_min = torch.clamp(x.data - eps, min=0)        # input interval lower bound (values)
        self.x_max = torch.clamp(x.data + eps, max=1)        # input interval upper bound
        initialize_properties(input=[self.x_min, self.x_max], trainable=False)

        layers = [modLayer(layer) for layer in model.layers]
        self.net = nn.Sequential(*layers)

        self.lx = torch.ones_like(x)                    # relational lower bound (coefficients w.r.t. input neurons)
        self.ux = torch.ones_like(x)                    # relational uppper bound
        self.lc = torch.zeros_like(x)                   # relational lower bound coefficient
        self.uc = torch.zeros_like(x)                   # relational uppper bound coefficient

        self.true_label = true_label
        self.forward()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05)

    def forward(self):
        return self.net([self.lx, self.ux, self.lc, self.uc])

    def parameters(self) -> torch.Tensor:
        for layer in self.net:
            if isinstance(layer, ReLU) and hasattr(layer, "slope"):
                yield layer.slope
    
    def updateParams(self):
        # Calculates the gradient of `loss` wrt to ReLU slopes.
        loss = -self.lb.sum()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
    
    def verify(self):
        lx, ux, lc, uc = self.forward()
        x_min = self.x_min.flatten(1, -1).squeeze()
        x_max = self.x_max.flatten(1, -1).squeeze()

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
        





