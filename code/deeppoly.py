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

        '''
        1. Network inititalization
        layers = [modLayer(layer) for layer in self.model.layers]
        self.net = nn.Sequential(*layers)

        2. Set iteration
        iterations = 10
        for iter in range(iterations):

            3. Forward pass
            l, u, lx, ux, lc, uc = self.forward()

            3. Define objectives (9 objectives)

            4. Backsubstitution to compute l,u for objectives

            5. Compute loss
            self.loss = torch.sum(torch.clamp(lb, max=0))

            6. Update parameters
            if self.loss == 0:
                return True
            else:
                self.updateParams()
        '''





        iterations = 10

        # Box evaluate (no backsub)
        layers = [modLayer(layer) for layer in self.model.layers]
        self.net = nn.Sequential(*layers)

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

        ###### BACKSUBSTITUTION
        for backsub in range(self.config.backsub_layers):
            #print('\nBACKSUB: ', backsub)
            layers = [modLayer(layer) for layer in self.model.layers]
            self.net = nn.Sequential(*layers)
            self.forward()
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.5)

            for iter in range(iterations):
                l, u, lx, ux, lc, uc = self.forward()

                objective_x = []
                objective_c = []

                # Define objective (backsub 0)
                correct_lx = lx[-1][:, self.true_label]
                correct_lc = lc[-1][self.true_label]
                for label in range(NUM_CLASSES):
                    if label != self.true_label:
                        wrong_ux = ux[-1][:, label]
                        wrong_uc = uc[-1][label]

                        objective_x.append(correct_lx - wrong_ux)
                        objective_c.append(correct_lc - wrong_uc)

                # Backsubstitution (backsub 1 onwards)
                for bs in range(backsub):
                    for obj in range(len(objective_x)):
                        objective_x_ = objective_x[obj]
                        objective_c_ = objective_c[obj]

                        mask = torch.sign(objective_x_)
                        mask_pos = torch.zeros_like(mask)
                        mask_neg = torch.zeros_like(mask)
                        mask_pos[mask > 0] = 1
                        mask_neg[mask < 0] = 1
                        mask_pos = mask_pos * objective_x_
                        mask_neg = mask_neg * objective_x_

                        lx_ = lx[-bs-2].T
                        ux_ = ux[-bs-2].T
                        lc_ = lc[-bs-2].T
                        uc_ = uc[-bs-2].T

                        # compute objective_c_
                        objective_c_ = torch.sum(mask_pos * lc_ + mask_neg * uc_) + objective_c_

                        # compute objective_x_
                        mask_pos = mask_pos.unsqueeze(1).repeat(1, lx_.shape[1])
                        mask_neg = mask_neg.unsqueeze(1).repeat(1, ux_.shape[1])
                        objective_x_ = torch.sum(mask_pos * lx_ + mask_neg * ux_, dim=0)

                        # update objective_x, objective_c
                        objective_x[obj] = objective_x_
                        objective_c[obj] = objective_c_

                objective_x = torch.stack(objective_x).T
                objective_c = torch.stack(objective_c).T

                #print('objective_x:', objective_x.shape)
                #print('objective_c:', objective_c.shape)
                #print('l: ', l[-backsub-2].shape)

                # Insert l, u to check verification
                l_ = l[-backsub-2].unsqueeze(0).T.repeat(1, objective_x.shape[1])
                u_ = u[-backsub-2].unsqueeze(0).T.repeat(1, objective_x.shape[1])

                # compute lower bound of the difference
                mask = torch.sign(objective_x)
                mask_pos = torch.zeros_like(mask)
                mask_neg = torch.zeros_like(mask)
                mask_pos[mask > 0] = 1
                mask_neg[mask < 0] = 1

                mask_lower = l_ * mask_pos + u_ * mask_neg
                lb = torch.sum(mask_lower * objective_x, dim=0) + objective_c

                # Penalize negative values (not verified)
                self.loss = torch.sum(torch.clamp(lb, max=0))
                #print('loss: ', self.loss)

                if self.loss == 0:
                    return True
                else:
                    self.updateParams()












