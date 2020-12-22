from layers_fc import modLayer, Objective, ReLU
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import ReduceLROnPlateau

INPUT_SIZE = 28
NUM_CLASSES = 10

class Model(nn.Module):
    def __init__(self, model, eps, x, true_label):
        super().__init__()
        # remove the batch dimension
        x = x.squeeze(0)
        self.true_label = true_label
        # find input range while ensuring values stay in [0, 1]
        self.x_min = torch.clamp(x.data - eps, min=0)
        self.x_max = torch.clamp(x.data + eps, max=1)
        self.model = model


    def forward(self):
        return self.net([[self.x_min], [self.x_max], [], [], [], []])  # [[l], [u], [lx], [ux], [lc], [uc]]


    def parameters(self):
        '''
        Returns all the learnable parameters
        '''
        params = []
        for layer in self.net:
            if isinstance(layer, ReLU) and hasattr(layer, 'slope_upper'):
                # add slopes of the lower and upper bound of ReLU
                params.append(layer.slope_upper)
                params.append(layer.slope_lower)
        
        return params


    # Calculates the gradient of `loss` wrt to ReLU slopes.
    def updateParams(self):
        loss = -self.loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    
    # Modified layers after transformation
    def get_layers(self):
        layers = []
        for i in range(len(self.model.layers)):
            layers.append(modLayer(self.model.layers[i]))
        return layers


    def verify(self, config):
        lr = config.lr

        # Initialize transformed network
        layers = self.get_layers()
        layers.append(Objective(num_classes=NUM_CLASSES, true_label=self.true_label))
        self.net = nn.Sequential(*layers)
        self.forward()

        # Catch if the parameter list is empty
        try:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        except:
            print('Nothing to optimize')
            return False
        
        # Adjust learning for deep networks
        if int(config.net[-1]) > 3:
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=config.patience, factor=0.5)

        while True:
            l, u, lx, ux, lc, uc = self.forward()

            # Penalize negative values (not verified)
            idx = torch.where(l[-1] < 0)[0]
            self.loss = torch.sum(l[-1][idx])


            if self.loss == 0:
                return True

            self.updateParams()
            if int(config.net[-1]) > 3:
                self.scheduler.step(-self.loss)

