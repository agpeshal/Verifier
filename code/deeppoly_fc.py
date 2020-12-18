from layers_fc import *
torch.autograd.set_detect_anomaly(True)

INPUT_SIZE = 28
NUM_CLASSES = 10


class Model(nn.Module):
    def __init__(self, model, eps, x, true_label):
        super().__init__()
        x = x.squeeze(0)

        self.true_label = true_label
        self.x_min = torch.clamp(x.data - eps, min=0)
        self.x_max = torch.clamp(x.data + eps, max=1)
        self.model = model


    def forward(self):
        return self.net([[self.x_min], [self.x_max], [], [], [], []])       # [[l], [u], [lx], [ux], [lc], [uc]]


    def parameters(self):
        for layer in self.net:
            if isinstance(layer, ReLU) and hasattr(layer, 'slope'):
                yield layer.slope


    # Calculates the gradient of `loss` wrt to ReLU slopes.
    def updateParams(self):
        loss = -self.loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()


    def get_layers(self):
        layers = []
        for i in range(len(self.model.layers)):
            layers.append(modLayer(self.model.layers[i]))
        return layers


    def verify(self, config):
        iterations = config.iterations
        lr = config.lr

        # Initialize transformed network
        layers = self.get_layers()
        layers.append(Verifier(num_classes=NUM_CLASSES, true_label=self.true_label))
        self.net = nn.Sequential(*layers)

        # Optimizer
        self.forward()
        # If not crossing ReLU in Conv -> parameters empty!
        try:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        except:
            print('Nothing to optimize')
            return False

        for i in range(iterations):
            l, u, lx, ux, lc, uc = self.forward()

            # Penalize negative values (not verified)
            idx = torch.where(l[-1] < 0)[0]
            self.loss = torch.sum(l[-1][idx])

            # self.loss = torch.sum(l[-1])
            # if not torch.any(l[-1] < 0):
            #     return True

            if self.loss == 0:
                return True

            self.updateParams()

            # print('Iteration: {}, loss: {}'.format(i, -self.loss.item()))


