import torch
import torch.nn as nn


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class FullyConnected(nn.Module):

    def __init__(self, device, input_size, fc_layers):
        super(FullyConnected, self).__init__()

        layers = [Normalization(device), nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):

    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super(Conv, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device)]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Normalization_Dummy(nn.Module):
    def __init__(self):
        super(Normalization_Dummy, self).__init__()
        self.mean = 0.5
        self.sigma = 1.0

    def forward(self, x):
        return (x - self.mean) / self.sigma



class Dummy(nn.Module):
    def __init__(self) -> None:
        super(Dummy, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.layers = nn.Sequential(nn.Flatten(),
                                    self.fc1,
                                    nn.ReLU(),
                                    self.fc2,
                                    nn.ReLU(),
                                    self.fc3)
    
    def forward(self, x):
        self.layers(x)


class Dummy_Norm(nn.Module):
    def __init__(self) -> None:
        super(Dummy_Norm, self).__init__()
        self.normalization = Normalization_Dummy()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 3)
        self.layers = nn.Sequential(self.normalization,
                                    nn.Flatten(),
                                    self.fc1,
                                    nn.ReLU(),
                                    self.fc2)

    def forward(self, x):
        self.layers(x)
