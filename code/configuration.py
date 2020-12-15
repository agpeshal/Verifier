class Config:
    def __init__(self, args, net):
        self.args = args
        self.model = net
        self.get_verifier_params()

    def get_verifier_params(self):
        self.trainable = False

        self.n_layers = len(self.model.layers)

        backsub = {'fc1': 3,
                    'fc2': 5,
                    'fc3': 5,
                    'fc4': 7,
                    'fc5': 7,
                    'fc6': 9,
                    'fc7': 11,
                    'conv1': 3,
                    'conv2': 3,
                    'conv3': 5}
        self.backsub_layers = backsub[self.args.net]

        self.verbose = False

        self.forward_layers = self.n_layers - self.backsub_layers




