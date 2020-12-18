class Configuration:
    def __init__(self, args):
        self.args = args
        self.get_verifier_params()

    def get_verifier_params(self):
        iterations = {'fc1': 20,
                      'fc2': 20,
                      'fc3': 20,
                      'fc4': 20,
                      'fc5': 20,
                      'fc6': 20,
                      'fc7': 20,
                      'conv1': 20,
                      'conv2': 20,
                      'conv3': 20}

        lr = {'fc1': 0.01,
              'fc2': 0.01,
              'fc3': 0.01,
              'fc4': 0.01,
              'fc5': 0.01,
              'fc6': 0.01,
              'fc7': 0.01,
              'conv1': 0.5,
              'conv2': 0.5,
              'conv3': 0.5}

        self.iterations = iterations[self.args.net]
        self.lr = lr[self.args.net]


