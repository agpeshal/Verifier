import argparse
import torch
from networks import FullyConnected, Conv

import deeppoly_fc
import deeppoly_conv
from configuration import Configuration
import warnings

DEVICE = 'cpu'
INPUT_SIZE = 28

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)
torch.set_grad_enabled(True)

parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
parser.add_argument('--net',
                    type=str,
                    choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                    required=False,
                    help='Neural network architecture which is supposed to be verified.')
parser.add_argument('--spec', type=str, required=False, help='Test case to verify.')
args = parser.parse_args()


def analyze(net, inputs, eps, true_label):
    if 'fc' in args.net:
        model = deeppoly_fc.Model(net, eps=eps, x=inputs, true_label=true_label)
    elif 'conv' in args.net:
        model = deeppoly_conv.Model(net, eps=eps, x=inputs, true_label=true_label)
    del net
    
    # load configuration based on architecture
    config = Configuration(args=args)

    if model.verify(config):
        print('verified')
    else:
        print('not verified')


def main():
    # Load network
    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False

    # Load image
    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

    # Sanity check
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    # Test verification
    analyze(net, inputs, eps, true_label)


if __name__ == '__main__':
    main()
