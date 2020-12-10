import argparse
import torch
from networks import FullyConnected, Conv
import time

import logging
import deeppoly_fc
import deeppoly_conv
import warnings

DEVICE = 'cpu'
INPUT_SIZE = 28

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
parser.add_argument('--net',
                    type=str,
                    choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                    required=False,
                    help='Neural network architecture which is supposed to be verified.')
parser.add_argument('--spec', type=str, required=False, help='Test case to verify.')
parser.add_argument('--debug', default=True, required=False, help='Flag to enable debug.')
args = parser.parse_args()

torch.set_grad_enabled(True)
logging.basicConfig(level=(10 if args.debug else 20), format="%(asctime)s :: %(message)s")
logger = logging.getLogger(__name__)


def analyze(net, inputs, eps, true_label):
    start_time = time.time()
    print('Initialization ...')
    if 'fc' in args.net:
        model = deeppoly_fc.Model(net, eps=eps, x=inputs, true_label=true_label)
    else:
        model = deeppoly_conv.Model(net, eps=eps, x=inputs, true_label=true_label)
    del net
    print('initialization: time=', round(time.time() - start_time, 4))

    iter = 1
    while iter <= 50:
        print('\nIteration: ', iter)
        start_time = time.time()
        is_verified = model.verify()
        print('verification: time=', round(time.time() -  start_time,  4))

        if is_verified:
            return True
        else:
            start_time = time.time()
            model.updateParams()
            print('update param: time=', round(time.time() - start_time, 4))
            iter += 1

    return False


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
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    # Load image
    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    # Test verification
    total_time = time.time()
    is_verified = analyze(net, inputs, eps, true_label)

    print('\n\nResult:')
    if is_verified:
        print('verified')
    else:
        print('not verified')
    print('Total time=', round(time.time() - total_time, 4), 's\n')


if __name__ == '__main__':
    main()
