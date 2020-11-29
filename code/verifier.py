import argparse
import torch
from networks import FullyConnected, Conv

import logging
import deeppoly

DEVICE = 'cpu'
INPUT_SIZE = 28


parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
parser.add_argument('--net',
                    type=str,
                    choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                    required=True,
                    help='Neural network architecture which is supposed to be verified.')
parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
parser.add_argument('--debug', default=True, required=False, help='Flag to enable debug.')
args = parser.parse_args()

torch.set_grad_enabled(True)
logging.basicConfig(level=(10 if args.debug else 20), format="%(asctime)s :: %(message)s")
logger = logging.getLogger(__name__)


def analyze(net, inputs, eps, true_label):
    base_pred = net(inputs)     # logits
    model = deeppoly.Model(net, eps=eps, x=inputs, true_label=true_label)
    del net

    model.verify()

    exit()

    '''
    logger.debug(f"[+] True label: {true_label}, Epsilon: {eps}")
    logger.debug(f"Base predictions: {base_pred[0]}")
    while not model.verify():
        model.updateParams()
    return True

    return 0
    #'''


def main():


    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

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

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()