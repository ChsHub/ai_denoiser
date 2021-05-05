from logging import info
from os import listdir

import torch
from numpy import product
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):

    def __init__(self, size):
        super(Net, self).__init__()

        # kernel
        convolutions = [8, 16]
        # 3x3 square convolutions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=convolutions[0], kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=convolutions[0], out_channels=convolutions[1], kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=convolutions[1], out_channels=3, kernel_size=3)
        self.flat_features = 3 * 14 * 14
        self.fc1 = nn.Linear(self.flat_features, size * size * 3)
        self.fc2 = nn.Linear(size * size * 3, size * size * 3)
        info('CONV LAYERS: %s' % convolutions)
        # info('NUM NEURONS: %s' % (convolutions[0] * 24 * 24 + convolutions[1] * 4 * 4))

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.flat_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        return product(size.numpy())

    def load_last_state(self, directory: str = 'nets') -> (int, float):
        try:
            net_states = listdir(directory)
            if net_states:
                last_state = net_states[-1]
                self.load_state_dict(torch.load('nets/%s' % last_state))  # load last net
                info('Load: %s' % last_state)

                last_epoch, last_loss = last_state.split(' ')
                last_loss = float(last_loss[:-4])
                last_epoch = int(last_epoch[1:-1])

                return last_epoch, last_loss
        except Exception:
            info('Network changed')
        return 0, 0.0
