from logging import info
from os import listdir

import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # kernel
        convolutions = [8, 16]
        linears = [convolutions[-1] * 4 * 4, 500, 200, 100, 50, character_count]  # 6*6 from image dimension
        # 3x3 square convolutions
        self.conv1 = nn.Conv2d(1, convolutions[0], 3)  # 1 input image channel, <a> output channels
        self.conv2 = nn.Conv2d(convolutions[0], convolutions[1], 3)  # <a> input image channel, <b> output channels
        # an affine operation: y = Wx + b

        self.fc1 = nn.Linear(linears[0], linears[1])
        self.fc2 = nn.Linear(linears[1], linears[2])
        self.fc3 = nn.Linear(linears[2], linears[3])
        self.fc4 = nn.Linear(linears[3], linears[4])
        self.fc5 = nn.Linear(linears[4], linears[5])

        info('CONV LAYERS: %s' % convolutions)
        info('LINE LAYERS: %s' % linears)
        info('NUM NEURONS: %s' % (sum(linears) + convolutions[0] * 24 * 24 + convolutions[1] * 4 * 4))
        # params = list(self.parameters())
        # for para in params:
        #    print(para.size())

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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
