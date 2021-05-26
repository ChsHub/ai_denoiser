from datetime import datetime
from logging import info, error
from os import listdir, remove
from os.path import join, exists

import torch
from numpy import product
from send2trash import send2trash
from torch import nn
from torch.nn import functional


class Net(nn.Module):

    def __init__(self, size):
        super(Net, self).__init__()

        # kernel
        convolutions = [8, 16, 7]
        # 3x3 square convolutions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=convolutions[0], kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=convolutions[0], out_channels=convolutions[1], kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=convolutions[1], out_channels=convolutions[2], kernel_size=3)
        self.flat_features = 14 * 14 * convolutions[2]  # TODO SMALLER THAN IMAGE?
        self.fc1 = nn.Linear(self.flat_features, size * size * 3)
        # self.fc2 = nn.Linear(size * size * 3, size * size * 3)
        info('CONV LAYERS: %s' % convolutions)
        # info('NUM NEURONS: %s' % (convolutions[0] * 24 * 24 + convolutions[1] * 4 * 4))

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.flat_features)
        x = functional.relu(self.fc1(x))
        # x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        return product(size.numpy())

    def load_last_state(self, directory: str = 'nets') -> (int, float):
        """
        Load previous network state with lowest lost
        :param directory: Net state directory
        :return: Last epoch number, and last loss value
        """
        if not exists(directory):
            info('path does not exists')
            return 0, 0.0

        net_states = listdir(directory)
        if net_states:
            last_state = net_states[-1]
            self.load_state_dict(torch.load(join(directory, last_state)))  # load last net
            info('Load: %s' % last_state)

            _, last_loss, last_epoch = last_state.split(' ')
            last_loss = float(last_loss)
            last_epoch = int(last_epoch[1:-5])

            return last_epoch, last_loss

        info('Network changed')
        return 0, 0.0

    def save_state(self, running_loss: float, epoch: int, directory: str = 'nets') -> None:
        """
        Save state if specified time has past
        :param running_loss: Running loss
        :param epoch: Current epoch count
        :param directory: Net state directory
        """
        if running_loss == float('nan'):
            error('Loss is nan [%s]' % epoch)
            raise ValueError()

        file_name = '%s %.4f [%s].pth' % (datetime.now().strftime("%Y-%m-%d-%H-%M"), running_loss, epoch)
        torch.save(self.state_dict(), join(directory, file_name))
        # print(abspath(join(directory, file_name)))

        # Delete previous states
        net_states = listdir(directory)
        while len(net_states) > 10:
            remove(join(directory, net_states.pop(0)))  # Delete old state
