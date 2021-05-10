from logging import info, error
from os import listdir
from os.path import join
from time import perf_counter_ns

import torch
from numpy import product
from send2trash import send2trash
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
        """
        Load previous network state with lowest lost
        :param directory: Net state directory
        :return: Last epoch number, and last loss value
        """
        try:
            net_states = listdir(directory)
            if net_states:
                last_state = net_states[-1]
                self.load_state_dict(torch.load('nets/%s' % last_state))  # load last net
                info('Load: %s' % last_state)

                last_loss, last_epoch = last_state.split(' ')
                last_loss = float(last_loss)
                last_epoch = int(last_epoch[1:-5])

                return last_epoch, last_loss
        except Exception:
            info('Network changed')
        return 0, 0.0

    def save_state(self, running_loss: float, epoch: int, last_save: int, directory: str = 'nets', periodic_save_time: int = 10) -> int:
        """
        Save state if specified time has past
        :param net: Net
        :param running_loss: Running loss
        :param epoch: Current epoch count
        :param last_save: Time of last state saving
        :param directory: Net state directory
        :param periodic_save_time: Time in seconds between each saving
        :return: Current time
        """
        # Save the net to disk every n minutes
        if ((perf_counter_ns() - last_save) / 60_000_000_000) > periodic_save_time:
            if running_loss == float('nan'):
                error('Loss is nan [%s]' % epoch)
                raise ValueError()

            file_name = '%.4f [%s].pth' % (running_loss, epoch)
            torch.save(self.state_dict(), join(directory, file_name))
            last_save = perf_counter_ns()
            info('STATE SAVED ' + file_name)

            # Delete previous states
            net_states = listdir(directory)
            while len(net_states) > 10:
                send2trash(join(directory, net_states.pop()))  # Delete old state

        return last_save
