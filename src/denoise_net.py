from datetime import datetime
from logging import info, error
from os import listdir, remove
from os.path import join, exists

import torch
from torch import nn, Tensor
from torch.nn import Linear
from torch.nn.functional import relu


class Net(nn.Module):

    def __init__(self, size):
        super(Net, self).__init__()

        # kernel
        convolutions = [5]
        info('CONV LAYERS: %s' % convolutions)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=convolutions[0], kernel_size=(5, 5))
        # Flat
        self.flat_features = 16 * 16 * convolutions[-1]
        linears = [self.flat_features, 20 * 20, 20 * 20, 20 * 20, 20 * 20]
        info('LIN LAYERS: %s' % linears)
        self.fc1 = Linear(linears[0], linears[1])
        self.fc2 = Linear(linears[1], linears[2])
        self.fc3 = Linear(linears[2], linears[3])
        self.fc4 = Linear(linears[3], linears[4])

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = x.view(-1, self.flat_features)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def load_last_state(self, directory: str = 'nets') -> (int, float):
        """
        Load previous network state with lowest lost
        :param directory: Net state directory
        :return: Last epoch number, and last loss value
        """
        if not exists(directory):
            info('path does not exists')
            return 0, 0.0

        try:
            net_states = listdir(directory)
            if net_states:
                last_state = net_states[-1]
                self.load_state_dict(torch.load(join(directory, last_state)))  # load last net
                info('Load: %s' % last_state)

                _, last_loss, last_epoch = last_state.split(' ')
                last_loss = float(last_loss)
                last_epoch = int(last_epoch[1:-5])

                return last_epoch, last_loss
        except RuntimeError:
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

        # Delete previous states
        net_states = listdir(directory)
        while len(net_states) > 10:
            remove(join(directory, net_states.pop(0)))  # Delete old state
