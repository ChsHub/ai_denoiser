# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# C:\Python38\python.exe -m pip install D:\Making\Python\image_classifier\numpy-1.19.2+mkl-cp38-cp38-win_amd64.whl
# C:\Python38\python.exe -m pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
from logging import info, error
from os import listdir
from os.path import join
from time import perf_counter_ns

import torch
from logger_default import Logger
from send2trash import send2trash
from timerpy import Timer
from torch import nn, optim
from torch.utils.data import DataLoader

from src.denoise_net import Net
from src.image_dataset import ImageDataset
from src.paths import dataset_path
from transforms.to_tensor import ToTensor

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
net_dir = 'nets'


def get_cuda_device():
    info('CUDA available: %s' % torch.cuda.is_available())
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    info(device)
    return device


def get_accuracy(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images = data[0]
            labels = data[1]
            outputs = net(images)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()

    info('Accuracy of the network on the training set: %.4f %%' % (100 * correct / total))


def get_info(string: float, length=12) -> str:
    string = "%.4f" % string
    length -= len(string)
    return ' ' * length + string


def train_network(dataset_path, device, lr, momentum, batch_size, check_accuracy=False):
    """
    Trainings loop
    :param dataset_path: Directory for dataset images
    :param device: GPU device
    :param lr: Learning rate
    :param momentum: Learning momentum
    :param batch_size: Batch tensor size
    :param check_accuracy: Tests net accuracy on test data
    """
    info('lr: %s, momentum: %s, batch size: %s' % (lr, momentum, batch_size))

    # Load Neural net and Data set
    size = 20
    train_loader = DataLoader(
        ImageDataset(size=size, image_directory=dataset_path, transform=ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=0)

    net = Net(size)
    last_epoch, last_loss = net.load_last_state()
    # net.to(device)

    # Look at accuracy from trained net
    if check_accuracy:
        with Timer('Get accuracy', log_function=info):
            get_accuracy(net, DataLoader(ImageDataset('resources/test_dataset', transform=ToTensor()),
                                         batch_size=batch_size, shuffle=True, num_workers=0))

    # Load Optimizer and Loss function
    criterion = nn.L1Loss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    running_loss = 0.0
    prev_loss = last_loss
    repetitions = 60
    counter = repetitions * len(train_loader) // batch_size
    last_save = perf_counter_ns()

    # Loop over the dataset multiple times
    for epoch in range(last_epoch + 1, 48000):
        with Timer('Net Training: Epoch', log_function=info) as timer:
            for _ in range(repetitions):
                for data in train_loader:
                    # Zero the parameter gradients
                    for param in net.parameters():
                        param.grad = None

                    inputs = data[0]  # .to(device)
                    labels = data[1]  # .to(device)
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            timer._message += '[%d] loss: %s\t%s\t%s  ' % (epoch,
                                                           get_info(running_loss / counter),
                                                           get_info((prev_loss - running_loss) / counter),
                                                           get_info((last_loss - running_loss) / counter))

        if running_loss == float('nan'):
            error('Loss is nan [%s]' % epoch)
            raise ValueError()

        # Save the net to disk every 10 minutes
        if ((perf_counter_ns() - last_save) / 60_000_000_000) > 1:
            torch.save(net.state_dict(), join(net_dir, '[%s] %.4f.pth' % (epoch, running_loss)))
            info('STATE SAVED')
            last_save = perf_counter_ns()

        # Delete previous states
        net_states = listdir(net_dir)
        while len(net_states) > 30:
            send2trash(join(net_dir, net_states.pop(0)))  # delete old state

        prev_loss = running_loss
        running_loss = 0.0

    info('Finished Training')


if __name__ == '__main__':
    with Logger(debug=True, max_logfile_count=50):
        torch.autograd.set_detect_anomaly(False)  # Turn off for performance
        device = get_cuda_device()
        train_network(dataset_path, device, lr=0.001, momentum=0.4, batch_size=4)
