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


def print_accuracy(net, test_loader) -> None:
    """
    Test accuracy of the network, using test data not used during training
    :param net: Current network
    :param test_loader: Dataload of test data
    """
    correct = 0
    total = 0
    criterion = nn.L1Loss(reduction='mean')
    loss_total = 0
    with torch.no_grad():
        for data in test_loader:
            images = data[0]
            labels = data[1]
            outputs = net(images)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
            # Get loss
            loss = criterion(outputs, labels)
            loss_total += loss.item()

    info('Accuracy of the network on the training set: %.4f %% Mean loss: %.4f' % (100 * correct / total,
                                                                                   loss_total / len(test_loader)))


def format_number(string: float, length=12) -> str:
    """
    Convert float to fixed length str
    :param string: Float number
    :param length: Length of output string
    :return: String of number
    """
    string = "%.10f" % string
    length -= len(string)
    return ' ' * length + string


def get_loss_info(epoch, running_loss, prev_loss, last_loss, counter):
    """
    Print trainings info
    :param epoch: Current epoch
    :param running_loss: Running loss of current epoch
    :param prev_loss: Loss of previous epoch
    :param last_loss: Loss of previously saved epoch
    :param counter: Number of batches in one trainings epoch
    :return: String containing current trainings info
    """
    return '[%d] loss: %s\t%s\t%s  ' % (epoch,
                                        format_number(running_loss / counter),
                                        format_number((prev_loss - running_loss) / counter),
                                        format_number((last_loss - running_loss) / counter))


def save_state(net, running_loss: float, epoch: int, last_save: int, periodic_save_time: int = 10) -> int:
    """
    Save state if specified time has past
    :param net: Net
    :param running_loss: Running loss
    :param epoch: Current epoch count
    :param last_save: Time of last state saving
    :param periodic_save_time: Time in seconds between each saving
    :return: Current time
    """
    # Save the net to disk every n minutes
    if ((perf_counter_ns() - last_save) / 60_000_000_000) > periodic_save_time:
        if running_loss == float('nan'):
            error('Loss is nan [%s]' % epoch)
            raise ValueError()

        file_name = '[%s] %.4f.pth' % (epoch, running_loss)
        torch.save(net.state_dict(), join(net_dir, file_name))
        last_save = perf_counter_ns()
        info('STATE SAVED ' + file_name)

        # Delete previous states
        net_states = listdir(net_dir)
        while len(net_states) > 10:
            send2trash(join(net_dir, net_states.pop(0)))  # delete old state

    return last_save


def train_network(dataset_path, device, lr, momentum, batch_size: int, check_accuracy=False):
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
            print_accuracy(net, DataLoader(ImageDataset('resources/test_dataset', transform=ToTensor()),
                                           batch_size=batch_size, shuffle=True, num_workers=0))

    # Load Optimizer and Loss function
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    prev_loss = last_loss
    repetitions = 60
    # counter = repetitions * len(train_loader) // batch_size
    last_save = perf_counter_ns()

    # Loop over the dataset multiple times
    for epoch in range(last_epoch + 1, 48000):
        running_loss = 0.0
        counter = 0
        with Timer('Net Training: Epoch', log_function=info) as timer:
            for _ in range(repetitions):
                for data in train_loader:
                    counter += 1
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
                    last_save = save_state(net, running_loss / counter, epoch, last_save)

            timer._message += get_loss_info(epoch, running_loss, prev_loss, last_loss, counter)

        prev_loss = running_loss

    info('Finished Training')


if __name__ == '__main__':
    with Logger(debug=True, max_logfile_count=50):
        torch.autograd.set_detect_anomaly(False)  # Turn off for performance
        device = get_cuda_device()
        train_network(dataset_path, device, lr=0.001, momentum=0.4, batch_size=4, check_accuracy=True)
