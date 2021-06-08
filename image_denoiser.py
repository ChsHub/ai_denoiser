from logging import info

import torch
from logger_default import Logger
from timerpy import Timer
from torch import nn, optim
from torch.utils.data import DataLoader

from src.denoise_net import Net
from src.image_dataset import ImageDataset
from src.paths import dataset_path
from src.saver import Saver

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# C:\Python38\python.exe -m pip install D:\Making\Python\image_classifier\numpy-1.19.2+mkl-cp38-cp38-win_amd64.whl
# C:\Python38\python.exe -m pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
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
    with Timer('print_accuracy', log_function=info):
        correct = 0
        total = 0
        criterion = nn.L1Loss(reduction='mean')
        loss_total = 0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
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


def get_loss_info(epoch, running_loss, last_loss, counter):
    """
    Print trainings info
    :param epoch: Current epoch
    :param running_loss: Running loss of current epoch
    :param last_loss: Loss of previously saved epoch
    :param counter: Number of batches in one trainings epoch
    :return: String containing current trainings info
    """
    return '[%d] loss: %s\t%s  ' % (epoch,
                                    format_number(running_loss / counter),
                                    format_number((last_loss - running_loss) / counter))


def train_network(dataset_path, device, lr, momentum, batch_size: int, check_accuracy=False, num_workers: int = 0):
    """
    Trainings loop
    :param dataset_path: Directory for dataset images
    :param device: GPU device
    :param lr: Learning rate
    :param momentum: Learning momentum
    :param batch_size: Batch tensor size
    :param check_accuracy: Tests net accuracy on test data
    :param num_workers: Number of additional processes launched by data-loaders
    """
    info('lr: %s, momentum: %s, batch size: %s' % (lr, momentum, batch_size))

    # Load Neural net and Data set
    image_dataset = ImageDataset(image_directory=dataset_path)
    train_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = Net(train_loader.dataset.size)
    last_epoch, last_loss = net.load_last_state()
    # net.to(device)

    # Look at accuracy from trained net
    if check_accuracy:
        print_accuracy(net, DataLoader(
            ImageDataset('resources/test_dataset', add_noise_always=True),
            batch_size=batch_size, shuffle=False, num_workers=num_workers))

    # Load Optimizer and Loss function
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    repetitions = 60
    saver = Saver(net)
    saver.start()

    # Loop over the dataset multiple times
    for saver.epoch in range(last_epoch + 1, 48000):
        saver.running_loss = 0.0
        saver.counter = 0

        with Timer('Net Training: Epoch', log_function=info) as saver.timer:
            for _ in range(repetitions):
                for data in train_loader:
                    saver.counter += 1
                    # Zero the parameter gradients
                    for param in net.parameters():
                        param.grad = None

                    inputs, labels = data  # .to(device)
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    saver.running_loss += loss.item()

            saver.timer._message += get_loss_info(saver.epoch, saver.running_loss, last_loss, saver.counter)


if __name__ == '__main__':
    with Logger(debug=True, max_logfile_count=50):
        torch.autograd.set_detect_anomaly(False)  # Turn off for performance
        device = get_cuda_device()
        train_network(dataset_path, device, lr=0.001, momentum=0.9, batch_size=4, check_accuracy=True)
