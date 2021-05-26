from time import strftime

import numpy
import numpy as np
import torch
from PIL import Image
from numpy import ndarray, multiply, int16, divide, subtract
from torch import tensor


def show_image(n: tensor):
    """
    Undo normalization and show image
    :param n: Image tensor
    """
    n = unnormalize(n)
    n = n.reshape((n.shape[1], n.shape[2], n.shape[0]))
    Image.fromarray(n).show()


def unnormalize(n: tensor):
    """
    Turn Tensor into image tile
    :param n: Tensor with value range [-1, 1]
    :return: Numpy image tile with value range [0, 255]
    """
    n = n.numpy()
    n += 1.0  # [-1,1] to [0,2]
    n = multiply(n, 127.5001)  # [0,2] to [0, 255]
    n = n.astype(numpy.uint8)
    return n


def get_normalized_tensor(input_tensor: ndarray):
    """
    Turn Image tile into Tensor
    :param input_tensor: Numpy image tile with value range [0, 255]
    :return: Tensor with value range [-1, 1]
    """
    input_tensor = divide(input_tensor, 127.5, dtype=np.float)  # Convert image values from [0,255] to [0,2]
    input_tensor = subtract(input_tensor, 1.0)  # Convert image values from [0,2] to [-1,1]
    input_tensor = tensor(input_tensor.copy(), dtype=torch.float)
    return input_tensor


class ToTensor:
    def __call__(self, sample):
        # Convert to tensors, set float type and normalize
        return list(map(get_normalized_tensor, sample))
