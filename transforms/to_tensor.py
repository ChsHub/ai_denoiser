from time import strftime

import numpy
import numpy as np
import torch
from PIL import Image
from torch import tensor


def show_image(n: tensor):
    """
    Undo normalization and show image
    :param n: Image tensor
    """
    n = n.numpy()
    n += 1  # Pixel range [0, 2]
    n = np.multiply(n, 127.5)  # Pixel range [0, 255]
    n = n.astype(numpy.uint8)
    n = n.reshape((n.shape[1], n.shape[2], n.shape[0]))
    Image.fromarray(n).show()


def get_normalized_tensor(input_tensor):
    input_tensor = input_tensor * 2 / 255 - 1  # Convert image values from [0,255] to [-1,1]
    input_tensor = tensor(input_tensor.copy(), dtype=torch.float)
    return input_tensor


class ToTensor:
    def __call__(self, sample):
        # Convert to tensors, set float type and normalize
        return list(map(get_normalized_tensor, sample))
