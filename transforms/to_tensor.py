from time import strftime

import numpy
import torch
from PIL import Image
from torch import tensor


def normalize_tensor(n: tensor):
    """
    Convert image values from [0,255] to [-1,1]
    """
    return n * 2 / 255 - 1


def show_image(n: tensor):
    """
    Convert image values from [0,255] to [-1,1]
    """
    if strftime('%M') < '21':
        return
    n = n.numpy()
    n = n.squeeze(0)
    min_value = numpy.amin(n)

    n -= min_value  # Lowest value = 0
    max_value = numpy.amax(n)
    n *= 255
    n /= max_value

    n = n.astype(numpy.int32)
    Image.fromarray(n).show()


def get_normalized_tensor(image):
    input_tensor = numpy.expand_dims(numpy.asarray(image), axis=0)
    input_tensor = tensor(input_tensor.copy(), dtype=torch.float)
    input_tensor = normalize_tensor(input_tensor)
    return input_tensor


class ToTensor:
    def __call__(self, sample):
        # Convert to tensors, set float type and normalize
        return {'image': get_normalized_tensor(sample['image']),
                'character': get_normalized_tensor(sample['character'])}
