from unittest import TestCase
import hypothesis
from numpy import array

from transforms.to_tensor import unnormalize, get_normalized_tensor


def test_unnormalize():
    """
    Test identity normalization and unnormalization
    :return:
    """
    comp = unnormalize(get_normalized_tensor(array(range(256)))) == array(range(256))
    assert comp.all()


def test_show_image(self):
    self.fail()


def test_get_normalized_tensor(self):
    self.fail()


def test_to_tensor(self):
    self.fail()


if __name__ == '__main__':
    test_unnormalize()
