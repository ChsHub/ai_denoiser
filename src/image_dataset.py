from logging import info
from os import walk
from os.path import join, exists, splitext
from random import shuffle, randint

import torch
from PIL import Image
from logger_default import Logger
from numpy import asarray, add, uint8, clip, array
from numpy.random import randint
from timerpy import Timer
from torch.utils.data import Dataset

from src.denoise_net import Net
from src.paths import test_image_path
from transforms.to_tensor import unnormalize, get_normalized_tensor


def add_noise(result: array, lower: int = 0) -> array:
    """
    Add random noise to array
    :param result: Image tile array
    :param lower: Lower bound for adding random noise
    :return: Noisy array
    """
    # Add noise 50% of the time
    if randint(lower, 2):
        noise = randint(-10, 10, result.shape)
        result = add(result, noise)
        result = clip(result, 0, 255)
        result = result.astype(uint8)
    return result


class ImageDataset(Dataset):
    """
    Data set for loading images from path and returning small tiles
    """

    # https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html?highlight=dataloader
    def __init__(self, image_directory: str = '../resources/dataset', size: int = 20, transform=get_normalized_tensor,
                 add_noise_always: bool = False):
        """
        Initialize image dataset
        :param image_directory: Path to the directory containing training images
        :param size: Width and height of image slices
        :param transform: Optional transform to be applied on a sample.
        """

        if not exists(image_directory):
            raise FileNotFoundError

        info('IMG SIZE: %s' % size)
        self._image_paths = []
        self._transform = transform
        self._image_directory = image_directory
        self.size = size
        self.lower = 0
        self._num_slices = self.get_slice_count(size)
        info('DATASET SIZE: %s' % self._num_slices)
        # Generators
        shuffle(self._image_paths)
        if add_noise_always:
            self.lower = 1

        self._image_generator = self.generate_slice()

    def get_slice_count(self, size: int, _image_open=Image.open) -> int:
        """
        Count how many image slices can be generated from all dataset images
        :param image_directory: Directory path containing images
        :param size: Height and width of square image slices
        :return: Total slice count
        """
        counter = 0
        for root, _, files in walk(self._image_directory):
            for file in filter(lambda x: splitext(x)[-1] in ('.webp', '.jpg', '.png'), files):
                file = join(root, file)
                with _image_open(file) as image:
                    self._image_paths.append(file)
                    size_x, size_y = image.size
                    counter += (size_x // size) * (size_y // size)
        return counter

    def __len__(self) -> int:
        """
        Get size of dataset
        :return: Number of image slices
        """
        return self._num_slices

    def generate_slice(self) -> array:
        """
        Generator method for image slices
        :return: Numpy array of noisy image data, Numpy array of original image data
        """
        for file in self._image_paths:
            with Image.open(file) as image:
                size_x, size_y = image.size
                data = asarray(image)

            for z in range(data.shape[-1]):
                for x in range(0, size_x - self.size, self.size):
                    for y in range(0, size_y - self.size, self.size):
                        tile = data[y:y + self.size, x:x + self.size, z]  # Ignore alpha layer
                        noise_tile = add_noise(tile.copy(), lower=self.lower)
                        noise_tile = noise_tile.reshape((1, self.size, self.size))
                        tile = tile.reshape((self.size * self.size))
                        yield self._transform(noise_tile), self._transform(tile)

    def __getitem__(self, idx):
        """
        Creating 3D Tensor from image. Array shape is nChannels x Height x Width.
        """
        return next(self._image_generator)

    def denoise_image(self, path):
        """
        Denoise image by using the trained net
        :param path: Image path
        :return: None
        """
        with torch.no_grad():
            net = Net(self.size)
            last_epoch, last_loss = net.load_last_state('../nets')

            for file in [path]:
                with Image.open(file) as image:
                    size_x, size_y = image.size
                    data = asarray(image).copy()
                # Iterate image tiles
                for z in range(data.shape[-1]):
                    for x in range(0, size_x - self.size, self.size):
                        for y in range(0, size_y - self.size, self.size):
                            tile = data[y:y + self.size, x:x + self.size, z]
                            tile = tile.reshape((1, self.size, self.size))
                            tile = get_normalized_tensor(tile)
                            tile = tile.unsqueeze(0)
                            tile = net(tile)
                            tile = unnormalize(tile)
                            tile = tile.reshape((self.size, self.size))
                            data[y:y + self.size, x:x + self.size, z] = tile

                return Image.fromarray(data.reshape((size_y, size_x, 3)))


if __name__ == '__main__':
    with Logger(debug=True):
        dataset = ImageDataset('../resources/dataset')
        with Timer('INFERENCE'):
            image = dataset.denoise_image(test_image_path)
        image.show()
