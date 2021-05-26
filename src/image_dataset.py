from logging import info
from os import walk
from os.path import join, exists, splitext
from random import shuffle, randint

import torch
from PIL import Image
from logger_default import Logger
from numpy import asarray, add, uint8, clip, array
from numpy.random import randint
from torch import Tensor
from torch.utils.data import Dataset

from src.denoise_net import Net
from transforms.to_tensor import ToTensor, unnormalize, get_normalized_tensor


def _noise_open(path: str) -> (array, int, int):
    """
    Open image and apply random noise
    :param path: Image path
    :return: Image object
    """
    with Image.open(path) as image:
        image = image.convert('RGB')
        result = asarray(image)
        # Add noise 50% of the time
        if 1 == randint(0, 1):
            # info('ADD NOISE')
            noise = randint(-10, 10, result.shape)
            result = add(result, noise)
            result = clip(result, 0, 255)
            result = result.astype(uint8)
        # else:
        # info('NO  NOISE')
        return result, *image.size


def _normal_open(path: str) -> (array, int, int):
    """
    Open image and apply random noise
    :param path: Image path
    :return: Image object, x and y size
    """
    with Image.open(path) as image:
        image = image.convert('RGB')
        return asarray(image), *image.size


class ImageDataset(Dataset):

    # https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html?highlight=dataloader
    def __init__(self, image_directory: str = '../resources/dataset', size: int = 20, transform=None):
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
        self.transform = transform
        self._image_directory = image_directory
        self._size = size
        self._num_slices = self.get_slice_count(size)
        info('DATASET SIZE: %s' % self._num_slices)
        # Generators
        shuffle(self._image_paths)
        self._noise_image_generator = self.generate_slice(self._image_paths, size, (3, size, size), _noise_open)
        self._image_generator = self.generate_slice(self._image_paths, size, (3 * size * size), _normal_open)

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

    def generate_slice(self, image_paths, slice_size, shape, _image_open) -> array:
        """
        Generator method for image slices
        :param image_paths: List of image paths
        :param slice_size: Width and Height of slices
        :param shape: Shape of output array
        :param _image_open: Method to open images
        :return: Numpy array of image data
        """
        for file in image_paths:
            data, size_x, size_y = _image_open(file)
            for x in range(0, size_x - self._size, self._size):
                for y in range(0, size_y - self._size, self._size):
                    yield data[y:y + slice_size, x:x + slice_size, :3].reshape(shape)  # Ignore alpha layer

    def __getitem__(self, idx):
        """
        Creating 3D Tensor from image. Array shape is nChannels x Height x Width.
        """

        data = next(self._noise_image_generator)
        data2 = next(self._image_generator)

        if self.transform:
            sample = self.transform([data, data2])

        return sample

    def _denoise_tile(self, net, generator):
        input = self.transform([next(generator)])
        input = Tensor(input[0])
        input = input.unsqueeze(0)
        output = net(input)
        return output.reshape((3, self._size, self._size))

    def denoise_image(self, path):
        """
        Denoise image by using the trained net
        :param path: Image path
        :return: None
        """
        net = Net(self._size)
        last_epoch, last_loss = net.load_last_state('../nets')
        image = []
        with Image.open(path) as img:
            width, height = img.size
            width -= (width % self._size)
            height -= (height % self._size)

        for file in [path]:
            data, size_x, size_y = _normal_open(file)
            data = data.copy()
            for x in range(0, size_x - self._size, self._size):
                for y in range(0, size_y - self._size, self._size):
                    tile1 = data[y:y + self._size, x:x + self._size, :3]  # .reshape((3, self._size, self._size))
                    tile = get_normalized_tensor(tile1)
                    tile = tile[0].unsqueeze(0)
                    tile = net(tile)
                    tile = tile.reshape((self._size, self._size, 3))
                    tile = unnormalize(tile)

                    for i in range(self._size):
                        for j in range(self._size):
                            for w in range(3):
                                if int(tile[i, j, w]) != int(tile1[i, j, w]):
                                    print(int(tile[i, j, w]), int(tile1[i, j, w]))
                                    print(tile1)
                    data[y:y + self._size, x:x + self._size, :3] = tile

        Image.fromarray(data).show()


if __name__ == '__main__':
    with Logger(debug=True):
        dataset = ImageDataset('../resources/dataset', transform=ToTensor())
        with torch.no_grad():
            dataset.denoise_image("")
        print(len(dataset))
        item = dataset[0]
        item = dataset[0]
        # show_image(item[0])
        print(item)
