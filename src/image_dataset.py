from logging import info
from os import walk
from os.path import join

from PIL import Image
from numpy import asarray, add, uint8, clip
from numpy.random import randint
from torch import Tensor
from torch.utils.data import Dataset

from transforms.to_tensor import ToTensor


def show_image(input_tensor: Tensor, index: int) -> None:
    """
    Display image from Tensor
    :param input_tensor: Tensor to show
    :param index: Image number in tensor
    """
    Image.fromarray(input_tensor.numpy()[index, :, :, ::], 'RGB').show()


def _noise_open(path: str) -> Image:
    """
    Open image and apply random noise
    :param path: Image path
    :return: Image object
    """
    result = Image.open(path)
    result = asarray(result)
    noise = randint(-10, 10, result.shape)
    result = add(result, noise)
    result = clip(result, 0, 255)
    result = result.astype(uint8)
    result = Image.fromarray(result, 'RGB')
    return result


class ImageDataset(Dataset):

    # https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html?highlight=dataloader
    def __init__(self, image_directory: str = '../resources/dataset', size: int = 20, transform=None):
        """
        Initialize image dataset
        :param image_directory: Path to the directory containing training images
        :param size: Width and height of image slices
        :param transform: Optional transform to be applied on a sample.
        """

        info('IMG SIZE: %s' % size)
        self.transform = transform
        self._image_directory = image_directory
        self._size = size
        self._num_slices = self.get_slice_count(image_directory, size)
        self._image_generator = self.generate_slice()
        self._noise_image_generator = self.generate_slice(_noise_open)

    @staticmethod
    def get_slice_count(image_directory: str, size: int) -> int:
        """
        Count how many image slices can be generated from all dataset images
        :param image_directory: Directory path containing images
        :param size: Height and width of square image slices
        :return: Total slice count
        """
        counter = 0
        for root, _, files in walk(image_directory):
            for file in files:
                with Image.open(join(root, file)) as image:
                    size_x, size_y = image.size
                    counter += (size_x // size) * (size_y // size)
        return counter

    def __len__(self) -> int:
        """
        Get size of dataset
        :return: Number of image slices
        """
        return self._num_slices

    def generate_slice(self, _image_open=Image.open):
        """
        Generator method for image slices
        :param _image_open: Method to open images
        :return:
        """
        for root, _, files in walk(self._image_directory):
            for file in files:
                with _image_open(join(root, file)) as image:
                    data = asarray(image)
                    size_x, size_y = image.size
                    for x in range(0, size_x - self._size, self._size):
                        for y in range(0, size_y - self._size, self._size):
                            yield data[y:y + self._size, x:x + self._size, :]

    def __getitem__(self, idx):
        """
        Creating 3D Tensor from image. Array shape is nChannels x Height x Width.
        """

        # if torch.is_tensor(idx):
        #    idx = idx.tolist()

        sample = {'image': next(self._noise_image_generator),
                  'character': next(self._image_generator)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def denoise_image(self, net, path):
        pass


if __name__ == '__main__':
    dataset = ImageDataset('../resources/dataset', transform=ToTensor())
    print(len(dataset))
    item = dataset[0]
    item = dataset[0]
    print(item)
