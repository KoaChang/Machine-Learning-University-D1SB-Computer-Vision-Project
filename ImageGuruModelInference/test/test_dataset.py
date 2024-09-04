import numpy as np
import os
from PIL import Image, ImageOps
import tempfile
import torchvision
import unittest

from image_guru_model_inference.dataset import ImageDataset


class TestImageDataset(unittest.TestCase):

    def _create_random_image(self, seed=0):
        np.random.seed(seed)
        imarr = 255 * np.random.rand(512, 512, 3)
        imarr = imarr.astype('uint8')
        img = Image.fromarray(imarr).convert("RGB")
        return img

    def test_dataset(self):
        img0 = self._create_random_image(seed=0)
        img1 = self._create_random_image(seed=1)
        samples = [img0, img1]

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(2),
        ])

        # create a pytorch dataset from the provided samples.
        dataset = ImageDataset(samples, sample_type='pil', transforms=transform)

        # test the dataset
        self.assertEqual(2, len(dataset))

        # create expected padded images
        exp_img0 = ImageOps.expand(img0, border=2, fill=0)
        exp_img1 = ImageOps.expand(img1, border=2, fill=0)

        # verify expected images
        self.assertEqual(0, dataset[0][0])
        np.testing.assert_equal(np.array(exp_img0), np.array(dataset[0][1]))
        self.assertEqual(1, dataset[1][0])
        np.testing.assert_equal(np.array(exp_img1), np.array(dataset[1][1]))

    def test_read_image_pil(self):
        img = self._create_random_image(seed=0)
        samples = [img]
        dataset = ImageDataset(samples, sample_type='pil', transforms=None)
        # verify expected images
        np.testing.assert_equal(np.array(img), np.array(dataset[0][1]))

    def test_read_image_numpy(self):
        img = self._create_random_image(seed=0)
        samples = [np.array(img)]
        dataset = ImageDataset(samples, sample_type='numpy', transforms=None)
        # verify expected images
        np.testing.assert_equal(np.array(img), np.array(dataset[0][1]))

    def test_read_image_from_filepath(self):
        # create a random image
        img = self._create_random_image()

        # same image to a temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fd:
            img.save(fd)
            filepath = fd.name

        samples = [filepath]
        dataset = ImageDataset(samples, sample_type='file', transforms=None)

        # test of the image read from file is the same as the created one.
        np.testing.assert_equal(np.array(img), np.array(dataset[0][1]))

        # remove the temp file
        os.unlink(filepath)

    def test_read_image_from_fileobject(self):
        # create a random image
        img = self._create_random_image()

        # same image to a temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fd:
            img.save(fd)
            filepath = fd.name

        with open(filepath, 'rb') as fd:
            samples = [fd]
            dataset = ImageDataset(samples, sample_type='file', transforms=None)
            np.testing.assert_equal(np.array(img), np.array(dataset[0][1]))

        # remove the temp file
        os.unlink(filepath)

    def test_read_image_from_bytes(self):
        # create a random image
        img = self._create_random_image()

        # same image to a temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fd:
            img.save(fd)
            filepath = fd.name

        with open(filepath, 'rb') as fd:
            bytes = fd.read()

        # remove the temp file
        os.unlink(filepath)

        samples = [bytes]
        dataset = ImageDataset(samples, sample_type='bytes', transforms=None)
        np.testing.assert_equal(np.array(img), np.array(dataset[0][1]))


if __name__ == "__main__":
    unittest.main()
