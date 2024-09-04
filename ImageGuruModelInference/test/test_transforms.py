import numpy as np
from PIL import Image, ImageOps
import unittest

from image_guru_model_inference.transforms import PadToSquare


class TestPadToSquare(unittest.TestCase):

    def _create_random_image(self, height=512, width=512, seed=0):
        np.random.seed(seed)
        imarr = 255 * np.random.rand(height, width, 3)
        imarr = imarr.astype('uint8')
        img = Image.fromarray(imarr).convert("RGB")
        return img

    def test_PadToSquare_largerwidth(self):
        img = self._create_random_image(height=100, width=120)
        padded_img = PadToSquare()(img)
        # create expected padded image
        # width is larger, so top and bottom will be padded
        left, top, right, bottom = 0, 10, 0, 10
        exp_img = ImageOps.expand(img, border=(left, top, right, bottom), fill=(255,255,255))
        # verify expected image
        np.testing.assert_equal(np.array(exp_img), np.array(padded_img))

    def test_PadToSquare_largerheight(self):
        img = self._create_random_image(height=120, width=100)
        padded_img = PadToSquare()(img)
        # create expected padded image
        # height is larger, so left and right will be padded
        left, top, right, bottom = 10, 0, 10, 0
        exp_img = ImageOps.expand(img, border=(left, top, right, bottom), fill=(255,255,255))
        # verify expected image
        np.testing.assert_equal(np.array(exp_img), np.array(padded_img))

    def test_PadToSquare_equal_dim(self):
        img = self._create_random_image(height=100, width=100)
        padded_img = PadToSquare()(img)
        # create expected padded image
        # height = width. Nothing will be padded
        exp_img = img
        # verify expected image
        np.testing.assert_equal(np.array(exp_img), np.array(padded_img))

    def test_PadToSquare_largerheight_odd(self):
        img = self._create_random_image(height=121, width=100)
        padded_img = PadToSquare()(img)
        # create expected padded image
        # height is larger, so left and right will be padded. Right will be padded 1 pixel more
        left, top, right, bottom = 10, 0, 11, 0
        exp_img = ImageOps.expand(img, border=(left, top, right, bottom), fill=(255,255,255))
        # verify expected image
        np.testing.assert_equal(np.array(exp_img), np.array(padded_img))

    def test_PadToSquare_fill_value_single(self):
        img = self._create_random_image(height=120, width=100)
        # provide a scalar '0' as a fill value
        padded_img = PadToSquare(fill=0)(img)
        # create expected padded image
        # height is larger, so left and right will be padded
        left, top, right, bottom = 10, 0, 10, 0
        exp_img = ImageOps.expand(img, border=(left, top, right, bottom), fill=(0,0,0))
        # verify expected image shape
        np.testing.assert_equal(np.array(exp_img), np.array(padded_img))
        # verify fill values
        exp_img_arr = np.array(exp_img)
        self.assertTrue(np.all(exp_img_arr[:, 0:10] == 0))
        self.assertTrue(np.all(exp_img_arr[:, -10:] == 0))

    def test_PadToSquare_fill_value_tuple(self):
        img = self._create_random_image(height=120, width=100)
        fill = (100, 150, 200)
        padded_img = PadToSquare(fill=fill)(img)
        # create expected padded image
        # height is larger, so left and right will be padded
        left, top, right, bottom = 10, 0, 10, 0
        exp_img = ImageOps.expand(img, border=(left, top, right, bottom), fill=fill)
        # verify expected image shape
        np.testing.assert_equal(np.array(exp_img), np.array(padded_img))
        # verify fill values
        exp_img_arr = np.array(exp_img)
        self.assertTrue(np.all(exp_img_arr[:, 0:10] == np.array(fill)))
        self.assertTrue(np.all(exp_img_arr[:, -10:] == np.array(fill)))


if __name__ == "__main__":
    unittest.main()
