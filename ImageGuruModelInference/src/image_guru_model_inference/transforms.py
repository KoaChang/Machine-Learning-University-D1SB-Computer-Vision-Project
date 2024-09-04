import torchvision


class PadToSquare(object):
    """
    Pads the image with a given RGB values to make it a square size image.
    """

    def __init__(self, fill=(255, 255, 255)):
        """
        :param fill: Either a single value or a tuple of (R,G,B) values to fill in the padded region.
                     If a single value is provided, then R=G=B=fill will be used.
        """
        self.fill = fill

    def __call__(self, img):
        """
        :param img: PIL Image to be padded.
        :return: Padded image.
        """
        h, w = img.height, img.width
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0
        if h > w:
            total_pad = h - w
            left_pad = int(total_pad / 2)
            right_pad = total_pad - left_pad
        elif w > h:
            total_pad = w - h
            top_pad = int(total_pad / 2)
            bottom_pad = total_pad - top_pad

        pad = (left_pad, top_pad, right_pad, bottom_pad)
        img = torchvision.transforms.functional.pad(img, pad, self.fill)
        assert img.height == img.width, 'h, w: {}, {}, pad: {}'.format(img.height, img.width, pad)
        return img
