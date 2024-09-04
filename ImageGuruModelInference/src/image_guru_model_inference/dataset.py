from io import BytesIO
import os
from PIL import Image
import requests
from torch.utils.data import Dataset
import traceback


class ImageDataset(Dataset):
    """
    Image Dataset to support input samples as a list of images.
    """

    def __init__(self, samples, transforms=None, sample_type='physical_id', image_download_dir=None, phy_style=None):
        """
        :param samples: List of samples, where each sample is either an image physical_id or URL.
        :param transforms: Transformation to be applied to every sample
        :param sample_type: Sample type, should be one of ['physical_id', 'url', 'file', 'pil', 'bytes', 'numpy'].
        :param image_download_dir: Image download directory.
                                   If not provided, downloaded images will be used but not saved.
                                   Used when sample_type is in ['physical_id', 'url'].
        :param phy_style: Used for image download when sample_type='physical_id'.
                          It should be one of the image styles supported by Media Services.
                          If None, no style is applied.
                          A style 'SL256' resizes the longer side to 256, maintaining the aspect ratio. If the
                          longer side is small than 256, it is not enlarged.
                          https://w.amazon.com/bin/view/MSA/HowTo/ImageStyleCodes
        """
        self.samples = samples
        self.transforms = transforms
        sample_type = sample_type.lower()
        assert sample_type in ['physical_id', 'url', 'file', 'pil', 'bytes', 'numpy'], \
            'Unsupported sample_type: {}'.format(sample_type)
        self.sample_type = sample_type
        self.image_download_dir = image_download_dir
        self.phy_style = phy_style

    def __getitem__(self, idx):
        try:
            img = self._read_image_from_sample(self.samples[idx])
        except:
            print('Exception in reading image. Returning None output.')
            print(traceback.format_exc())
            img = None
        if img is not None and self.transforms is not None:
            img = self.transforms(img)
        return idx, img

    def _read_image_from_sample(self, sample):
        """
        Reads an image from the given sample according to the sample_type param.

        :param sample: Input sample
        :return: PIL Image
        """
        if self.sample_type == 'pil':
            img = sample
        elif self.sample_type == 'numpy':
            img = Image.fromarray(sample.astype('uint8'))
        elif self.sample_type == 'bytes':
            img = Image.open(BytesIO(sample))
        elif self.sample_type == 'file':
            img = Image.open(sample)
        else:
            if self.sample_type == 'physical_id':
                img_url = self.physical_id_to_image_url(sample)
            else:
                img_url = sample
            # check if image is already present on the filesystem
            if self.image_download_dir:
                filepath = self.image_url_to_filepath(img_url)
                try:
                    img = Image.open(filepath)
                except:
                    # download the image
                    response = requests.get(img_url)
                    # save to file
                    with open(filepath, 'wb') as fd:
                        fd.write(response.content)
                    img = Image.open(BytesIO(response.content))
            else:
                # no download dir. Read the image in memory without saving to file.
                response = requests.get(img_url)
                img = Image.open(BytesIO(response.content))

        return img.convert("RGB")

    def __len__(self):
        return len(self.samples)

    def physical_id_to_filename(self, phy):
        """
        Converts an image physical ID to a filename.

        :param phy: Physical ID (str)
        :return: Filename
        """
        if self.phy_style is None:
            style_str = ''
        else:
            style_str = '._{}_'.format(self.phy_style)
        return '{}{}.jpg'.format(phy, style_str)

    def physical_id_to_image_url(self, phy):
        """
        Converts an image physical ID to a URL.

        :param phy: Physical ID (str)
        :return: Image URL
        """
        filename = self.physical_id_to_filename(phy)
        return 'https://m.media-amazon.com/images/I/{}'.format(filename)

    def image_url_to_filepath(self, url):
        """
        Converts an image URL to a filepath.

        :param url: URL (str)
        :return: File path
        """
        assert self.image_download_dir is not None, 'image_download_dir must be provided to get a file path'
        filename = os.path.basename(url)
        return os.path.join(self.image_download_dir, filename)
