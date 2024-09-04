import torch

from image_guru_model_inference.dataset import ImageDataset


class ImageAndLabelDataset(ImageDataset):
    """
    Image Dataset to support input samples as a list of images and their labels.
    """

    def __init__(self, samples, labels, transforms=None, sample_type='physical_id', image_download_dir=None):
        """
        :param samples: List of samples, where each sample is either an image physical_id or URL.
        :param labels: List of labels, where each label is either
                       1. a single number for the class ID in a multi-class scenario, or
                       2. a list of 0/1 in a multi-label scenario, with 1 indicating the presence of a class and 0 indicating otherwise.
        :param transforms: Transformation to be applied to every sample
        :param sample_type: Sample type, should be one of ['physical_id', 'url', 'file', 'pil', 'bytes', 'numpy'].
        :param image_download_dir: Image download directory.
                                   If not provided, downloaded images will be used but not saved.
                                   Used when sample_type is in ['physical_id', 'url'].
        """
        super().__init__(samples, transforms=transforms, sample_type=sample_type, image_download_dir=image_download_dir)
        self.labels = labels

    def __getitem__(self, idx):
        _, img = super().__getitem__(idx)
        label = torch.tensor(self.labels[idx])
        return idx, img, label
