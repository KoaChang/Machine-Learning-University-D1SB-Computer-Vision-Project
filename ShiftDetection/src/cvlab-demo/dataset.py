import torch
import torchvision

from image_guru_model_inference.dataset import ImageDataset
from image_guru_model_inference.transforms import PadToSquare


def read_input(filepath):
    result = []
    if not filepath:
        return result
    try:
        print('reading input from file: {} ...'.format(filepath))
        with open(filepath) as fd:
            lines = fd.read().splitlines()
    except FileNotFoundError:
        return result
    for line in lines:
        fields = line.split('\t')
        phy = fields[0]
        result.append((phy, line))
    return result


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, filepath):
        input_data = read_input(filepath)
        print('input samples: {}'.format(len(input_data)))
        samples = [x[0] for x in input_data]
        transform = torchvision.transforms.Compose([
                PadToSquare(fill=(255, 255, 255)),
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
        self._dataset = ImageDataset(samples=samples, transforms=transform)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index][1], 0 # Return dummy label


class NoisyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: torch.utils.data.Dataset, std: float):
        self._dataset = dataset
        self._std = std

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        x, y = self._dataset[index]
        return x + self._std * torch.randn_like(x), y