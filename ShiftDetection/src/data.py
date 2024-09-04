import os
import pickle

import boto3
import datasets
import torch
import torchvision


DATASET_PATHNAMES = {
    "CIFAR10": "cifar-10-batches-py",
    "CIFAR100": "cifar-100-python",
    "ImageNet32": "imagenet32",
}

TEXT_DATASETS = ["ag_news", "tweet_eval__emoji", "yelp_review_full", "SetFit/20_newsgroups"]


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of an s3 folder.

    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: The folder whose contents do download.
        local_dir: Path where to download the folnder contents.
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if os.path.exists(target):
            continue
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target), exist_ok=True)
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)


def download_dataset(dataset: str, data_root_dir: str = "./data"):
    if dataset in TEXT_DATASETS:
        return
    dataset_pathname = DATASET_PATHNAMES[dataset]
    if os.path.exists(os.path.join(data_root_dir, dataset_pathname)):
        print("Dataset already exists.")
    else:
        download_s3_folder("mnemosyne-team-bucket",
                            os.path.join("dataset", dataset_pathname),
                            os.path.join(data_root_dir, dataset_pathname))


def get_dataset(dataset: str, data_root_dir: str = "./data", tokenizer=None):

    if dataset == "CIFAR100":
        download_dataset(dataset, data_root_dir)
        transform_train = torchvision.transforms.Compose(
            [
                # torchvision.transforms.RandomCrop(32, padding=4),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        ds_tr = torchvision.datasets.CIFAR100(
            root=data_root_dir, train=True, download=False, transform=transform_train)
        ds_te = torchvision.datasets.CIFAR100(
            root=data_root_dir, train=False, download=False, transform=transform_test)
        return ds_tr, ds_te

    elif dataset == "CIFAR10":
        download_dataset(dataset, data_root_dir)
        transform_train = torchvision.transforms.Compose(
            [
                # torchvision.transforms.RandomCrop(32, padding=4),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # TODO: FIX THIS!
            ]
        )
        transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        ds_tr = torchvision.datasets.CIFAR10(
            root=data_root_dir, train=True, download=False, transform=transform_train)
        ds_te = torchvision.datasets.CIFAR10(
            root=data_root_dir, train=False, download=False, transform=transform_test)
        return ds_tr, ds_te

    elif dataset == "ImageNet32":
        download_dataset(dataset, data_root_dir)
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        transform_train = torchvision.transforms.Compose(
            [
                # torchvision.transforms.RandomCrop(32, padding=4),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize
            ]
        )
        transform_test = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), normalize])
        ds_tr = PickleDataset(os.path.join(data_root_dir, "imagenet32", "trainset.pickle"),
                              transform=transform_train)
        ds_te = PickleDataset(os.path.join(data_root_dir, "imagenet32", "valset.pickle"),
                              transform=transform_test)
        return ds_tr, ds_te

    elif dataset in TEXT_DATASETS:
        assert tokenizer is not None
        if "__" in dataset:
            dataset, config = dataset.split("__")
        else:
            config = None
        ds_tr = datasets.load_dataset(dataset, config, split="train", cache_dir=data_root_dir)
        ds_te = datasets.load_dataset(dataset, config, split="test", cache_dir=data_root_dir)
        ds_tr = CompactTextDataset(ds_tr, tokenizer)
        ds_te = CompactTextDataset(ds_te, tokenizer)
        return ds_tr, ds_te


class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, fname, transform=None, target_transform=None) -> None:
        """Creates a PickleDataset instance. It is like a TensorDataset, but the input is a pickle file. s

        :param fname: python pickle file name
        :type fname: str
        :param transform: Image transforms to be applied, defaults to None. It should at least convert PIL Images to torch.Tensor.
        :type transform: Callable, optional
        :param target_transform: Transforms to the labels, if the transforms to the images modify anything, defaults to None
        :type target_transform: Callable, optional
        """
        with open(fname, "rb") as fp:
            self.all_train = pickle.load(fp)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample, target = self.all_train[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.all_train)


class CompactTextDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenizer):
        self._dataset = dataset.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128), batched=True)
        self._dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        elt = self._dataset[idx]
        x = torch.stack([elt['input_ids'], elt['attention_mask']], dim=1)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        y = elt["label"]
        return x, y
