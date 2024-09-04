from collections import Iterable
from datetime import datetime
import os
import sys
import torch
import torchvision
from tqdm import tqdm

from image_guru_model_training.dataset import ImageAndLabelDataset
from image_guru_model_inference.transforms import PadToSquare


class PyTorchTrain(object):
    """
    Training class supporting multi-label and multi-class models with pytorch models as the base networks.
    """

    def __init__(self, classes, model_type='resnet50', device=None, train_transforms='default', use_train_data_aug=True,
                 val_transforms='default', multi_label=False, multi_multi_class=False, multi_label_pred_thresholds=0.5,
                 load_model_path=None, load_model_n_classes=None, keep_last_layer_weights=True,
                 save_model_dir='./models', save_model_prefix='model-', image_download_dir=None):
        """
        :param classes: List of classes
        :param model_type: One of the pre-defined model types. Currently supported: ['resnet50']
        :param device: Torch device to use. Either 'cpu', GPU ID (int), or a torch.device instance
        :param train_transforms: Transformation to be applied to every sample in the training set.
                                 If 'default', default transformations are used.
        :param use_train_data_aug: If True, add data augmentation transforms to the default transforms while training.
        :param val_transforms: Transformation to be applied to every sample in the validation set.
                               If 'default', default transformations are used.
        :param multi_label: A multi-label model is trained if this param is True.
        :param multi_multi_class: A multiple multi-class model is trained if this param is True.
        :param multi_label_pred_thresholds: Prediction thresholds to use in case of a multi-label model. If a single
                                            is provided, then the same threshold is used for all classes. If a list
                                            is provided, then the threshold for classes[i] is multi_label_pred_thresholds[i]
        :param load_model_path: Load initial weights from this model file.
        :param load_model_n_classes: Number of classes in the model specified by the load_model_path. Needed to
                                     create a matching network architecture before loading the weights.
        :param keep_last_layer_weights: Whether to keep the last layer weights as well in the initial weights.
                                        Useful when the model training should be resumes from an earlier checkpoint.
        :param save_model_dir: Dir to save generated model files. Default: ./models
        :param save_model_prefix: Prefix to use in the model filenames.
        :param image_download_dir: Image download directory.
                                   If not provided, downloaded images will be used but not saved.
                                   Used when sample_type is in ['physical_id', 'url'].
        """
        self.classes = classes
        n_classes = len(classes) if not multi_multi_class else sum(len(cs) for cs in classes)

        # create a pytorch device
        self.device = self.get_torch_device(device)

        # create a model using the provided params and weights
        self.model = self.get_model(model_type, n_classes,
                                    load_model_path=load_model_path,
                                    load_model_n_classes=load_model_n_classes,
                                    keep_last_layer_weights=keep_last_layer_weights)

        # create default transformations if not provided by the user
        if train_transforms == 'default':
            self.train_transform = self.get_default_transformations(use_data_aug=use_train_data_aug)
        else:
            self.train_transform = train_transforms
        if val_transforms == 'default':
            self.val_transform = self.get_default_transformations(use_data_aug=False)
        else:
            self.val_transform = val_transforms

        # in case of multi-label predictions, we need to decide on the prediction thresholds
        # to be applied on the model outputs
        self.multi_label = multi_label
        self.multi_multi_class = multi_multi_class
        assert not(self.multi_label and self.multi_multi_class), 'Only one of multi_label or multi_multi_class can be True'
        if self.multi_label:
            if isinstance(multi_label_pred_thresholds, Iterable):
                # A list of thresholds is provided, one per class
                self.multi_label_pred_thresholds = torch.Tensor(multi_label_pred_thresholds)
            else:
                # If a single threshold is provided, use it for all the classes
                self.multi_label_pred_thresholds = torch.Tensor([multi_label_pred_thresholds]*n_classes)
            self.multi_label_pred_thresholds = self.multi_label_pred_thresholds.to(self.device)

        # loss function for training
        if multi_label:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif multi_multi_class:
            self.criterion = self.multi_multi_class_loss
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.save_model_dir = save_model_dir
        self.save_model_prefix = save_model_prefix
        self.image_download_dir = image_download_dir

    def get_torch_device(self, device=None):
        """
        Create a torch.device instance.

        :param device: Torch device to use. Either 'cpu', GPU ID (int), or a torch.device instance
        :return: Created torch.device
        """
        if device is None:
            # choose either GPU#0 or cpu based on cuda availability
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == 'cpu':
            # CPU
            return torch.device('cpu')
        elif isinstance(device, int):
            # GPU Id is provided. Create an appropriate device.
            return torch.device("cuda:{}".format(device))
        # assume that the input was already a torch device. Return as is.
        return device

    def get_default_transformations(self, use_data_aug=False):
        """
        Create default transformations for the input images.

        :param use_data_aug: If True, add data augmentation transforms to the default transforms
        :return: Created transformations.
        """
        resize = 224

        transform_list = [
            PadToSquare(fill=(255, 255, 255)),
            torchvision.transforms.Resize(resize)]

        if use_data_aug:
            transform_list += [
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                ], p=0.5),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                ], p=0.5)
            ]

        transform_list += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]

        return torchvision.transforms.Compose(transform_list)

    def get_model(self, model_type, n_classes, load_model_path=None, load_model_n_classes=None, keep_last_layer_weights=True):
        """
        Creates a model from the given params.

        :param model_type: Type of the model. Should be one of ['resnet50'].
        :param n_classes: Number of classes.
        :param load_model_path: Load initial weights from this model file.
        :param load_model_n_classes: Number of classes in the model specified by the load_model_path. Needed to
                                     create a matching network architecture before loading the weights.
        :param keep_last_layer_weights: Whether to keep the last layer weights as well in the initial weights.
                                        Useful when the model training should be resumes from an earlier checkpoint.

        """
        assert model_type in ['resnet50'], 'Unsupported model type: {}'.format(model_type)
        if model_type == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False if load_model_path else True)
        else:
            model = None
        n_features = model.fc.in_features

        if load_model_path:
            # load initial weights from a specified model file
            # first create a final layer corresponding to the load_model #classes.
            model.fc = torch.nn.Linear(n_features, load_model_n_classes)
            # load_model architecture is now created. Load the weights.
            print('loading model weights from file: {} ...'.format(load_model_path))
            model.load_state_dict(torch.load(load_model_path, map_location=self.device))
            # if n_classes are different compared to the load_model or the final layer weights are not supposed to be kept,
            # create a fresh layer with random weights.
            if (not keep_last_layer_weights) or n_classes != load_model_n_classes:
                model.fc = torch.nn.Linear(n_features, n_classes)
        else:
            # no previous model to start from. Create a last layer corresponding to the n_classes.
            model.fc = torch.nn.Linear(n_features, n_classes)

        model = model.to(self.device)
        return model

    def multi_multi_class_loss(self, outputs, target):
        total_loss = None
        class_offset = 0
        for i_c, class_group in enumerate(self.classes):
            group_output = outputs[:, class_offset:class_offset + len(class_group)]
            group_target = target[:, i_c] - class_offset
            loss = torch.nn.CrossEntropyLoss()(group_output, group_target)
            total_loss = loss if not total_loss else total_loss + loss
            class_offset += len(class_group)
        return total_loss

    def train(self, train_dataset=None, train_samples=None, train_labels=None, sample_type='physical_id',
              batch_size=32, num_workers=0, start_lr=0.001, momentum=0.9, num_epochs=20,
              val_dataset=None, val_samples=None, val_labels=None):
        """
        Run training on the provided dataset or samples. Either train_samples or train_dataset must be provided.

        :param train_dataset: ImageAndLabelDataset instance. Either this or train_samples must be provided.
        :param train_samples: List of samples, where each sample is either an image physical_id or URL.
                              Either train_samples or train_dataset must be provided.
        :param train_labels: List of training labels, where each label is either a
                                1. a single number in a multi-class scenario, or
                                2. a list of 0/1 in a multi-label scenario, with 1 indicating the presence of a class
                                    and 0 indicating otherwise.
        :param sample_type: Sample type, used in case samples is provided.
                            Should be one of ['physical_id', 'url', 'file', 'pil', 'bytes', 'numpy'].
        :param batch_size: Batch size
        :param num_workers: Number of workers for the pytorch dataloader.
        :param start_lr: Starting learning rate.
        :param momentum: momentum
        :param num_epochs: Number of training epochs.
        :param val_dataset: ImageAndLabelDataset instance.
        :param val_samples: List of validation samples, where each sample is either an image physical_id or URL.
        :param val_labels: List of validation labels, where each label is either a
                                1. a single number in a multi-class scenario, or
                                2. a list of 0/1 in a multi-label scenario, with 1 indicating the presence of a class
                                    and 0 indicating otherwise.
        """
        begin_train_time = datetime.now()

        assert train_dataset or (train_samples and train_labels), \
            'Either (train_samples and train_labels) or train_dataset must be provided'
        if not train_dataset:
            # create a training dataset from the provided samples
            train_dataset = ImageAndLabelDataset(train_samples, train_labels, transforms=self.train_transform,
                                                 sample_type=sample_type, image_download_dir=self.image_download_dir)

        # override the default collate function to ignore samples where the img is None.
        # this can happen in case of corrupted or missing images, which makes the dataset return a 'None' image.
        default_collate = torch.utils.data.dataloader.default_collate
        custom_collate = lambda batch: default_collate([item for item in batch if item[1] is not None])

        dataloader = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                           num_workers=num_workers, collate_fn=custom_collate)}
        phases = ['train']

        if (not val_dataset) and (val_samples and val_labels):
            # create a validation dataset from the provided samples
            val_dataset = ImageAndLabelDataset(val_samples, val_labels, transforms=self.val_transform,
                                               sample_type=sample_type, image_download_dir=self.image_download_dir)
        if val_dataset:
            dataloader['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                            num_workers=num_workers, collate_fn=custom_collate)
            phases.append('val')

        optimizer = torch.optim.SGD(self.model.parameters(), lr=start_lr, momentum=momentum)
        # Decay LR by a factor of gamma every step_size epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        best_acc = 0.0

        # directory to save model weights
        os.makedirs(self.save_model_dir, exist_ok=True)

        # run for num_epochs epochs
        for epoch in range(num_epochs):
            print('[{}] Epoch {}/{}'.format(datetime.now(), epoch, num_epochs - 1))
            print('-' * 40)

            # Each epoch has a training and validation phase
            for phase in phases:
                begin_phase_time = datetime.now()

                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                n_samples = 0
                for indices, images, labels in tqdm(dataloader[phase], file=sys.stdout):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    if self.multi_label:
                        # float labels needed for BCEWithLogitsLoss
                        labels = labels.float()
                    n_samples += len(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * images.size(0)

                    if self.multi_label:
                        probs = torch.sigmoid(outputs)
                        preds = probs >= self.multi_label_pred_thresholds
                        running_corrects += torch.sum(preds == labels.data) / labels.shape[1]
                    elif self.multi_multi_class:
                        class_offset = 0
                        for i_c, class_group in enumerate(self.classes):
                            _, preds = torch.max(outputs[:, class_offset:class_offset + len(class_group)], dim=1)
                            preds += class_offset
                            running_corrects += torch.sum(preds == labels[:, i_c].data) / len(self.classes)
                            class_offset += len(class_group)
                    else:
                        _, preds = torch.max(outputs, dim=1)
                        running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    lr_scheduler.step()

                epoch_loss = running_loss / n_samples
                epoch_acc = running_corrects.double().item() / n_samples
                time_elapsed = datetime.now() - begin_phase_time

                print('{} Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, Samples: {}, Time: {}'.format(
                    phase, epoch, epoch_loss, epoch_acc, n_samples, time_elapsed))

                # model weights could be saved either to a file tracking the best model weights, or to a file
                # tracking weights every 10 epochs, or both
                files_to_save = []
                if phase == 'val' and epoch_acc > best_acc:
                    # best model so far on the val set.
                    best_acc = epoch_acc
                    files_to_save.append(os.path.join(self.save_model_dir, '{}best.pth'.format(self.save_model_prefix)))

                if phase == 'train' and epoch % 10 == 0:
                    # save every 10 epochs.
                    files_to_save.append(os.path.join(self.save_model_dir, '{}epoch{}.pth'.format(self.save_model_prefix, epoch)))

                for fl_save in files_to_save:
                    print('saving model to {} ...'.format(fl_save))
                    state_dict = self.model.state_dict()
                    torch.save(state_dict, fl_save)

            print()

        time_elapsed = datetime.now() - begin_train_time
        print('Training complete in {}'.format(time_elapsed))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights from file
        fl_best_model = os.path.join(self.save_model_dir, '{}best.pth'.format(self.save_model_prefix))
        self.model.load_state_dict(torch.load(fl_best_model, map_location=self.device))
