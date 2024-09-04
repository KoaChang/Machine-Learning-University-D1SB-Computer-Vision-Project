from collections import Iterable

import torch
import torchvision
from tqdm import tqdm

from image_guru_model_inference.base import PredictorBase
from image_guru_model_inference.dataset import ImageDataset
from image_guru_model_inference.transforms import PadToSquare


class PyTorchPredictor(PredictorBase):
    """
    Predictor class supporting multi-label, multi-class, and multiple multi-class models with pytorch models
    as the base networks.
    Multi-label: Any combination of labels could be true.
    Multi-class: One and only one label can be true.
    Multiple multi-class: A group of multi-class labels. In each group, one and only one label can be true.
    """

    def __init__(self, model_path, classes, model_type='resnet50', device=None, transforms='default',
                 multi_label=False, multi_multi_class=False, multi_label_pred_thresholds=0.5, image_download_dir=None):
        """
        :param model_path: Path to the model file.
        :param classes: List of classes. A simple list for multi_class and multi_label models. A list of lists for a
                        multi_multi_class model.
        :param model_type: One of the pre-defined model types. Currently supported: ['resnet50']
        :param device: Torch device to use. Either 'cpu', GPU ID (int), or a torch.device instance
        :param transforms: Transformation to be applied to every sample. If 'default', default transformations are used.
        :param multi_label: The provided model is used as a multi-label model if this param is True.
        :param multi_multi_class: The provided model is used as a multiple multi-class model if this param is True.
        :param multi_label_pred_thresholds: Prediction thresholds to use in case of a multi-label model. If a single
                                            is provided, then the same threshold is used for all classes. If a list
                                            is provided, then the threshold for classes[i] is multi_label_pred_thresholds[i]
        :param image_download_dir: Image download directory.
                                   If not provided, downloaded images will be used but not saved.
                                   Used when sample_type is in ['physical_id', 'url'].
        """
        # classes is a simple list for multi_class and multi_label models. A list of lists for a multi_multi_class model.
        self.classes = classes
        n_classes = len(classes) if not multi_multi_class else sum(len(cs) for cs in classes)

        # create a pytorch device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == 'cpu':
            self.device = torch.device('cpu')
        elif isinstance(device, int):
            self.device = torch.device("cuda:{}".format(device))
        else:
            self.device = device

        print('using device: {}'.format(self.device))

        # create a model using the provided params and weights
        self.model = self.get_model(model_type, n_classes, model_path)

        # create default transformations if not provided by the user
        if transforms == 'default':
            resize = 224
            self.transform = torchvision.transforms.Compose([
                PadToSquare(fill=(255, 255, 255)),
                torchvision.transforms.Resize(resize),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms

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

        self.image_download_dir = image_download_dir

    def get_model(self, model_type, n_classes, model_path):
        """
        Creates a model from the given params.

        :param model_type: Type of the model. Should be one of ['resnet50'].
        :param n_classes: Number of classes.
        :param model_path: Path to the model file.
        """
        assert model_type in ['resnet50', 'clip_resnet50_finetuned', 'vit-mae'], 'Unsupported model type: {}'.format(model_type)
        ## The output of the following loop should be a model with graph and the weights loaded
        if model_type == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False)
            n_features = model.fc.in_features
            model.fc = torch.nn.Linear(n_features, n_classes)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif model_type == 'clip_resnet50_finetuned':
            model = torch.load(model_path, map_location = self.device) # the model saved using torch.save
        elif model_type == 'vit-mae':
            from image_guru_model_inference.models import models_vit
            model = models_vit.__dict__['vit_large_patch16'](
                            num_classes=n_classes,
                            drop_path_rate=0.2,
                            global_pool= True,
                            )
            checkpoint = torch.load(model_path, map_location='cpu')

            msg = model.load_state_dict(checkpoint, strict=False)
            print(msg)
        else:
            model = None
        
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, dataset=None, samples=None, sample_type='physical_id', batch_size=32, num_workers=0):
        """
        Run prediction on the provided dataset or samples. Either samples or dataset must be provided.

        :param dataset: ImageDataset instance. Either this or samples must be provided.
        :param samples: List of samples, where each sample is either an image physical_id or URL.
                        Either samples or dataset must be provided.
        :param sample_type: Sample type, used in case samples is provided.
                            Should be one of ['physical_id', 'url', 'file', 'pil', 'bytes', 'numpy'].
        :param batch_size: Batch size
        :param num_workers: Number of workers for the pytorch dataloader.
        :return: A tuple of (output_classes, output_probabilities, metadata).
                output_classes is a list of lists.
                Outer list is over the samples.
                Inner list, corresponding to a sample, contains a list of predicted classes.
                output_probabilities is a list of lists.
                Outer list is over the samples.
                Inner list, corresponding to a sample, contains a list of probabilities corresponding to the predicted classes.
                metadata is a dict containing any metadata.
        :rtype: tuple
        """
        assert dataset or samples, 'Either samples or dataset must be provided'
        if not dataset:
            # create a dataset from the provided samples
            dataset = ImageDataset(samples, transforms=self.transform, sample_type=sample_type,
                                    image_download_dir=self.image_download_dir)

        # override the default collate function to ignore samples where the img is None.
        # this can happen in case of corrupted or missing images, which makes the dataset return a 'None' image.
        default_collate = torch.utils.data.dataloader.default_collate

        def custom_collate(batch):
            # create a subset where the image (item[1]) is not None
            batch = [item for item in batch if item[1] is not None]
            if batch:
                # call the pytorch default_collate only if the subset contains at least 1 item
                return default_collate(batch)
            # return None if the batch is empty
            return None

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                 collate_fn=custom_collate)

        # Lists to collect class predictions and probabilities
        prediction_indices = []
        predictions = []
        prediction_probabilities = []
        self.model.eval()
        # Iterate over data.
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if batch is None:
                    # empty batch
                    continue
                indices, images = batch
    
                images = images.to(self.device)
                outputs = self.model(images)
    
                if self.multi_label:
                    # apply sigmoid to each output and then use the provided threshold to get the final predictions.
                    probs = torch.sigmoid(outputs)
                    preds = probs >= self.multi_label_pred_thresholds
                elif self.multi_multi_class:
                    # Apply softmax to each multi_class group. In each group, predict the max probability class.
                    class_offset = 0
                    preds, probs = None, None
                    for class_group in self.classes:
                        # extract the output for the current class group
                        group_output = outputs[:, class_offset:class_offset + len(class_group)]
                        # get predicted class for the current group.
                        _, _preds = torch.max(group_output, dim=1, keepdim=True)
                        # concatenate the predicted class for the current group to the predicted classes of previous groups.
                        preds = torch.cat([preds, _preds], dim=1) if preds is not None else _preds
                        # get prediction probabilities for each class in the current group.
                        _probs = torch.nn.functional.softmax(group_output, dim=1)
                        # concatenate the prediction probabilities of the current group to the probabilities computed for previous groups.
                        probs = torch.cat([probs, _probs], dim=1) if probs is not None else _probs
                        # update the class offset to move to the next class group.
                        class_offset += len(class_group)
                else:
                    # Predict the class with the max output value
                    _, preds = torch.max(outputs, dim=1)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
    
                prediction_indices += indices.detach().cpu().numpy().tolist()
                # predictions contains 0/1 for each class in case of multi-label, the class ID in case of multi-class,
                # and list of class IDs in case of multi-multi-class
                predictions += preds.detach().cpu().numpy().tolist()
                # prediction_probabilities contains the probabilities of all the classes.
                prediction_probabilities += probs.detach().cpu().numpy().tolist()

        # Lists to collect class predictions and probabilities of only the predicted classes.
        # TODO: change predictions and prediction_probabilities above so that we don't need to distinguish the two cases here.
        # initialize the output lists. For samples which had no predictions (could happen in case of image not loading),
        # the output will be an empty list.
        output_classes = [[] for _ in range(len(dataset))]
        output_probabilities = [[] for _ in range(len(dataset))]
        if self.multi_label:
            # In case of multi-label predictions, prediction for a single sample will be a list of predicted classes,
            # and probabilities will be a list of probabilities corresponding to the predicted classes.
            for sample_id, preds, probs in zip(prediction_indices, predictions, prediction_probabilities):
                # loop through the classes and collect class_names and probabilities wherever the prediction is 1.
                for c in range(len(self.classes)):
                    if preds[c] == 1:
                        output_classes[sample_id].append(self.classes[c])
                        output_probabilities[sample_id].append(probs[c])
        elif self.multi_multi_class:
            # In case of multi-multi-class predictions, prediction for a single sample will be a list of classes,
            # and probability will be a list of probabilities corresponding to the predicted class in each class group.
            for sample_id, preds, probs in zip(prediction_indices, predictions, prediction_probabilities):
                class_offset = 0
                # loop over class groups
                for i_c, class_group in enumerate(self.classes):
                    # extract predicted class and probabilities for the current group
                    c = preds[i_c]
                    output_classes[sample_id].append(class_group[c])
                    output_probabilities[sample_id].append(probs[class_offset+c])
                    class_offset += len(class_group)
        else:
            # In case of multi-class predictions, prediction for a single sample will be a single class,
            # and probability will be a single probability corresponding to the predicted class.
            # the output for each sample is still a list, but of length 1.
            for sample_id, preds, probs in zip(prediction_indices, predictions, prediction_probabilities):
                c = preds
                output_classes[sample_id].append(self.classes[c])
                output_probabilities[sample_id].append(probs[c])

        metadata = {
            'prediction_indices': prediction_indices,
            'prediction_probabilies': prediction_probabilities,  # probabilities of all the classes for all the samples.
            'classes': self.classes,  # list of all the classes
        }

        return output_classes, output_probabilities, metadata

    def predict_one(self, sample, sample_type='physical_id'):
        """
        Run prediction on the provided single sample.

        :param sample: Representation of the sample itself, e.g. a physical ID.
        :param sample_type: Should be one of ['physical_id', 'url', 'file', 'pil', 'bytes', 'numpy'].
        :return: A tuple of (probabilities, classes).
                Each is a list over all supported classes in the same order.
        :rtype: tuple
        """
        dataset = ImageDataset(
            [sample], transforms=self.transform, sample_type=sample_type,
            image_download_dir=self.image_download_dir)
        with torch.no_grad():
            _, image = dataset[0]
            outputs = self.model(image.to(self.device).unsqueeze(0))
            if self.multi_label:
                probs = torch.sigmoid(outputs)
            elif self.multi_multi_class:
                i_c = 0
                probs = None
                # loop over class groups
                for class_group in self.classes:
                    # extract probabilities for the current group
                    _probs = torch.nn.functional.softmax(outputs[:, i_c:i_c+len(class_group)], dim=1)
                    probs = torch.cat([probs, _probs], dim=1) if probs is not None else _probs
                    # update the offset to move to the next class group.
                    i_c += len(class_group)
            else:
                probs = torch.nn.functional.softmax(outputs, dim=1)
            probs = probs.squeeze(0).tolist()
        return probs, self.classes
