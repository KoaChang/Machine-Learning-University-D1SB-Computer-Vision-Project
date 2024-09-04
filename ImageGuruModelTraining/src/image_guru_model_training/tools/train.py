import argparse
import os
import sys

from image_guru_model_training.models.pytorch_train import PyTorchTrain


def train(classes, train_annotations, val_annotations, device=None, model_dir=None,
          multi_label=False, multi_multi_class=False,
          multi_label_pred_thresholds=0.5, multi_labels_for_multi_class='ignore',
          batch_size=32, num_workers=0, start_lr=0.001, num_epochs=20, image_download_dir=None,
          load_model_path=None, load_model_n_classes=None, keep_last_layer_weights=False):
    """
    Trains a model

    :param classes: List of classes
    :param train_annotations: Training annotations as a list of (physical_id, label(s))
    :param val_annotations: Validation annotations as a list of (physical_id, label(s))
    :param device: Torch device to use. Either 'cpu', GPU ID (int), or a torch.device instance
    :param model_dir: Dir to save generated model files
    :param multi_label: A multi-label model is trained if this param is True.
    :param multi_multi_class: A multiple multi-class model is trained if this param is True.
    :param multi_label_pred_thresholds: Prediction thresholds to use in case of a multi-label model. If a single
                                            is provided, then the same threshold is used for all classes. If a list
                                            is provided, then the threshold for classes[i] is multi_label_pred_thresholds[i]
    :param multi_labels_for_multi_class: Used in case a multi-class model is to be trained but multiple labels are
                                         found for an instance. If 'ignore', such instances are ignored. If 'split',
                                         an instance having k labels is split into k instances, one per label.
    :param batch_size: Batch size
    :param num_workers: Number of workers for the pytorch dataloader.
    :param start_lr: Starting learning rate.
    :param num_epochs: Number of training epochs.
    :param image_download_dir: Image download directory. If not provided, downloaded images will be used but not saved.
    :param load_model_path: Load initial weights from this model file.
    :param load_model_n_classes: Number of classes in the model specified by the load_model_path. Needed to
                                 create a matching network architecture before loading the weights.
    :param keep_last_layer_weights: Whether to keep the last layer weights as well in the initial weights.
                                    Useful when the model training should be resumes from an earlier checkpoint.
    :return: A trained model as a PyTorchTrain instance.
    """

    trainer = PyTorchTrain(
        classes[0] if not multi_multi_class else classes,
        model_type='resnet50', device=device, train_transforms='default', use_train_data_aug=True,
        val_transforms='default',
        multi_label=multi_label, multi_multi_class=multi_multi_class,
        multi_label_pred_thresholds=multi_label_pred_thresholds,
        load_model_path=load_model_path, load_model_n_classes=load_model_n_classes, keep_last_layer_weights=keep_last_layer_weights,
        save_model_dir=model_dir, save_model_prefix='model-', image_download_dir=image_download_dir,
    )

    i = 0
    class_id = {}
    for class_group in classes:
        for c in class_group:
            class_id[c] = i
            i += 1

    annotations = {'train': train_annotations, 'val': val_annotations}
    samples = {k: [] for k in annotations}
    labels = {k: [] for k in annotations}

    if multi_label:
        # for multi-labels, create a binary label vector per instance.
        for k, anns in annotations.items():
            for phy, label in anns:
                samples[k].append(phy)
                vec = [0] * len(classes[0])
                for l in label:
                    vec[class_id[l]] = 1
                labels[k].append(vec)
    elif multi_multi_class:
        # for multi-multi class, take one label per class group.
        for k, anns in annotations.items():
            for phy, label in anns:
                classes_found = set()
                for l in label:
                    classes_found.add(class_id[l])
                # check if there one and only one class per group.
                i = 0
                for class_group in classes:
                    class_group_ids = set(range(i, i+len(class_group)))
                    _intersection = classes_found & class_group_ids
                    assert _intersection, 'No classes found in class group {} in annotation {}: {}: {}'\
                        .format(class_group, k, phy, label)
                    assert len(_intersection) == 1, 'More than one classes found in class group {} in annotation {}: {}: {}'\
                        .format(class_group, k, phy, label)
                    i += len(class_group)
                # checks passed. Add to the dataset
                samples[k].append(phy)
                labels[k].append(list(sorted(classes_found)))
    else:
        # for multi-class, take the instance as is if it has only 1 label. If it has multiple labels (k), skip it if
        # multi_labels_for_multi_class=='ignore', otherwise split it into k instances, one per label.
        for k, anns in annotations.items():
            for phy, label in anns:
                if len(label) == 1 or multi_labels_for_multi_class == 'split':
                    for l in label:
                        samples[k].append(phy)
                        labels[k].append(class_id[l])

    trainer.train(
        train_samples=samples['train'],
        train_labels=labels['train'],
        val_samples=samples['val'],
        val_labels=labels['val'],
        sample_type='physical_id',
        batch_size=batch_size,
        num_workers=num_workers,
        start_lr=start_lr,
        momentum=0.9,
        num_epochs=num_epochs
    )

    return trainer


def read_annotations(filepath):
    """
    Read annotations from file. Input file should be in TSV format with at least two columns: <physicalId><TAB><Comma-sep Label(s)>

    :param filepath: Path to the input file
    :return: Annotations as a list of (physical_id, label(s))
    """
    annotations = []
    if not filepath:
        return annotations
    try:
        print('reading annotations from file: {} ...'.format(filepath))
        with open(filepath) as fd:
            lines = fd.read().splitlines()
    except FileNotFoundError:
        return annotations
    for line in lines:
        fields = line.split('\t')
        phy = fields[0]
        c = list(set(fields[1].split(',')))
        annotations.append((phy, c))
    return annotations


if __name__ == "__main__":
    # Command-line args
    arg_parser = argparse.ArgumentParser(description="Run training on provided annotations")
    arg_parser.add_argument('--train-annotations', default=None, type=str, required=True, nargs='+',
                            help='Input file with list of training image annotations. '
                                 '2 column TSV: <physicalId><TAB><Comma-sep Label(s)>')
    arg_parser.add_argument('--val-annotations', default=None, type=str, required=True, nargs='+',
                            help='Input file with list of validation image annotations. '
                                 '2 column TSV: <physicalId><TAB><Comma-sep Label(s)>')

    arg_parser.add_argument('--classes', default=None, type=str, required=True, nargs='+',
                            help='File(s) containing a list of classes. Provide multiple files for '
                                 'multi-multi-class setup. The class files should be provided in the same order as '
                                 'they are expected in the model output.')
    arg_parser.add_argument('--model-dir', default=None, type=str, required=True,
                            help='Output dir to store model files')
    arg_parser.add_argument('--device', default=None, required=False, type=int)
    arg_parser.add_argument('--multi_label', default=False, action='store_true')
    arg_parser.add_argument('--multi_multi_class', default=False, action='store_true',
                            help='Multiple multi-class labeling. Each provided class file corresponds to a '
                                 'single multi-class labeling.')
    arg_parser.add_argument('--multi_labels_for_multi_class', default='ignore', type=str, required=False,
                            help="Whether to 'ignore' or 'split' multiple label samples in case of multi-class training")
    arg_parser.add_argument('--batch_size', default=32, required=False, type=int)
    arg_parser.add_argument('--num_workers', default=0, required=False, type=int)
    arg_parser.add_argument('--start_lr', default=0.001, required=False, type=float)
    arg_parser.add_argument('--num_epochs', default=20, required=False, type=int)
    arg_parser.add_argument('--image_download_dir', default=None, required=False, type=str)
    arg_parser.add_argument('--load_model_path', default=None, required=False, type=str,
                            help='Load initial weights from this model file.')
    arg_parser.add_argument('--load_model_n_classes', default=None, required=False, type=int,
                            help='Number of classes in the model specified by the load_model_path. Needed to create a '
                                 'matching network architecture before loading the weights.')
    arg_parser.add_argument('--keep_last_layer_weights', default=False, required=False, action='store_true',
                            help='Whether to keep the last layer weights as well in the initial weights. '
                                 'Useful when the model training should be resumes from an earlier checkpoint. '
                                 'default=False')

    args = vars(arg_parser.parse_args())
    sys.stderr.write(str(args) + '\n')

    print('# class files: {}'.format(len(args['classes'])))
    if len(args['classes']) > 1:
        assert args['multi_multi_class'] and not args['multi_label'], 'Multiple class files only allowed for a ' \
                                                                      'multiple multi-class setup.'
    classes = []
    for fl_classes in args['classes']:
        with open(fl_classes) as fd:
            classes.append(fd.read().splitlines())
    print('read a list of {} classes: {}'.format(len(classes), classes))

    train_annotations = []
    for fl in args['train_annotations']:
        train_annotations += read_annotations(fl)
    print('annotated training images: {}'.format(len(train_annotations)))

    val_annotations = []
    for fl in args['val_annotations']:
        val_annotations += read_annotations(fl)
    print('annotated val images: {}'.format(len(val_annotations)))

    if args['device'] is None:
        device = 'cpu'
    else:
        device = args['device']

    train(classes, train_annotations, val_annotations, device=device, model_dir=args['model_dir'],
          multi_label=args['multi_label'],
          multi_multi_class=args['multi_multi_class'],
          multi_label_pred_thresholds=0.5,
          multi_labels_for_multi_class=args['multi_labels_for_multi_class'],
          batch_size=args['batch_size'], num_workers=args['num_workers'],
          start_lr=args['start_lr'], num_epochs=args['num_epochs'],
          image_download_dir=args['image_download_dir'],
          load_model_path=args['load_model_path'], load_model_n_classes=args['load_model_n_classes'],
          keep_last_layer_weights=args['keep_last_layer_weights'])

    sys.stderr.write('script finished\n')
