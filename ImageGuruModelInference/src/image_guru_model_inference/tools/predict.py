import argparse
import json
import os
import sys

from image_guru_model_inference.models.pytorch_predictor import PyTorchPredictor


def predict(classes, model_path, samples, model_type='resnet50', device=None, multi_label=False,
            multi_multi_class=False,
            multi_label_pred_thresholds=0.5,
            batch_size=32, num_workers=0,
            image_download_dir=None):
    """
    :param classes: List of classes. A simple list for multi_class and multi_label models. A list of lists for a
                    multi_multi_class model.
    :param model_path: Path to the model file.
    :param samples: List of samples, where each sample is an image physical_id.
    :param model_type: One of the pre-defined model types. Currently supported: ['resnet50']
    :param device: Torch device to use. Either 'cpu', GPU ID (int), or a torch.device instance
    :param multi_label: The provided model is used as a multi-label model if this param is True.
    :param multi_multi_class: The provided model is used as a multiple multi-class model if this param is True.
    :param multi_label_pred_thresholds: Prediction thresholds to use in case of a multi-label model. If a single
                                        is provided, then the same threshold is used for all classes. If a list
                                        is provided, then the threshold for classes[i] is multi_label_pred_thresholds[i]
    :param batch_size: Batch size
    :param num_workers: Number of workers for the pytorch dataloader.
    :param image_download_dir: Image download directory.
                               If not provided, downloaded images will be used but not saved.

    """
    predictor = PyTorchPredictor(
        model_path, classes, model_type=model_type, device=device, transforms='default',
        multi_label=multi_label, multi_multi_class=multi_multi_class,
        multi_label_pred_thresholds=multi_label_pred_thresholds,
        image_download_dir=image_download_dir
    )

    return predictor.predict(
        samples=samples, sample_type='physical_id', batch_size=batch_size, num_workers=num_workers)


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


def write_output(filepath, input_data, output_classes, output_probabilities, metadata):
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w') as fd:
        for i, inp_data in enumerate(input_data):
            preds = ','.join(output_classes[i])
            probs = ','.join([str(x) for x in output_probabilities[i]])
            fd.write('{}\t{}\t{}\n'.format(inp_data[1], preds, probs))


if __name__ == "__main__":
    # Command-line args
    arg_parser = argparse.ArgumentParser(description="Run training on provided annotations")
    arg_parser.add_argument('--input', default=None, type=str, required=True,
                            help='Input file with list of physical IDs. Input columns are copied as is to output'
                                 '<physicalId><TAB><other columns)>')
    arg_parser.add_argument('--output', default=None, type=str, required=True,
                            help='Output file path')
    arg_parser.add_argument('--model-path', default=None, type=str, required=True,
                            help='Model file path')
    arg_parser.add_argument('--config', default=None, type=str, required=False,
                            help='If provided, read the following params from config instead of command line: '
                                 '[classes, model-type, multi_label, multi_multi_class, multi_label_pred_thresholds]')
    arg_parser.add_argument('--classes', default=None, type=str, required=False, nargs='+',
                            help='File(s) containing a list of classes. Provide multiple files for '
                                 'multi-multi-class setup. The class files should be provided in the same order as '
                                 'they are expected in the model output. Required if --config is not specified.')
    arg_parser.add_argument('--model-type', default='resnet50', type=str, required=False,
                            help='Model type. Required if --config is not specified.')
    arg_parser.add_argument('--device', default=None, required=False, type=int)
    arg_parser.add_argument('--multi_label', default=False, action='store_true',
                            help='True if multi-label classification used be used. '
                                 'Required if --config is not specified.')
    arg_parser.add_argument('--multi_multi_class', default=False, action='store_true',
                            help='Multiple multi-class labeling. Each provided class file corresponds to a '
                                 'single multi-class labeling. Required if --config is not specified.')
    arg_parser.add_argument('--batch_size', default=32, required=False, type=int)
    arg_parser.add_argument('--num_workers', default=0, required=False, type=int)
    arg_parser.add_argument('--multi_label_pred_thresholds', default=0.5, required=False, type=float,
                            help='Confidence threshold for multi label classification. '
                                 'Required if --config is not specified.')
    arg_parser.add_argument('--image_download_dir', default=None, required=False, type=str)

    args = vars(arg_parser.parse_args())
    sys.stderr.write(str(args) + '\n')

    # first attempt to fetch model params from the config file
    if args['config']:
        with open(args['config']) as fd:
            config = json.load(fd)
        classes = config['params']['classes']
        model_type = config['params']['model_type']
        multi_label = config['params']['multi_label']
        multi_multi_class = config['params']['multi_multi_class']
        multi_label_pred_thresholds = config['params']['multi_label_pred_thresholds']

    else:
        # config file unavailable, try to fetch params from command line
        print('# class files: {}'.format(len(args['classes'])))
        if len(args['classes']) > 1:
            assert args['multi_multi_class'] and not args['multi_label'], 'Multiple class files only allowed for a ' \
                                                                          'multiple multi-class setup.'
        # Read class-groups.
        # Each class-group corresponds to a single multi-class setup.
        classes = []
        for fl_classes in args['classes']:
            with open(fl_classes) as fd:
                classes.append(fd.read().splitlines())

        model_type = args['model_type']
        multi_label = args['multi_label']
        multi_multi_class = args['multi_multi_class']
        multi_label_pred_thresholds = args['multi_label_pred_thresholds']
        if not multi_multi_class:
            classes = classes[0]

    print('classes: {}'.format(classes))

    input_data = read_input(args['input'])
    print('input samples: {}'.format(len(input_data)))
    samples = [x[0] for x in input_data]

    if args['device'] is None:
        device = 'cpu'
    else:
        device = args['device']

    output_classes, output_probabilities, metadata = predict(
        classes, args['model_path'], samples,
        model_type=model_type, device=device, multi_label=multi_label,
        multi_multi_class=multi_multi_class, multi_label_pred_thresholds=multi_label_pred_thresholds,
        batch_size=args['batch_size'], num_workers=args['num_workers'],
        image_download_dir=args['image_download_dir']
    )

    print('writing output to file: {} ...'.format(args['output']))
    write_output(args['output'], input_data, output_classes, output_probabilities, metadata)

    sys.stderr.write('script finished\n')
