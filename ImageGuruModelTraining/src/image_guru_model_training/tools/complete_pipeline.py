import argparse
import os
import sys
import torch

from image_guru_model_training.tools.create_splits import split_annotations
from image_guru_model_inference.tools.predict import predict
from image_guru_model_training.evaluate import evaluate_multi_class, evaluate_multi_label
from image_guru_model_training.tools.train import read_annotations, train


if __name__ == "__main__":
    # Command-line args
    arg_parser = argparse.ArgumentParser(description="Run training on provided annotations")
    arg_parser.add_argument('--annotations', default=None, type=str, required=True, nargs='+',
                            help='Input files with list of image annotations. '
                                 '2 column TSV: <physicalId><TAB><Comma-sep Label(s)>')
    arg_parser.add_argument('--classes', default=None, type=str, required=True, help='File containing a list of classes')
    arg_parser.add_argument('--model-dir', default=None, type=str, required=True,
                            help='Output dir to store model files')
    arg_parser.add_argument('--device', default=None, required=False, type=int)
    arg_parser.add_argument('--multi_label', default=False, action='store_true')
    arg_parser.add_argument('--multi_labels_for_multi_class', default='ignore', type=str, required=False,
                            help="Whether to 'ignore' or 'split' multiple label samples in case of multi-class training")
    arg_parser.add_argument('--batch_size', default=32, required=False, type=int)
    arg_parser.add_argument('--num_workers', default=0, required=False, type=int)
    arg_parser.add_argument('--start_lr', default=0.001, required=False, type=float)
    arg_parser.add_argument('--num_epochs', default=20, required=False, type=int)
    arg_parser.add_argument('--image_download_dir', default=None, required=False, type=str)
    arg_parser.add_argument('--output-dir', default=None, type=str, required=True,
                            help='Output dir for prediction and evaluation files')

    args = vars(arg_parser.parse_args())
    sys.stderr.write(str(args) + '\n')

    fl_classes = os.path.join(args['classes'])
    with open(fl_classes) as fd:
        classes = fd.read().splitlines()
    print('read a list of {} classes: {}'.format(len(classes), classes))

    annotations = []
    for fl in args['annotations']:
        annotations += read_annotations(fl)
    print('#annotations: {}'.format(len(annotations)))

    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print('creating splits ...')
    splits = split_annotations(annotations, train_frac=0.7, val_frac=0.15, output_dir=output_dir)

    if args['device'] is None:
        device = 'cpu'
    else:
        device = args['device']

    train(classes, splits['train'], splits['val'], device=device, model_dir=args['model_dir'],
          multi_label=args['multi_label'], multi_label_pred_thresholds=0.5,
          multi_labels_for_multi_class=args['multi_labels_for_multi_class'],
          batch_size=args['batch_size'], num_workers=args['num_workers'],
          start_lr=args['start_lr'], num_epochs=args['num_epochs'],
          image_download_dir=args['image_download_dir'])
    torch.cuda.empty_cache()
    model_path = os.path.join(args['model_dir'], 'model-best.pth')

    for partition, partition_annotations in splits.items():
        print('running prediction on {} set ...'.format(partition))
        phys = [phy for phy, c in partition_annotations]
        output_classes, output_probabilities, metadata = predict(
            classes, model_path, phys,
            model_type='resnet50', device=device, multi_label=args['multi_label'],
            multi_label_pred_thresholds=0.5,
            batch_size=args['batch_size'], num_workers=args['num_workers'],
            image_download_dir=args['image_download_dir']
        )
        fl_output = os.path.join(output_dir, '{}.predictions.txt'.format(partition))
        print('writing predictions to file: {} ...'.format(fl_output))
        with open(fl_output, 'w') as fd:
            for i in range(len(partition_annotations)):
                phy, c = partition_annotations[i]
                true_labels = ','.join(c)
                preds = ','.join(output_classes[i])
                probs = ','.join([str(x) for x in output_probabilities[i]])
                fd.write('{}\t{}\t{}\t{}\n'.format(phy, true_labels, preds, probs))

        print('running evalaution on {} set ...'.format(partition))
        if args['multi_label']:
            true_labels, pred_labels = [], []
            for i in range(len(partition_annotations)):
                phy, c = partition_annotations[i]
                true_labels.append(c)
                pred_labels.append(output_classes[i])
            evaluate_multi_label(true_labels, pred_labels, classes, output_dir, output_prefix='{}.'.format(partition))
        else:
            true_labels, pred_labels = [], []
            for i in range(len(partition_annotations)):
                phy, c = partition_annotations[i]
                if len(c) == 1 or args['multi_labels_for_multi_class'] == 'split':
                    true_labels += c
                    pred_labels += output_classes[i]
            evaluate_multi_class(true_labels, pred_labels, classes, output_dir, output_prefix='{}.'.format(partition))

    sys.stderr.write('script finished\n')
