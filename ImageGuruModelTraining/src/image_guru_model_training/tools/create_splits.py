import argparse
from collections import defaultdict
from itertools import chain
import os
import random

from image_guru_model_training.tools.train import read_annotations


def split_annotations(annotations, train_frac=0.7, val_frac=0.15, output_dir=None):
    annotations_per_phy = defaultdict(list)
    for phy, c in annotations:
        annotations_per_phy[phy].append((phy, c))

    phys = list(annotations_per_phy.keys())
    n_train = int(train_frac * len(phys))
    n_val = int(val_frac * len(phys))
    random.shuffle(phys)
    split_phys = {
        'train': phys[:n_train],
        'val': phys[n_train:n_train+n_val],
        'test': phys[n_train+n_val:]
    }

    splits = {
        partition: list(chain.from_iterable([annotations_per_phy[phy] for phy in partition_phys]))
        for partition, partition_phys in split_phys.items()
    }

    if output_dir:
        print('splits:')
        for partition, anns in splits.items():
            print('{}: {}'.format(partition, len(anns)))
            fl_output = os.path.join(output_dir, '{}.txt'.format(partition))
            print('saving {} {} annotations to file: {}'.format(len(anns), partition, fl_output))
            with open(fl_output, 'w') as fd:
                for phy, c in anns:
                    # c is a list of labels
                    fd.write('{}\t{}\n'.format(phy, ','.join(c)))

    return splits


if __name__ == "__main__":
    # Command-line args
    arg_parser = argparse.ArgumentParser(description="Create splits")
    arg_parser.add_argument('--annotations', default=None, type=str, required=True, nargs='+',
                            help='Input files with list of image annotations. '
                                 '2 column TSV: <physicalId><TAB><Comma-sep Label(s)>')
    arg_parser.add_argument('--train-frac', default=0.7, type=float, required=False,
                            help='Training fraction')
    arg_parser.add_argument('--val-frac', default=0.15, type=float, required=False,
                            help='Validation fraction')
    arg_parser.add_argument('--output-dir', default=None, type=str, required=True,
                            help='Output dir for saving split files')

    args = vars(arg_parser.parse_args())
    print(str(args) + '\n')

    annotations = []
    for fl in args['annotations']:
        annotations += read_annotations(fl)
    print('#annotations: {}'.format(len(annotations)))

    output_dir = args['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    split_annotations(annotations,
                      train_frac=args['train_frac'],
                      val_frac=args['val_frac'],
                      output_dir=output_dir)
