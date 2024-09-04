import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, hamming_loss, multilabel_confusion_matrix


def evaluate_multi_class(true_labels, pred_labels, classes, output_dir, output_prefix=''):
    """
    Evaluate multi-class predictions. Computed accuracy, classification_report, and confusion matrix.

    :param true_labels: True labels as a list. Each label should be one of the elements in the 'classes' list.
    :param pred_labels: Predicted labels as a list. Each label should be one of the elements in the 'classes' list.
    :param classes: List of classes.
    :param output_dir: Output directory to store any output files or visualizations.
    :param output_prefix: option prefix added to the output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    class_id = {c: i for (i, c) in enumerate(classes)}
    y_true = [class_id[label] for label in true_labels]
    y_pred = [class_id[label] for label in pred_labels]

    fl_metrics = os.path.join(output_dir, '{}metrics.txt'.format(output_prefix))
    with open(fl_metrics, 'w') as fd_metrics:
        print('n_examples: {}'.format(len(y_true)))
        fd_metrics.write('n_examples: {}\n'.format(len(y_true)))

        acc = accuracy_score(y_true, y_pred)
        print('accuracy: {}'.format(acc))
        fd_metrics.write('accuracy: {}\n'.format(acc))

        class_report = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes)
        print(class_report)
        fd_metrics.write('classification_report:\n{}\n'.format(class_report))

        confmat = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        fd_metrics.write('confusion_matrix:\n{}\n'.format(confmat))

    s = np.sum(confmat, axis=1).reshape(len(classes), 1)
    normalized_confmat = confmat / s

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(normalized_confmat)
    # plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticks(list(range(len(classes))))
    ax.set_yticks(list(range(len(classes))))
    ax.set_xticklabels(classes, rotation='vertical', fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fl = os.path.join(output_dir, '{}confmat.png'.format(output_prefix))
    print('saving confusion matrix to file: {} ...'.format(fl))
    fig.tight_layout()
    plt.savefig(fl)


def evaluate_multi_label(true_labels, pred_labels, classes, output_dir, output_prefix=''):
    """
    Evaluate multi-label predictions. Computed accuracy, classification_report, and confusion matrix.

    :param true_labels: True labels as a list of lists. Each item in the inner list should be one of the elements in the 'classes' list.
    :param pred_labels: Predicted labels as a list of lists. Each item in the inner list should be one of the elements in the 'classes' list.
    :param classes: List of classes.
    :param output_dir: Output directory to store any output files or visualizations.
    :param output_prefix: option prefix added to the output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    class_id = {c: i for (i, c) in enumerate(classes)}

    # set-up binary indicator vectors for true and predicted labels
    y_true = np.zeros((len(true_labels), len(classes)))
    y_pred = np.zeros((len(true_labels), len(classes)))
    for i, sample_labels in enumerate(true_labels):
        for label in sample_labels:
            y_true[i, class_id[label]] = 1
    for i, sample_labels in enumerate(pred_labels):
        for label in sample_labels:
            y_pred[i, class_id[label]] = 1

    fl_metrics = os.path.join(output_dir, '{}metrics.txt'.format(output_prefix))
    with open(fl_metrics, 'w') as fd_metrics:
        print('n_examples: {}'.format(len(y_true)))
        fd_metrics.write('n_examples: {}\n'.format(len(y_true)))

        acc = accuracy_score(y_true, y_pred)
        print('accuracy: {}'.format(acc))
        fd_metrics.write('accuracy: {}\n'.format(acc))

        hamm_loss = hamming_loss(y_true, y_pred)
        print('hamming_loss: {}'.format(hamm_loss))
        fd_metrics.write('hamming_loss: {}\n'.format(hamm_loss))

        class_report = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes)
        print(class_report)
        fd_metrics.write('classification_report:\n{}\n'.format(class_report))

        confmat = np.zeros((len(classes), len(classes)))
        for i in range(len(y_true)):
            T = set(c for c in range(len(classes)) if y_true[i, c] == 1)
            P = set(c for c in range(len(classes)) if y_pred[i, c] == 1)
            for c in T:
                A = (P - T) | (P & {c})
                for c2 in A:
                    confmat[c, c2] += 1
        fd_metrics.write('confusion_matrix:\n{}\n'.format(confmat))

    s = np.sum(confmat, axis=1).reshape(len(classes), 1)
    normalized_confmat = confmat / s

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(normalized_confmat)
    # plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticks(list(range(len(classes))))
    ax.set_yticks(list(range(len(classes))))
    ax.set_xticklabels(classes, rotation='vertical', fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fl = os.path.join(output_dir, '{}confmat.png'.format(output_prefix))
    print('saving confusion matrix to file: {} ...'.format(fl))
    fig.tight_layout()
    plt.savefig(fl)


def read_data(filepath):
    """
    Reads evaluation from the provided input file. The file should be in the TSV format with at least 3 columns.
    1. sample/image ID.
    2. true labels as a comma-separated list.
    3. predicted labels as a comma-separated list.

    :param filepath: Path to the input file.
    :return: The data as a list of tuples (sample_id, true_labels, pred_labels)
    """
    data = []
    if not filepath:
        return data
    try:
        print('reading data from file: {} ...'.format(filepath))
        with open(filepath) as fd:
            lines = fd.read().splitlines()
    except FileNotFoundError:
        return data
    for line in lines:
        fields = line.split('\t')
        sample_id = fields[0]
        true_labels = list(set(fields[1].split(','))) if fields[1] else []
        pred_labels = list(set(fields[2].split(','))) if fields[2] else []
        data.append((sample_id, true_labels, pred_labels))
    return data


if __name__ == "__main__":
    # Command-line args
    arg_parser = argparse.ArgumentParser(description="Run evaluation on provided samples")
    arg_parser.add_argument('--input', default=None, type=str, required=True,
                            help='Input TSV file with 3 columns. '
                                 '<SampleId><TAB><Comma-sep True Label(s)><TAB><Comma-sep Predicted Label(s)>')

    arg_parser.add_argument('--output-dir', default=None, type=str, required=True)
    arg_parser.add_argument('--classes', default=None, type=str, required=True, help='File containing a list of classes')
    arg_parser.add_argument('--multi_label', default=False, action='store_true')
    arg_parser.add_argument('--multi_labels_for_multi_class', default='ignore', type=str, required=False,
                            help="Whether to 'ignore' or 'split' multiple label samples in case of multi-class training")

    args = vars(arg_parser.parse_args())
    print(str(args))

    fl_classes = os.path.join(args['classes'])
    with open(fl_classes) as fd:
        classes = fd.read().splitlines()
    print('read a list of {} classes: {}'.format(len(classes), classes))

    samples = read_data(args['input'])
    print('#samples: {}'.format(len(samples)))

    if args['multi_label']:
        true_labels, pred_labels = [], []
        for _sample_id, _true, _pred in samples:
            true_labels.append(_true)
            pred_labels.append(_pred)
        evaluate_multi_label(true_labels, pred_labels, classes, args['output_dir'])
    else:
        true_labels, pred_labels = [], []
        for _sample_id, _true, _pred in samples:
            if len(_true) == 1 or args['multi_labels_for_multi_class'] == 'split':
                true_labels += _true
                pred_labels += _pred
        evaluate_multi_class(true_labels, pred_labels, classes, args['output_dir'])

    print('script finished')
