import argparse

import torch
import torchvision

from dataset import CustomDataset, NoisyDataset
from renate.shift.mmd_detectors import MMDCovariateShiftDetector


def main(args):
    # Load ResNet with ImageNet-pretrained weights; remove the last fully-connected layer.
    print("Loading pretrained model...")
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Identity()
    model.eval()

    # Create two subsets the training data.
    print("Preparing dataset...")
    dataset = CustomDataset(args["input"])
    if len(dataset) < 2 * args["num_points"]:
        raise ValueError(f"Reference and query size exceed dataset size.")
    dataset_ref = torch.utils.data.Subset(dataset, range(args["num_points"]))
    dataset_query = torch.utils.data.Subset(dataset, range(args["num_points"], 2 * args["num_points"]))

    # Apply noise to the query dataset.
    dataset_query = NoisyDataset(dataset_query, args["noise_std"])

    # Create the detector and fit it to the reference dataset
    print("Fitting shift detector to reference data...")
    shift_detector = MMDCovariateShiftDetector(feature_extractor=model)
    shift_detector.fit(dataset_ref)

    # Score the query dataset
    print("Scoring query data...")
    score = shift_detector.score(dataset_query)

    print(f"Score {score}")


if __name__ == "__main__":
    # Command-line args
    arg_parser = argparse.ArgumentParser(description="Run training on provided annotations")
    arg_parser.add_argument("--input", default=None, type=str, required=True,
        help="Input file with list of physical IDs. Input columns are copied as is to output <physicalId><TAB><other columns)>")
    arg_parser.add_argument("--num_points", type=int, default=100,
        help="Size of the reference and query dataset. Needs to be less than two times the dataset size.")
    arg_parser.add_argument("--noise_std", type=float, default=0.05,
        help="Standard deviation of the noise added to the query data to simulate shift.")
    args = vars(arg_parser.parse_args())
    main(args)
