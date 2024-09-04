import argparse
import json
import math
import os
import time

import boto3
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

from data import get_dataset
from models import get_model
from detectors import MMDConceptShiftDetector, AlibiConceptShiftDetector
import scenarios


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Dataset.")
    parser.add_argument("--scenario", type=str, default="BinarySuperclass", help="Scenario.")
    parser.add_argument("--query_type", type=str, default="pure_concept_shift")
    parser.add_argument("--detector", type=str, default="MMD", help="Detector to use.")
    parser.add_argument("--mmd_kernel", type=str, default="RBF_median", help="Kernel to use in MMD.")
    parser.add_argument("--prop_score_estimator", type=str, default="SVM")
    parser.add_argument("--prop_score_cutoff", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="ResNet18", help="Model")
    parser.add_argument("--model_init", type=str, default="pretrain")
    parser.add_argument("--model_training", type=str, default="none")
    parser.add_argument("--reference_size", type=int, default=1000)
    parser.add_argument("--query_size", type=int, default=100)
    parser.add_argument("--out_percentage", type=float, default=1.0)
    parser.add_argument("--num_query_batches", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=100, help="Batch size used to iterate over datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--data_root_dir", type=str, default="./data")
    parser.add_argument("--experiment_name", type=str, default="balleslb-contest-1",
                        help="A name for the experiment, will be prepended to the results file.")
    parser.add_argument("--results_bucket", type=str, default="sven-experiments",
                        help="S3 bucket to store results.")
    parser.add_argument("--device", type=str, default="cpu", help="Device type. Options: cpu or gpu")

    args = parser.parse_args()

    run(args)


def sample(ds_in: Dataset, ds_out: Dataset, num_points: int, out_percent: float) -> Dataset:
    """Samples a random subset mixed from two datasets.

    The resulting dataset has size `num_points` and contains a fraction
    `out_percent` of points from dataset `ds_out`, the remainder are points
    from `ds_in`. Sampling is done uniformly without replacement.

    Args:
        ds_in: In-distribution dataset.
        ds_out: Out-of-distribution dataset.
        num_points: Size of the desired subset.
        out_percent: Percentage of the sampled subset that comes fom `ds_out`.

    Returns:
        A pytorch dataset object containing the random subset.
    """
    num_out = math.ceil(out_percent * num_points)
    num_in = num_points - num_out
    if len(ds_in) < num_in or len(ds_out) < num_out:
        raise ValueError("Datasets too small to sample desired subsets.")
    inds_in = torch.randperm(len(ds_in))[:num_in]
    inds_out = torch.randperm(len(ds_out))[:num_out]
    return ConcatDataset([Subset(ds_in, inds_in), Subset(ds_out, inds_out)])

def run(args):

    # General set-up.
    torch.manual_seed(args.seed)

    # Set up model.
    model = get_model(args.model, pretrain=(args.model_init == "pretrain"))
    if isinstance(model, tuple):
        model, tokenizer = model
    else:
        tokenizer = None

    # Load dataset and set up scenario.
    dataset_tr, dataset_te = get_dataset(args.dataset, data_root_dir=args.data_root_dir, tokenizer=tokenizer)
    dataset = ConcatDataset([dataset_tr, dataset_te])
    if args.scenario == "BinarySuperclass":
        ds_ref, query_sets = scenarios.make_binary_superclass_scenario(dataset, args.reference_size)
    ds_in = query_sets["in_distribution"]
    ds_out = query_sets[args.query_type]

    # Set up shift detector.
    if args.detector == "MMD":
        detector = MMDConceptShiftDetector(
            prop_score_estimator=args.prop_score_estimator,
            prop_score_cutoff=args.prop_score_cutoff,
            feature_extractor=model,
            return_p_value=True,
            batch_size=args.batch_size,
            device=args.device,
            num_preprocessing_workers=0
        )
    elif args.detector == "Alibi":
        detector = AlibiConceptShiftDetector(
            feature_extractor=model,
            return_p_value=True,
            batch_size=args.batch_size,
            device=args.device,
            num_preprocessing_workers=0
        )
    else:
        raise ValueError(f"Unknown detector: {args.detector}.")
    detector.fit(ds_ref)


    # Run experiment for each combination of query batch size and percentage of
    # OOD data. Sample the same number of batches of in-domain and query data.
    scores = []
    t0 = time.time()
    for i in range(args.num_query_batches):
        print(f"Test batch {i}/{args.num_query_batches}")
        ds_query = sample(ds_in, ds_out, args.query_size, out_percent=args.out_percentage)
        assert len(ds_query) == args.query_size
        score = detector.score(ds_query)
        scores.append(score)
        print(f"score {score}")

    time_per_query = (time.time() - t0) / args.num_query_batches

    print("Experiment completed. Storing results.")
    results = {
        "scores": scores,
        "time_per_query": time_per_query,
    }
    results.update(vars(args))
    results_filename = os.path.join(args.experiment_name, str(time.time()))
    s3 = boto3.client("s3")
    s3.put_object(Bucket=args.results_bucket, Key=results_filename, Body=json.dumps(results))


if __name__ == "__main__":
    main()
