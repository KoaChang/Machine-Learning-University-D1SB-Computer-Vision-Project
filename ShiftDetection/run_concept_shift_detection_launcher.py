import os
import logging
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import sagemaker
from sagemaker.pytorch import PyTorch


def get_execution_role():
    """
    :return: sagemaker execution role that is specified with the environment variable `AWS_ROLE`, if not specified then
    we infer it by searching for the role associated to Sagemaker. Note that
    `import sagemaker; sagemaker.get_execution_role()`
    does not return the right role outside of a Sagemaker notebook.
    """
    if "AWS_ROLE" in os.environ:
        aws_role = os.environ["AWS_ROLE"]
        logging.info(f"Using Sagemaker role {aws_role} passed set as environment variable $AWS_ROLE")
        return aws_role
    else:
        logging.info(f"No Sagemaker role passed as environment variable $AWS_ROLE, inferring it.")
        client = boto3.client("iam")
        sm_roles = client.list_roles(PathPrefix="/service-role/")['Roles']
        for role in sm_roles:
            if 'AmazonSageMaker-ExecutionRole' in role['RoleName']:
                return role['Arn']
        raise Exception(
            "Could not infer Sagemaker role, specify it by specifying `AWS_ROLE` environement variable " \
            "or refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html to create a new one"
        )

root = "src"


## Set experiment options
experiment_name = "lb-cmmd-clf-2"

datasets_and_models = [
    # ("CIFAR10", "ResNet18"),
    # ("CIFAR100", "ResNet18"),
    # ("ImageNet32", "ResNet18"),
    # ("ag_news", "DistilBert"),
    # ("SetFit/20_newsgroups", "DistilBert"),
    ("yelp_review_full", "DistilBert"),
]
query_types = ["pure_concept_shift", "combined_shift"] #["in_distribution", "in_support_covariate_shift", "out_of_support_covariate_shift", "pure_concept_shift", "combined_shift"]
reference_sizes = [5000]
query_sizes = [500, 1000]
out_percentages = [1.0]

detectors = ["MMD"]
prop_score_estimators = ["SVM", "RF"]
prop_score_cutoffs = [0.0, 0.05, 0.1]




seeds = [42] #list(range(1010, 1015))


# Create jobs
for dataset, model in datasets_and_models:
    for query_type in query_types:
        for reference_size in reference_sizes:
            for query_size in query_sizes:
                for detector in detectors:
                    for prop_score_estimator in prop_score_estimators:
                        for prop_score_cutoff in prop_score_cutoffs:
                            for seed in seeds:

                                sagemaker_client = boto3.client(
                                    service_name='sagemaker',
                                    config=Config(retries={
                                        'max_attempts': 12,
                                        'mode': 'standard'
                                    }),
                                )

                                sagemaker_session = sagemaker.Session(sagemaker_client=sagemaker_client)

                                pt_estimator = PyTorch(
                                    base_job_name=f"topmatch-{experiment_name}",
                                    source_dir=root,
                                    entry_point="run_concept_shift_detection.py",
                                    sagemaker_session=sagemaker_session,
                                    role=get_execution_role(),
                                    py_version="py38",
                                    framework_version="1.12.1",
                                    max_retry_attempts=30,
                                    volume_size=50,
                                    instance_count=1,
                                    # instance_type="ml.m5.2xlarge",
                                    instance_type="ml.c4.4xlarge",
                                    hyperparameters = {
                                        "dataset": dataset,
                                        "model": model,
                                        "query_type": query_type,
                                        "reference_size": reference_size,
                                        "query_size": query_size,
                                        "detector": detector,
                                        "prop_score_estimator": prop_score_estimator,
                                        "prop_score_cutoff": prop_score_cutoff,
                                        "seed": seed,
                                        "experiment_name": experiment_name,
                                        "results_bucket": "sven-experiments",
                                        "data_root_dir": "/opt/ml/input",
                                    }
                                )

                                print("Submitting job...")

                                success = False
                                ntry = 0
                                wait = [1, 2, 4, 8, 16, 32, 64, 128]

                                while not success and ntry < len(wait):
                                    try:
                                        pt_estimator.fit(wait=False)
                                        success = True
                                    except ClientError as err:
                                        errcode = err.response['Error']['Code']
                                        if errcode in ('LimitExceededException', 'ThrottlingException'):
                                            w = wait[ntry]
                                            print(f"We're getting throttled... waiting {w}s...")
                                            time.sleep(w)
                                        elif errcode in ['ResourceLimitExceeded', 'CapacityError']:
                                            w = wait[ntry] * 2
                                            print(f"Resource limit exceeded... waiting {w}m...")
                                            time.sleep(w * 60)
                                        else:
                                            raise err
                                    ntry += 1
