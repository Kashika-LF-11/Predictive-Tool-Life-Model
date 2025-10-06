#!/usr/bin/env python3
"""
Small helper to create a SageMaker training job (BYOC) using boto3.
It parses the input filename (expected: input_A_B_C(.csv|.parquet|...)) and
creates sensible
hyperparameters that the container (train.py) will consume.
Usage example (local):
python jobs/create_training_job.py \
  --input-s3-path s3://trained_datasets/datasets/input_A_B_C.csv
The script returns the create_training_job response.
"""
import argparse
import time
import os
import json
from urllib.parse import urlparse
import boto3
import yaml


def parse_input_basename(s3_path: str):
    """Return (machine-id, program-id, tool-id, basename_no_ext) from an s3 path whose base filename
    follows input_machine-id_program-id_tool-id"""
    base = os.path.basename(s3_path.rstrip("/"))
    name = os.path.splitext(base)[0]
    parts = name.split("_")
    if parts[0].lower() == "input":
        tokens = parts[1:]
    else:
        tokens = parts
    # pad tokens so we always have at least 3
    while len(tokens) < 3:
        tokens.append("-")
    machine_id, program_id, tool_id = tokens[:3]
    return machine_id, program_id, tool_id, name

def fetch_features_from_s3(input_s3_path: str):
    s3 = boto3.client("s3")
    parsed = urlparse(input_s3_path)
    bucket = parsed.netloc
    prefix = os.path.dirname(parsed.path.lstrip("/"))
    # the final_feature.json is a json list of cols to consider as features
    features_key = f"{prefix}/final_features.json"

    obj = s3.get_object(Bucket=bucket, Key=features_key)
    features = json.loads(obj["Body"].read().decode("utf-8"))
    return features

def create_training_job(config: dict, input_s3_path: str):
    """Create a SageMaker training job using config dict + dynamic input"""
    job_cfg = config["job"]
    s3_cfg = config["s3"]
    train_cfg = config["training"]

    sm_client = boto3.client("sagemaker", region_name=job_cfg.get("region"))

    # Parse input path to build output/model paths
    machine_id, program_id, tool_id, basename = parse_input_basename(input_s3_path)
    model_filename = f"model_{machine_id}_{program_id}_{tool_id}.onnx"
    output_s3_dir = f"s3://{s3_cfg['output_bucket']}/{machine_id}/{program_id}/{tool_id}/"

    # Assemble hyperparameters (here, the features_list)
    hyperparameters = {
        "features" : fetch_features_from_s3(input_s3_path), 
        "model_filename": model_filename
    }

    job_name = f"tool-life-{machine_id}-{program_id}-{tool_id}-{int(time.time())}"

    response = sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": job_cfg["image_uri"],
            "TrainingInputMode": "File",
        },
        RoleArn=job_cfg["role_arn"],
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_s3_path,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={
            "S3OutputPath": output_s3_dir
        },
        ResourceConfig={
            "InstanceType": train_cfg["instance_type"],
            "InstanceCount": train_cfg["instance_count"],
            "VolumeSizeInGB": train_cfg["volume_size"],
        },
        StoppingCondition={"MaxRuntimeInSeconds": train_cfg["max_runtime"]},
        HyperParameters=hyperparameters,
    )

    print(f"Created training job {job_name}")
    print("Hyperparameters sent to container:")
    for k, v in hyperparameters.items():
        print(f"  {k}: {v}")

    return response


if __name__ == "__main__":
    # Always load job_config.yaml from same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "job_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-s3-path", required=True, help="Path to training dataset in S3")
    args = parser.parse_args()

    create_training_job(config, args.input_s3_path)