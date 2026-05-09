import boto3

from settings import Settings


def download_dir_from_s3(bucket: str, prefix: str, local_dir):
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith("/"):
                continue

            relative_path = key.removeprefix(prefix).lstrip("/")
            local_path = local_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading s3://{bucket}/{key} -> {local_path}")
            s3.download_file(bucket, key, str(local_path))


def download_artifacts(settings: Settings):
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=settings.aws_region)

    print("Downloading classifier.joblib...")
    s3.download_file(
        settings.s3_bucket,
        "classifier.joblib",
        str(settings.classifier_joblib_path),
    )

    print("Downloading sentence transformer model...")
    download_dir_from_s3(
        settings.s3_bucket,
        "sentence_transformer.model",
        settings.sentence_transformer_dir,
    )


if __name__ == "__main__":
    download_artifacts(Settings())