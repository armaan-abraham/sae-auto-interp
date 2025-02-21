from utils import data_dir, base_data_dir


def upload_data_dir():
    import shutil

    archive_name = f"{data_dir}.zip"
    shutil.make_archive(data_dir, "zip", data_dir)
    import boto3

    s3_client = boto3.client("s3")
    s3_client.upload_file(
        f"{data_dir}.zip", "deep-sae", f"auto-interp/{data_dir.name}.zip"
    )
    import os

    print(f"Zip file size: {os.path.getsize(archive_name) / (1024*1024):.2f} MB")


def download_data_dir(exp_name):
    import boto3
    import zipfile

    s3_client = boto3.client("s3")
    s3_client.download_file(
        "deep-sae", f"auto-interp/{exp_name}.zip", f"{exp_name}.zip"
    )
    with zipfile.ZipFile(f"{exp_name}.zip", "r") as zip_ref:
        zip_ref.extractall(base_data_dir / exp_name)


if __name__ == "__main__":
    download_data_dir("gpt2")
