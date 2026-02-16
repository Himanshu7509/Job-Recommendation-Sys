import boto3
import os
from uuid import uuid4
import dotenv

# Load environment variables with fallback
if not dotenv.load_dotenv():
    dotenv.load_dotenv("app/.env")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

def upload_to_s3(file_path, original_name, delete_after=False):
    ext = os.path.splitext(original_name)[1]
    key = f"job-recomandation-system/resumes/{uuid4()}{ext}"

    try:
        s3.upload_file(file_path, BUCKET_NAME, key)
        return key
    finally:
        if delete_after and os.path.exists(file_path):
            os.remove(file_path)

def list_resumes():
    res = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="job-recomandation-system/resumes/")
    files = []

    for obj in res.get("Contents", []):
        files.append({
            "key": obj["Key"],
            "size": obj["Size"]
        })

    return files

def delete_resume(key):
    s3.delete_object(Bucket=BUCKET_NAME, Key=key)
