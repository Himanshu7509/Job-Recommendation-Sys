import boto3
import os
import dotenv

dotenv.load_dotenv()

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

S3_KEY = "job-recomandation-system/faiss/jobs.index"
LOCAL_PATH = "data/jobs.index"


def download_index_from_s3(force_update=True):
    os.makedirs("data", exist_ok=True)

    if os.path.exists(LOCAL_PATH):
        if force_update:
            print("♻️ Existing FAISS index found. Replacing with latest version...")
            os.remove(LOCAL_PATH)
        else:
            raise FileExistsError("FAISS index already exists locally")

    print("⬇️ Downloading FAISS index from S3...")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(BUCKET_NAME, S3_KEY, LOCAL_PATH)

    print("✅ FAISS index downloaded and updated")
