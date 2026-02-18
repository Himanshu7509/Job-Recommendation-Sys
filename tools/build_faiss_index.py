# =============================
# app/services/build_faiss_index.py
# HNSW + accuracy-safe + S3 upload
# =============================

import sys
import os
import dotenv

# ---------------- Path & env setup ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

ENV_PATH = os.path.join(PROJECT_ROOT, "app", ".env")
dotenv.load_dotenv(ENV_PATH)

# os.environ["HF_HOME"] = "/tmp/hf_cache"
# os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

# ---------------- Imports ----------------
import numpy as np
import pandas as pd
import faiss
import boto3
from pymongo import MongoClient
from app.services.recommender import get_model
from app.core.config import DATA_DIR

# ---------------- Config ----------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "jobsDB"
COLLECTION_NAME = "jobs"

OUTPUT_INDEX_PATH = f"{DATA_DIR}/jobs.index"

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_KEY = "job-recomandation-system/faiss/jobs.index"

# ------------------------------------------------


def build_job_text(row):
    return " | ".join(filter(None, [
        f"Title: {row.get('title', '')}",
        f"Category: {row.get('category', '')}",
        f"Experience: {row.get('experienceLevel', '')}",
        f"Skills: {', '.join(row.get('skills') or [])}",
        f"Requirements: {', '.join(row.get('requirements') or [])}",
        f"Responsibilities: {', '.join(row.get('responsibilities') or [])}",
        f"Description: {row.get('description', '')}",
    ]))



def upload_to_s3(local_path: str):
    print("☁ Uploading index to S3...")

    if not os.path.exists(local_path):
        raise FileNotFoundError("jobs.index not found for upload")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    s3.upload_file(local_path, AWS_BUCKET_NAME, S3_KEY)

    print("✅ jobs.index uploaded to S3 successfully")



def build_faiss_index():
    print("🔌 Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    jobs = list(collection.find({}, {"_id": 0}))
    if not jobs:
        raise ValueError("No jobs found in database")

    df = pd.DataFrame(jobs)

    print(f"📄 Jobs loaded: {len(df)}")

    print("📝 Building job texts...")
    job_texts = df.apply(build_job_text, axis=1).tolist()

    print("🤖 Loading embedding model...")
    model = get_model()

    print("🧠 Generating embeddings...")
    embeddings = model.encode(
        job_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    print("📐 Creating FAISS HNSW index (accuracy-safe)...")
    dim = embeddings.shape[1]

    # HNSW parameters tuned for accuracy
    M = 32
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64

    index.add(embeddings)

    print("💾 Saving index locally...")
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, OUTPUT_INDEX_PATH)

    print("✅ FAISS index created successfully")

    return OUTPUT_INDEX_PATH



def main():
    try:
        index_path = build_faiss_index()
        upload_to_s3(index_path)
        print("🎉 Build + Upload pipeline completed successfully")

    except Exception as e:
        print("❌ Pipeline failed:", str(e))
        raise


if __name__ == "__main__":
    main()

