# =============================
# app/services/incremental_index_builder.py
# HNSW-compatible + accuracy-safe + S3 sync
# =============================

import sys
import os
import dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

ENV_PATH = os.path.join(PROJECT_ROOT, "app", ".env")
dotenv.load_dotenv(ENV_PATH)

# os.environ["HF_HOME"] = "/tmp/hf_cache"
# os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

import numpy as np
import faiss
import boto3
from pymongo import MongoClient
from app.services.recommender import get_model
from app.core.config import DATA_DIR

# ---------------- CONFIG ----------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "job_recommendation"
COLLECTION = "jobs"

BUCKET = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_KEY = "faiss/jobs.index"
LOCAL_INDEX = f"{DATA_DIR}/jobs.index"

# ----------------------------------------


def build_job_text(job):
    return f"""
Job Title: {job.get('Job Title', '')}
Category: {job.get('Category', '')}
Experience Level: {job.get('Experience Level', '')}
Skills: {job.get('Skills', '')}
Requirements: {job.get('Requirements', '')}
Responsibilities: {job.get('Responsibilities', '')}
Job Description: {job.get('Job Description', '')}
"""


# ---------- S3 helpers ----------

def download_existing_index():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        s3.download_file(BUCKET, S3_KEY, LOCAL_INDEX)
        print("📥 Existing index downloaded")
        return faiss.read_index(LOCAL_INDEX)
    except Exception:
        print("ℹ No existing index found, will create new")
        return None


def upload_index():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(LOCAL_INDEX, BUCKET, S3_KEY)
    print("☁ Updated index uploaded to S3")


# ---------- Main logic ----------

def main():
    print("🔌 Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]

    new_jobs = list(col.find({"indexed": {"$ne": True}}))

    if not new_jobs:
        print("✅ No new jobs to index")
        return

    print(f"🆕 New jobs found: {len(new_jobs)}")

    job_texts = [build_job_text(j) for j in new_jobs]

    print("🤖 Loading embedding model...")
    model = get_model()

    print("🧠 Generating embeddings...")
    embeddings = model.encode(
        job_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    index = download_existing_index()

    if index is None:
        print("📐 Creating new HNSW index...")
        dim = embeddings.shape[1]
        M = 32
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64

    # Ensure HNSW parameters for existing index
    if isinstance(index, faiss.IndexHNSWFlat):
        index.hnsw.efSearch = 64

    print("➕ Adding embeddings to index...")
    index.add(embeddings)

    print("💾 Saving updated index locally...")
    faiss.write_index(index, LOCAL_INDEX)

    upload_index()

    # Mark jobs as indexed
    ids = [j["_id"] for j in new_jobs]
    col.update_many({"_id": {"$in": ids}}, {"$set": {"indexed": True}})

    print("✅ Incremental index update completed successfully")


if __name__ == "__main__":
    main()
