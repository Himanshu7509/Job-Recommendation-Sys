# =============================
# app/services/incremental_index_builder.py
# Incremental add (indexed != True) + S3 sync + mark indexed=True
# =============================

import sys
import os
import json
import dotenv
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

ENV_PATH = os.path.join(PROJECT_ROOT, "app", ".env")
dotenv.load_dotenv(ENV_PATH)

import numpy as np
import faiss
import boto3
from pymongo import MongoClient
from app.services.recommender import get_model
from app.core.config import DATA_DIR

# ---------------- CONFIG ----------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "jobsDB"                 # <-- keep your DB name
COLLECTION_NAME = "jobs"

BUCKET = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_KEY_INDEX = "faiss/jobs.index"
LOCAL_INDEX = os.path.join(DATA_DIR, "jobs.index")

# id_map stored locally (recommended)
LOCAL_IDMAP = os.path.join(DATA_DIR, "jobs_id_map.json")
# ----------------------------------------


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


def download_existing_index():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        s3.download_file(BUCKET, S3_KEY_INDEX, LOCAL_INDEX)
        print("📥 Existing index downloaded")
        return faiss.read_index(LOCAL_INDEX)
    except Exception:
        print("ℹ No existing index found, will create new")
        return None


def upload_index():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(LOCAL_INDEX, BUCKET, S3_KEY_INDEX)
    print("☁ Updated index uploaded to S3")


def load_id_map():
    if not os.path.exists(LOCAL_IDMAP):
        return []
    with open(LOCAL_IDMAP, "r", encoding="utf-8") as f:
        return json.load(f)


def save_id_map(id_map):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOCAL_IDMAP, "w", encoding="utf-8") as f:
        json.dump(id_map, f)


def main():
    print("🔌 Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=20000)
    client.admin.command("ping")
    col = client[DB_NAME][COLLECTION_NAME]

    # ✅ Only jobs not indexed
    query = {"$or": [{"indexed": {"$exists": False}}, {"indexed": False}]}
    new_jobs = list(col.find(query))

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
    else:
        # Ensure HNSW params
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = 64

    # ✅ Append vectors
    print("➕ Adding embeddings to index...")
    index.add(embeddings)

    print("💾 Saving updated index locally...")
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, LOCAL_INDEX)

    # ✅ Update id_map in same order as embeddings added
    id_map = load_id_map()
    id_map.extend([str(j["_id"]) for j in new_jobs])
    save_id_map(id_map)

    # Upload index
    upload_index()

    # ✅ Mark jobs indexed ONLY after index saved + uploaded succeeded
    ids = [j["_id"] for j in new_jobs]
    now = datetime.now(timezone.utc)
    res = col.update_many(
        {"_id": {"$in": ids}},
        {"$set": {"indexed": True, "indexedAt": now}}
    )

    print(f"✅ Marked indexed=True for {res.modified_count} new jobs")
    print("✅ Incremental index update completed successfully")


if __name__ == "__main__":
    main()