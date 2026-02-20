# =============================
# app/services/build_faiss_index.py
# Full rebuild (all jobs) + S3 upload + mark indexed=True
# =============================

import sys
import os
import json
import dotenv
from datetime import datetime, timezone

# ---------------- Path & env setup ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

ENV_PATH = os.path.join(PROJECT_ROOT, "app", ".env")
dotenv.load_dotenv(ENV_PATH)

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
DB_NAME = "jobsDB"                 # <-- keep your DB name
COLLECTION_NAME = "jobs"

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_KEY = "job-recomandation-system/faiss/jobs.index"

LOCAL_INDEX = os.path.join(DATA_DIR, "jobs.index")
LOCAL_IDMAP = os.path.join(DATA_DIR, "jobs_id_map.json")


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
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(local_path, AWS_BUCKET_NAME, S3_KEY)
    print("✅ jobs.index uploaded to S3 successfully")


def build_faiss_index():
    print("🔌 Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=20000)
    client.admin.command("ping")

    col = client[DB_NAME][COLLECTION_NAME]

    # ✅ Keep _id so we can update and create id_map
    jobs = list(col.find({}))
    if not jobs:
        raise ValueError("No jobs found in database")

    df = pd.DataFrame(jobs)
    print(f"📄 Jobs loaded: {len(df)}")

    # Build texts
    print("📝 Building job texts...")
    job_texts = df.apply(build_job_text, axis=1).tolist()

    # Embed
    print("🤖 Loading embedding model...")
    model = get_model()

    print("🧠 Generating embeddings...")
    embeddings = model.encode(
        job_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    # FAISS HNSW
    print("📐 Creating FAISS HNSW index...")
    dim = embeddings.shape[1]
    M = 32
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(embeddings)

    # Save locally
    print("💾 Saving index + id_map locally...")
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, LOCAL_INDEX)

    # id_map: vector_position -> mongo_id (string)
    id_map = [str(x) for x in df["_id"].tolist()]
    with open(LOCAL_IDMAP, "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    print("✅ FAISS index built successfully")

    # ✅ Mark all jobs indexed=True after successful index write
    now = datetime.now(timezone.utc)
    res = col.update_many(
        {},
        {"$set": {"indexed": True, "indexedAt": now}}
    )
    print(f"✅ Marked indexed=True for {res.modified_count} jobs")

    return LOCAL_INDEX, LOCAL_IDMAP


def main():
    index_path, idmap_path = build_faiss_index()
    upload_to_s3(index_path)
    # (optional) also upload id_map to S3 for deployments
    # upload_to_s3(idmap_path)  # if you want, set a different S3 key
    print("🎉 Full rebuild + upload completed")


if __name__ == "__main__":
    main()