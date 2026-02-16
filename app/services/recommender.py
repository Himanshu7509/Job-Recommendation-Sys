# =============================
# app/services/recommender.py
# Production-safe + optimized
# =============================

import os
import re
import numpy as np
import torch
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

from app.services.resume_parser import parse_resume
from app.services.index_manager import get_index, get_jobs_df

# -----------------------------
# HuggingFace cache config
# -----------------------------
CACHE_DIR = "/app/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# Model config
# -----------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 20

# -----------------------------
# Load model once (singleton)
# -----------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        print("🔥 Loading embedding model at runtime...")
        _model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=CACHE_DIR
        )
        # Performance optimization for CPU
        if not torch.cuda.is_available():
            torch.set_num_threads(4) 
    return _model


# -----------------------------
# Utils
# -----------------------------
def clean_job_link(raw):
    if not raw:
        return ""

    raw = raw.strip()

    if "@" in raw and "http" not in raw:
        return f"mailto:{raw}"

    raw = raw.replace("https: ", "https://")
    raw = raw.replace("http: ", "http://")
    raw = raw.replace(" ", "")

    match = re.search(r"(https?://[^\s]+)", raw)
    return match.group(1) if match else raw


def recency_boost(created_date, max_boost=0.08, decay_days=30):
    try:
        if isinstance(created_date, str):
            created_dt = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
        elif isinstance(created_date, datetime):
            created_dt = created_date
        else:
            return 0.0

        now = datetime.now(timezone.utc)
        age_days = max((now - created_dt).days, 0)
        return max_boost * max(0, (decay_days - age_days) / decay_days)

    except Exception:
        return 0.0


def final_score(similarity, row, resume_data):
    score = similarity

    job_skills = str(row.get("Skills", "")).lower()
    overlap = sum(1 for s in resume_data["skills"] if s in job_skills)
    score += 0.07 * overlap

    if resume_data.get("experience_years"):
        if str(resume_data["experience_years"]) in str(row.get("Experience Level", "")):
            score += 0.15

    score += recency_boost(row.get("created_date"))
    return score


# -----------------------------
# Main recommender
# -----------------------------
def recommend_jobs(resume_text: str):
    index = get_index()
    df = get_jobs_df()

    if index is None or df is None or df.empty:
        return {
            "error": "Recommendation system is warming up. Please try again shortly."
        }

    resume_data = parse_resume(resume_text)
    model = get_model()

    emb_vec = model.encode(
        [resume_text],
        normalize_embeddings=True
    )[0]

    emb = np.asarray([emb_vec], dtype="float32")
    scores, indices = index.search(emb, TOP_K)

    ranked = []
    for rank, idx in enumerate(indices[0]):
        if idx >= len(df):
            continue

        row = df.iloc[idx]
        sim = float(scores[0][rank])
        score = final_score(sim, row, resume_data)

        created_date = row.get("created_date")
        try:
            if isinstance(created_date, str):
                created_dt = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
            else:
                created_dt = created_date
        except Exception:
            created_dt = datetime.min

        ranked.append((score, created_dt, idx))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    results = []
    for score, _, idx in ranked[:TOP_K]:
        job = df.iloc[idx]

        results.append({
            "job_title": str(job.get("Job Title", "")),
            "company": str(job.get("Company Name", "")),
            "location": str(job.get("Location", "")),
            "experience": str(job.get("Experience Level", "")),
            "skills": str(job.get("Skills", "")),
            "salary_min": str(job.get("Salary Min (?)")),
            "salary_max": str(job.get("Salary Max (?)")),
            "match_percentage": round(min(score * 100, 100), 2),
            "created_date": str(job.get("created_date", "")),
            "job_link": clean_job_link(job.get("Direct Link", "")),
        })

    return results