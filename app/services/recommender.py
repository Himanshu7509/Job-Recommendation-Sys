# =============================
# app/services/recommender.py
# MongoDB-schema compatible
# =============================

import os
import re
import numpy as np
import torch
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

from app.services.resume_parser import parse_resume
from app.services.index_manager import get_index, get_jobs_df

CACHE_DIR = "/app/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 20

_model = None


def get_model():
    global _model
    if _model is None:
        print("🔥 Loading embedding model at runtime...")
        _model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
        if not torch.cuda.is_available():
            torch.set_num_threads(4)
    return _model


# -----------------------------
# Helpers for your Mongo schema
# -----------------------------
def _as_text_list(v):
    """Return list[str] from value that may be list/str/None."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    return [str(v)]


def get_company_name(job):
    comp = job.get("company") or {}
    if isinstance(comp, dict):
        return str(comp.get("name", "")).strip()
    return ""


def get_location_text(job):
    locs = _as_text_list(job.get("location"))
    return ", ".join([x.strip() for x in locs if x.strip()])


def get_skills_text(job):
    skills = _as_text_list(job.get("skills"))
    return " ".join([x.lower().strip() for x in skills if x])


def get_salary_min(job):
    sal = job.get("salary") or {}
    if isinstance(sal, dict):
        return str(sal.get("min", "")).strip()
    return ""


def get_salary_max(job):
    sal = job.get("salary") or {}
    if isinstance(sal, dict):
        return str(sal.get("max", "")).strip()
    return ""


def clean_job_link(raw):
    if not raw:
        return ""

    raw = str(raw).strip()

    if "@" in raw and "http" not in raw:
        return f"mailto:{raw}"

    raw = raw.replace("https: ", "https://").replace("http: ", "http://").replace(" ", "")
    match = re.search(r"(https?://[^\s]+)", raw)
    return match.group(1) if match else raw


def parse_created_at(created_at):
    """
    Supports:
    - PyMongo datetime (datetime)
    - Extended JSON {"$date": "...Z"}
    - ISO string "...Z"
    """
    if not created_at:
        return None

    if isinstance(created_at, datetime):
        # Ensure timezone aware
        if created_at.tzinfo is None:
            return created_at.replace(tzinfo=timezone.utc)
        return created_at

    if isinstance(created_at, dict) and "$date" in created_at:
        created_at = created_at["$date"]

    if isinstance(created_at, str):
        try:
            return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            return None

    return None


def recency_boost(created_at, max_boost=0.08, decay_days=30):
    created_dt = parse_created_at(created_at)
    if not created_dt:
        return 0.0

    now = datetime.now(timezone.utc)
    age_days = max((now - created_dt).days, 0)
    return max_boost * max(0.0, (decay_days - age_days) / decay_days)


def final_score(similarity, job, resume_data):
    score = float(similarity)

    # skills overlap: your schema has skills: []
    job_skills_text = get_skills_text(job)
    overlap = sum(1 for s in resume_data.get("skills", []) if str(s).lower() in job_skills_text)
    score += 0.07 * overlap

    # experience check: your schema has experienceLevel
    exp_text = str(job.get("experienceLevel", "")).lower()
    exp_years = resume_data.get("experience_years")
    if exp_years is not None:
        # very simple heuristic: if "2 years" present etc.
        if str(exp_years) in exp_text:
            score += 0.15

    score += recency_boost(job.get("createdAt"))
    return score


# -----------------------------
# Main recommender
# -----------------------------
def recommend_jobs(resume_text: str):
    index = get_index()
    df = get_jobs_df()

    if index is None or df is None or df.empty:
        return {"error": "Recommendation system is warming up. Please try again shortly."}

    resume_data = parse_resume(resume_text)
    model = get_model()

    emb_vec = model.encode([resume_text], normalize_embeddings=True)[0]
    emb = np.asarray([emb_vec], dtype="float32")

    scores, indices = index.search(emb, TOP_K)

    ranked = []
    for rank, idx in enumerate(indices[0]):
        if idx >= len(df):
            continue

        job = df.iloc[idx]  # Pandas Series
        sim = float(scores[0][rank])
        score = final_score(sim, job, resume_data)

        created_dt = parse_created_at(job.get("createdAt")) or datetime.min.replace(tzinfo=timezone.utc)
        ranked.append((score, created_dt, idx))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    results = []
    for score, created_dt, idx in ranked[:TOP_K]:
        job = df.iloc[idx]

        results.append({
            "job_title": str(job.get("title", "")),
            "company": get_company_name(job),
            "location": get_location_text(job),
            "job_type": str(job.get("jobType", "")),
            "work_type": str(job.get("workType", "")),
            "experience": str(job.get("experienceLevel", "")),
            "skills": _as_text_list(job.get("skills")),
            "salary_min": get_salary_min(job),
            "salary_max": get_salary_max(job),
            "match_percentage": round(min(score * 100, 100), 2),
            "created_date": created_dt.isoformat() if created_dt and created_dt != datetime.min.replace(tzinfo=timezone.utc) else "",
            "job_link": clean_job_link(job.get("directLink", "")),
            "category": str(job.get("category", "")),
            "verification_status": str(job.get("verificationStatus", "")),
        })

    return results
