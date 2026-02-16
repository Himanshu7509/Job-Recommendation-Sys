FROM python:3.10-slim

# ------------------------
# Python runtime settings
# ------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TOKENIZERS_PARALLELISM=false

# ------------------------
# System dependencies
# ------------------------
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------
# Working directory
# ------------------------
WORKDIR /app

# ------------------------
# Python dependencies
# ------------------------
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ------------------------
# spaCy language model
# ------------------------
RUN pip install \
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# ------------------------
# Application code
# ------------------------
COPY . .

# ------------------------
# Cleanup
# ------------------------
RUN rm -rf /root/.cache

# ------------------------
# Port
# ------------------------
EXPOSE 8000

# ------------------------
# Start server
# ------------------------
CMD [ "gunicorn", "app.main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120" ]
