FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    SEED_DIR=/app/sample_docs

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY sample_docs ./sample_docs

# Bake the embedding model into the image so the container starts offline and fast.
RUN python -c "from app.embeddings import FastEmbedEmbedder; FastEmbedEmbedder('BAAI/bge-small-en-v1.5', cache_dir='/app/models')"
ENV EMBEDDING_CACHE_DIR=/app/models

RUN useradd --create-home appuser && mkdir -p /data && chown -R appuser /data /app/models
USER appuser
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
