# ── Stage 1: dependency + model builder ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the cross-encoder model weights into a known path.
# HF_HOME is set explicitly so we can COPY the cache into the runtime stage.
ENV HF_HOME=/hf_cache
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Cross-encoder pre-baked OK')
"

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pip packages
COPY --from=builder /install /usr/local

# Copy pre-baked HuggingFace model cache — this is the key step
COPY --from=builder /hf_cache /hf_cache

# Tell HuggingFace/sentence-transformers where the cache lives
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_CACHE=/hf_cache

# Copy application source
COPY . .

# Non-root user
RUN useradd -m -u 1000 nexus \
    && chown -R nexus:nexus /app \
    && chmod -R a+rX /hf_cache

USER nexus

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
