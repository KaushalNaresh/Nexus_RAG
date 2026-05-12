# ── Stage 1: dependency + model builder ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST, before any other packages.
# This prevents sentence-transformers from pulling in the CUDA build (~1.8 GB).
# CPU wheel: ~250 MB installed, ~100 MB RSS at runtime vs ~400 MB for CUDA.
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining app dependencies (torch already satisfied above).
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-bake cross-encoder weights so cold starts load from disk, not the internet.
ENV HF_HOME=/hf_cache
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('Cross-encoder pre-baked OK')"

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pip packages (includes CPU torch)
COPY --from=builder /install /usr/local

# Copy pre-baked HuggingFace model cache
COPY --from=builder /hf_cache /hf_cache
ENV HF_HOME=/hf_cache

# Copy application source
COPY . .

# Non-root user — must own /hf_cache so transformers can write its lock/migration files
RUN useradd -m -u 1000 nexus \
    && chown -R nexus:nexus /app \
    && chown -R nexus:nexus /hf_cache

USER nexus

EXPOSE 8000

# Longer start-period: warm-up loads models before first request
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
