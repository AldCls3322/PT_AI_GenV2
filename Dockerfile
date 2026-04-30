FROM python:3.12-slim AS builder

# Evita que Python genere archivos .pyc y buffers en stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Directorio del venv dentro del builder
    VENV_PATH=/opt/venv

# Instalar dependencias del sistema mínimas para compilar paquetes nativos
# (chromadb y sentence-transformers requieren gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crear el entorno virtual
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Actualizar pip dentro del venv
RUN pip install --upgrade pip --no-cache-dir

# Requirements 
# torch CPU-only primero
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# img
FROM python:3.12-slim AS runtime

# image metadata 
LABEL maintainer="Banorte Digital Team" \
      description="Banorte Client Assistant — AI Orchestrator Agent API" \
      version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    # pip install -e
    PYTHONPATH=/app \
    # Avoid showing warnings - HuggingFace y ChromaDB
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# Activar el venv copiado desde el builder
ENV PATH="$VENV_PATH/bin:$PATH"

# Instalar solo dependencias de runtime (sin gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # curl: usado por el HEALTHCHECK
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Seguridad: usuario no-root ─────────────────────────────────────────────
# Los contenedores no deben correr como root en producción
RUN groupadd --gid 1001 banorte && \
    useradd --uid 1001 --gid banorte --shell /bin/bash --create-home banorte

# Directorio de trabajo de la aplicación
WORKDIR /app

# Copiar el venv compilado desde el builder (sin herramientas de compilación)
COPY --from=builder /opt/venv /opt/venv

# Copiar el código fuente del proyecto
COPY --chown=banorte:banorte app/           ./app/
COPY --chown=banorte:banorte docs/          ./docs/
COPY --chown=banorte:banorte run.py         ./run.py
COPY --chown=banorte:banorte pyproject.toml ./pyproject.toml

# ── Directorios de datos persistentes ─────────────────────────────────────
# Estos directorios se montan como volúmenes en docker-compose
# para que los datos sobrevivan reinicios del contenedor
RUN mkdir -p \
    /app/chroma_store \
    /app/data \
    /app/.cache/huggingface \
    && chown -R banorte:banorte /app

# no root
USER banorte

EXPOSE 8000

# suggested healthcheck
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# main run
CMD ["python", "run.py"]