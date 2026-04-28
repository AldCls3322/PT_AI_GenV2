# =============================================================================
# Banorte Assistant — Dockerfile
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder) — instala dependencias en un venv aislado
#   Stage 2 (runtime) — copia solo lo necesario, imagen final ligera
#
# Decisiones de diseño:
#   - Python 3.12-slim  : imagen base mínima (~50 MB vs ~900 MB de la full)
#   - torch CPU-only    : evita descargar drivers CUDA (~2 GB innecesarios)
#   - Non-root user     : buena práctica de seguridad en contenedores
#   - Volumes para data : chroma_store/ y banorte.db persisten fuera del contenedor
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — Builder
# Instala todas las dependencias Python en un venv limpio.
# Este stage NO llega a la imagen final — solo sus artefactos.
# -----------------------------------------------------------------------------
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

# ── Copiar requirements e instalar en dos pasos ────────────────────────────
# Paso 1: torch CPU-only primero (evita descargar el build con CUDA ~2 GB)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Paso 2: el resto de las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2 — Runtime
# Imagen final: solo Python + venv + código fuente. Sin gcc ni build tools.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Metadatos de la imagen
LABEL maintainer="Banorte Digital Team" \
      description="Banorte Client Assistant — AI Orchestrator Agent API" \
      version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    # Garantiza que el código del proyecto sea importable (equivale a pip install -e .)
    PYTHONPATH=/app \
    # Suprime warnings no críticos de HuggingFace y ChromaDB
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    ANONYMIZED_TELEMETRY=False \
    # Directorio de caché de HuggingFace dentro del contenedor
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

# Cambiar al usuario no-root
USER banorte

# Puerto expuesto por la API
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────
# Docker verifica que la app esté respondiendo cada 30 segundos.
# Un contenedor "unhealthy" puede ser reiniciado automáticamente por orquestadores.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Comando de inicio ──────────────────────────────────────────────────────
# Usa run.py en lugar de uvicorn directo para garantizar PYTHONPATH correcto
CMD ["python", "run.py"]