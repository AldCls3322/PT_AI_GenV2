"""
run.py
──────
Cross-platform launcher for the Banorte Assistant API.

Why this file exists
─────────────────────
Running `uvicorn app.main:app` directly from the terminal only works if Python
already knows the project root is on sys.path. This is not guaranteed on all
platforms (especially Windows) when using virtual environments.

This script explicitly adds the project root to sys.path before handing off
to uvicorn — eliminating "ModuleNotFoundError: No module named 'app.*'" errors.

Usage
─────
    python run.py              # development (--reload on)
    python run.py --prod       # production  (--reload off, 4 workers)
"""

import sys
import os

# ── Guarantee the project root is on sys.path ─────────────────────────────────
# __file__ is always the absolute path of this script.
# Its parent directory IS the project root (where `app/` lives).
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Launch uvicorn ─────────────────────────────────────────────────────────────
import uvicorn

def _is_docker() -> bool:
    """
    Detecta si el proceso corre dentro de un contenedor Docker.
    Usa dos señales independientes para mayor robustez:
      1. /.dockerenv — archivo creado por Docker en cada contenedor
      2. Variable de entorno DOCKER_CONTAINER=true — puede setearse
         manualmente en docker-compose.yml si la primera falla
    """
    return (
        os.path.exists("/.dockerenv")
        or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"
    )
 
 
def _resolve_config() -> dict:
    """
    Determina la configuración de uvicorn según el entorno detectado.
    Retorna un dict listo para pasar a uvicorn.run().
    """
    in_docker = _is_docker()
    is_prod   = (
        "--prod" in sys.argv
        or os.environ.get("APP_ENV", "development").lower() == "production"
    )
 
    # --reload solo tiene sentido en desarrollo local fuera de Docker
    use_reload = not is_prod and not in_docker
 
    # Múltiples workers solo en producción
    # IMPORTANTE: con workers > 1 la memoria de conversación (en RAM)
    # NO se comparte entre workers. Para producción real usar Redis.
    workers = 4 if is_prod else 1
 
    # Puerto: lee APP_PORT del entorno (docker-compose lo puede sobreescribir)
    port = int(os.environ.get("APP_PORT", 8000))
 
    return {
        "app":       "app.main:app",
        "host":      "0.0.0.0",
        "port":      port,
        "reload":    use_reload,
        "workers":   workers,
        "log_level": "info",
    }
 
 
if __name__ == "__main__":
    config    = _resolve_config()
    in_docker = _is_docker()
    is_prod   = os.environ.get("APP_ENV", "development").lower() == "production"
 
    # ── Banner de arranque ────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  Banorte Assistant — Iniciando")
    print("=" * 55)
    print(f"  Entorno  : {'producción' if is_prod else 'desarrollo'}")
    print(f"  Docker   : {'sí' if in_docker else 'no'}")
    print(f"  Host     : 0.0.0.0:{config['port']}")
    print(f"  Workers  : {config['workers']}")
    print(f"  Reload   : {'activado' if config['reload'] else 'desactivado'}")
    print("=" * 55)
    print()
 
    uvicorn.run(**config)