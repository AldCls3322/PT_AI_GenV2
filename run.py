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

if __name__ == "__main__":
    is_prod = "--prod" in sys.argv

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=not is_prod,          # Hot-reload in dev, off in prod
        workers=4 if is_prod else 1, # Multi-worker only in prod
        log_level="info",
    )
