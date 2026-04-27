# ── Path bootstrap — MUST be first, before any app.* imports ─────────────────
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from fastapi import FastAPI

# ── Import models BEFORE seed/init so Base.metadata is fully populated ────────
import app.db.models  # noqa: F401
from app.api.routes import router
from app.db.seed import init_db
from app.rag.loader import ingest_all_documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("  Banorte Assistant - Starting up")
    print("="*60)

    print("\n[Startup] Initializing database...")
    init_db()

    print("\n[Startup] Running RAG ingestion pipeline...")
    ingest_all_documents()

    print("\n[Startup] Ready. Listening for requests.\n")
    yield
    print("\n[Shutdown] Banorte Assistant shutting down.")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Banorte Client Assistant API",
        description=(
            "Orchestrator Agent for Banorte bank clients. "
            "Handles 5 core processes: inquiry, cancellation, "
            "incident escalation, data update, and complaints."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router, prefix="/api/v1")
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)