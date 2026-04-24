from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Code before `yield` runs at startup; code after runs at shutdown.
    """
    print("\n" + "="*60)
    print("Banorte Assistant — Starting up")
    print("="*60)


    print("\nRunning RAG ingestion pipeline...")

    print("\nReady. Listening for requests.\n")

    yield  # ← App is running and accepting requests here

    print("\n[Shutdown] Banorte Assistant shutting down.")


# App

def create_app() -> FastAPI:
    app = FastAPI(
        title="Banorte Client Assistant API",
        description=(
            "Orchestrator Agent for Banorte bank clients. "
            "Handles 5 core processes: inquiry, cancellation, "
            "incident escalation, data update, and complaints."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    return app


app = create_app()


# Dev runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
