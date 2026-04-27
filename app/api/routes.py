"""
REST API

POST /chat — Main orchestration endpoint
POST /ingest — Trigger document ingestion into ChromaDB
GET  /processes — List all functional processes from DB
GET  /health — Health check
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.schemas.request_schema import ChatRequest, ChatResponse
from app.agent.orchestrator import run_agent
from app.rag.loader import ingest_all_documents
from app.db.database import get_db
from app.db.models import FunctionalProcess

router = APIRouter()


# POST
@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the Banorte assistant",
    tags=["Agent"],
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Main endpoint. Accepts a structured user message and returns
    an AI-generated response enriched with process metadata and RAG sources.

    - Validates input against the ChatRequest schema.
    - Routes to the orchestrator which handles classification, RAG, and LLM.
    - Returns a ChatResponse with reply + process info + document sources.
    """
    try:
        response = run_agent(request, db)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(e)}",
        )

@router.post(
    "/ingest",
    status_code=status.HTTP_200_OK,
    summary="Ingest documents from ./docs/ into ChromaDB",
    tags=["RAG"],
)
async def ingest():
    """
    Trigger RAG pipeline:
    Load → Chunk → Embedding → Store (ChromaDB + SQL).
    """
    try:
        results = ingest_all_documents()
        return {
            "status":    "success",
            "processed": len(results),
            "details":   results,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {str(e)}",
        )


# GET
@router.get(
    "/processes",
    status_code=status.HTTP_200_OK,
    summary="List all functional processes",
    tags=["Database"],
)
async def list_processes(db: Session = Depends(get_db)):
    """
    Returns all rows from functional_process table.
    Useful for verifying the seeded data or building a UI dropdown.
    """
    rows = db.query(FunctionalProcess).all()
    return {"processes": [row.to_dict() for row in rows]}

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    tags=["System"],
)
async def health():
    return {"status": "ok", "service": "banorte-assistant"}
