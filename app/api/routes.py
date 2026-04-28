import traceback
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
 
from app.schemas.request_schema import ChatRequest, ChatResponse
from app.agent.orchestrator import run_agent
from app.rag.loader import ingest_all_documents
from app.db.database import get_db
from app.db.models import FunctionalProcess

logger = logging.getLogger(__name__)
router = APIRouter()

# POST
@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the Banorte assistant",
    tags=["Agent"],
)
async def chat( request: ChatRequest, db: Session = Depends(get_db),) -> ChatResponse:
    try:
        response = run_agent(request, db)
        return response
    except Exception as e:
        # traceback for check logs debug
        full_trace = traceback.format_exc()
        logger.error("Agent error:\n%s", full_trace)
        print(f"\n[ERROR] Full traceback:\n{full_trace}")
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
        full_trace = traceback.format_exc()
        logger.error("Ingestion error:\n%s", full_trace)
        print(f"\n[ERROR] Full traceback:\n{full_trace}")
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
