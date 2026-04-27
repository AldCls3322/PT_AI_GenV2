"""
RAG Ingestion Pipeline

LOAD     — Read raw documents from ./docs/ directory
CHUNK    — Split into overlapping text windows
EMBED    — Generate dense vectors via HuggingFace sentence-transformer
STORE    — Persist to ChromaDB + register in SQL audit table

Hardcoded RAG hyperparameters (from settings):
    - CHUNK_SIZE      : 500 tokens
    - CHUNK_OVERLAP   : 50  tokens
    - EMBEDDING_MODEL : all-MiniLM-L6-v2  (384-dim vectors)
    - TOP_K           : 4   (used at retrieval time, not here)
    - Vector store    : ChromaDB (local, persistent)
"""

import os
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import settings
from app.db.database import SessionLocal
from app.db.models import RagDocumentRegistry


# ── Constants ─────────────────────────────────────────────────────────────────
DOCS_DIR = Path("./docs")

# Process keyword map — used to tag documents with process_code A–E
PROCESS_KEYWORD_MAP = {
    "A": ["inquiry", "clarification", "question", "faq", "consulta", "aclaración"],
    "B": ["cancel", "cancellation", "product", "cancelar", "baja"],
    "C": ["incident", "escalation", "fraud", "incidente", "escalamiento"],
    "D": ["data update", "address", "phone", "actualizar datos", "domicilio"],
    "E": ["complaint", "claim", "queja", "reclamación", "reclamacion"],
}


def _detect_process_code(filename: str, text: str) -> str:
    """
    Heuristic: scan filename + first 400 chars of text for keywords.
    Falls back to 'A' (general inquiry) if nothing matches.
    """
    probe = (filename + " " + text[:400]).lower()
    for code, keywords in PROCESS_KEYWORD_MAP.items():
        if any(kw in probe for kw in keywords):
            return code
    return "A"


def _get_loader(file_path: Path):
    """Return the appropriate LangChain loader based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    # Default: plain text (also handles .txt, .md)
    return TextLoader(str(file_path), encoding="utf-8")


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Instantiate the embedding model.
    Estrategia de Busqueda: all-MiniLM-L6-v2 → 384-dimensional vectors, ~80 MB download, free.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Open (or create) the persisted ChromaDB collection."""
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def ingest_document(file_path: Path) -> dict:
    """
    Run the full ingestion pipeline for a single file.

    ret: dict + chunk_count and process_code.
    """
    print(f"[RAG Loader] 📄  Processing: {file_path.name}")

    loader = _get_loader(file_path)
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError(f"No content loaded from {file_path}")

    # chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    process_code = _detect_process_code(file_path.name, raw_docs[0].page_content)
    for chunk in chunks:
        chunk.metadata["source"]       = file_path.name
        chunk.metadata["process_code"] = process_code
        chunk.metadata["ingested_at"]  = datetime.utcnow().isoformat()

    print(f"[RAG Loader] ✂️   {len(chunks)} chunks created (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")

    # ── Stage 3: EMBED + Stage 4: STORE ──────────────────────────────────────
    embeddings    = _get_embeddings()
    vector_store  = _get_vector_store(embeddings)
    vector_store.add_documents(chunks)

    print(f"[RAG Loader] 💾  Stored in ChromaDB collection '{settings.CHROMA_COLLECTION}'")

    # Guardar en SQL db
    db = SessionLocal()
    try:
        registry = RagDocumentRegistry(
            process_code    = process_code,
            document_name   = file_path.name,
            chunk_count     = len(chunks),
            embedding_model = settings.EMBEDDING_MODEL,
            chunk_size      = settings.CHUNK_SIZE,
            chunk_overlap   = settings.CHUNK_OVERLAP,
            notes           = f"Auto-detected process: {process_code}",
        )
        db.add(registry)
        db.commit()
    finally:
        db.close()

    return {
        "file":         file_path.name,
        "process_code": process_code,
        "chunk_count":  len(chunks),
    }


def ingest_all_documents() -> list[dict]:
    """
    Scan ./docs/ and ingest every .txt and .pdf found.
    """
    # creates if not already created
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"[RAG Loader] ⚠️  Created empty docs/ directory. Add .txt or .pdf files there.")
        return []

    files = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))
    if not files:
        print("[RAG Loader] ⚠️  No documents found in docs/. RAG will use empty knowledge base.")
        return []

    results = []
    for f in files:
        try:
            results.append(ingest_document(f))
        except Exception as e:
            print(f"[RAG Loader] ❌  Failed on {f.name}: {e}")

    print(f"[RAG Loader] ✅  Ingestion complete. {len(results)} document(s) processed.")
    return results
