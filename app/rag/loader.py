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

DOCS_DIR = Path("./docs")
PROCESS_KEYWORD_MAP = {
    "A": ["inquiry", "clarification", "question", "faq", "consulta", "aclaración"],
    "B": ["cancel", "cancellation", "product", "cancelar", "baja"],
    "C": ["incident", "escalation", "fraud", "incidente", "escalamiento"],
    "D": ["data update", "address", "phone", "actualizar datos", "domicilio"],
    "E": ["complaint", "claim", "queja", "reclamación", "reclamacion"],
}


def _detect_process_code(filename: str, text: str) -> str:
    tmp = (filename + " " + text[:400]).lower() # scan filename + first 400 chars for keywords
    for code, keywords in PROCESS_KEYWORD_MAP.items():
        if any(kw in tmp for kw in keywords):
            return code
    return "A"


def _get_loader(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    # else in txt file
    return TextLoader(str(file_path), encoding="utf-8")


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def ingest_document(file_path: Path) -> dict:
    # RAG Ingestion
    # LOAD - Read ./docs/
    # EMBED - HuggingFace for vectiruing
    # STORE - Persist to ChromaDB + register in SQL audit table

    # ingest documents to chroma db
    print(f"Processing: {file_path.name}")

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

    print(f"{len(chunks)} chunks created (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")
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
    # creates if not already created
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"Created empty docs/ directory. Add .txt or .pdf files there.")
        return []

    files = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))
    if not files:
        print("No documents found in docs/. RAG will use empty knowledge base.")
        return []

    results = []
    for f in files:
        try:
            results.append(ingest_document(f))
        except Exception as e:
            print(f"Failed on {f.name}: {e}")

    print(f"Ingestion complete. {len(results)} document(s) processed.")
    return results
