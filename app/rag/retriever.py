"""
rag/retriever.py
────────────────
Query-time RAG retriever.
Wraps ChromaDB with a LangChain retriever interface.

Search strategy: similarity search (cosine via normalized embeddings).
TOP_K is read from settings (default: 4).
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from app.config import settings


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def retrieve(query: str, process_code: str | None = None) -> list[Document]:
    """
    Retrieve the TOP_K most relevant document chunks for a query.

    Parameters
    ----------
    query        : Natural-language user question.
    process_code : Optional filter — restricts results to a specific process (A–E).
                   Uses ChromaDB metadata filtering if provided.

    Returns
    -------
    List of LangChain Document objects with .page_content and .metadata.
    """
    vector_store = _get_vector_store()

    # Build optional where-filter for ChromaDB
    where_filter = {"process_code": process_code} if process_code else None

    try:
        if where_filter:
            docs = vector_store.similarity_search(
                query,
                k=settings.TOP_K,
                filter=where_filter,
            )
        else:
            docs = vector_store.similarity_search(query, k=settings.TOP_K)
    except Exception:
        # If collection is empty or filter matches nothing, return []
        docs = []

    return docs


def retrieve_as_context(query: str, process_code: str | None = None) -> tuple[str, list[str]]:
    """
    Convenience wrapper that returns:
      - A single formatted context string (ready to inject into a prompt)
      - A list of source filenames for the API response

    Used by agent/tools.py.
    """
    docs = retrieve(query, process_code)

    if not docs:
        return "No relevant information found in the knowledge base.", []

    context_parts = []
    sources = []

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Excerpt {i} — {source}]\n{doc.page_content}")
        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(context_parts)
    return context, sources
