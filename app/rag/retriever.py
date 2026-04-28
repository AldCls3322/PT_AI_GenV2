from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from app.config import settings


def _get_embeddings() -> HuggingFaceEmbeddings:
    # Search strategy: similarity search (cosine via normalized embeddings).
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
        docs = []

    return docs


def retrieve_as_context(query: str, process_code: str | None = None) -> tuple[str, list[str]]:
    # Wraps ChromaDB with a LangChain retriever interface.
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
