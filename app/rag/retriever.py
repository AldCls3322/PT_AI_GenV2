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


def retrieve_with_scores(query: str) -> list[tuple[Document, float]]:
    try:
        vector_store = _get_vector_store()

        # Embedded model
        results = vector_store.similarity_search_with_score(
            query,
            k=settings.TOP_K,
        )
        # similarity_search_with_score returns (doc, distance)
        # Lower distance = more similar. Sort ascending to get best first.
        results.sort(key=lambda x: x[1])
        return results
    except Exception as e:
        print(f"[Retriever] ChromaDB query failed: {e}")
        return []


def retrieve_as_context(query: str) -> tuple[str, list[str], bool]:
    # Wraps ChromaDB with a LangChain retriever interface.
    raw_results = retrieve_with_scores(query)
 
    if not raw_results:
        print("No chunks returned from ChromaDB.")
        return "", [], False
 
    # Filter by relevance threshold
    threshold = settings.RAG_SCORE_THRESHOLD
    relevant = []
    for doc, score in raw_results:
        if score <= threshold:
            #add result
            relevant.append((doc, score))
    discarded = len(raw_results) - len(relevant)
 
    print(f"{len(raw_results)} chunks retrieved | "
          f"{len(relevant)} passed threshold ({threshold}) | "
          f"{discarded} discarded")
 
    if not relevant:
        print("All chunks below relevance threshold — RAG not used.")
        return "", [], False
 
    context_parts = []
    sources = []
 
    for i, (doc, score) in enumerate(relevant, 1):
        source       = doc.metadata.get("source", "unknown")
        process_code = doc.metadata.get("process_code", "?")
        context_parts.append(
            f"[Excerpt {i} | source: {source} | process: {process_code} | score: {score:.3f}]\n"
            f"{doc.page_content}"
        )
        if source not in sources:
            sources.append(source)
 
    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources, True
