from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_PROVIDER: str = "openai"          # ollama # openai
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    CHROMA_PERSIST_DIR: str = "./chroma_store"
    CHROMA_COLLECTION: str = "banorte_docs"
    # RAG hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    # EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    TOP_K: int = 4

    RAG_SCORE_THRESHOLD: float = 1.0

    DATABASE_URL: str = "sqlite:///./banorte.db"
    # DATABASE_URL=postgresql://user:pass@localhost:5432/banorte

    APP_ENV: str = "development"


settings = Settings()

"""
#?
Hardcoded RAG hyperparameters (from settings):
    - CHUNK_SIZE      : 500 tokens (200-300 | 500 | 800-1200) - chars in each fragment of document read
    - CHUNK_OVERLAP   : 50  tokens (| 100-150) 10% del actual. Riesgo mas almecenamiento al vectoring
    - EMBEDDING_MODEL : all-MiniLM-L6-v2  (384-dim vectors) (all-MiniLM-L12-v2 384 | all-mpnet-base-v2 media | paraphrase-multilingual-MiniLM-L12-v2 384 | intfloat/multilingual-e5-large 1024 - mejor en traduccion de docs ) Convertir texto en vectores numericos para busquedas semanticas. Modelos mas grandes = mejor calidad pero mas lentos y costosos. Borra chroma_store/ y reinjest.
    - TOP_K           : 4   (2 | 4 | 6-8 | >10 ) (used at retrieval time, not here) # fragmentos del RAG se inyectan al prompt de LLM # costara mas tokens y puede agregar ruido al prompt
    - Estrategia de Busqueda: similarity_search es Cosine similarity, Aqui se mete TOP_K retriever.py. Se usa porque normalize_embeddings=True en loader.py  https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma/similarity_search
    - Vector store    : ChromaDB (local, persistent)
#?
"""

## CUANDO usar el RAG, cuando usar base de datos estructurada, o cuando ambos.

## Hacer keywords parametrizables, Quitar keywords. default