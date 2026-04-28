"""
Single source of truth for all environment-driven configuration.
Uses pydantic-settings: reads from .env automatically.
Access anywhere via:  from app.config import settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_PROVIDER: str = "openai"          # ollama # openai
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    # OLLAMA_BASE_URL: str = "http://localhost:11434"

    CHROMA_PERSIST_DIR: str = "./chroma_store"
    CHROMA_COLLECTION: str = "banorte_docs"
    # RAG hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   
    TOP_K: int = 4

    DATABASE_URL: str = "sqlite:///./banorte.db"

    APP_ENV: str = "development"


settings = Settings()

"""
#?
Hardcoded RAG hyperparameters (from settings):
    - CHUNK_SIZE      : 500 tokens
    - CHUNK_OVERLAP   : 50  tokens
    - EMBEDDING_MODEL : all-MiniLM-L6-v2  (384-dim vectors)
    - TOP_K           : 4   (used at retrieval time, not here)
    - Vector store    : ChromaDB (local, persistent)
#?
"""
