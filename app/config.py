"""
config.py
─────────
Single source of truth for all environment-driven configuration.
Uses pydantic-settings: reads from .env automatically.
Access anywhere via:  from app.config import settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── LLM ──────────────────────────────────────────────────────
    LLM_PROVIDER: str = ""          # ollama openai
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── RAG / ChromaDB ────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_store"
    CHROMA_COLLECTION: str = "banorte_docs"

    # RAG hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   
    TOP_K: int = 4 

    # ── Database ──────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./banorte.db"

    # ── App ───────────────────────────────────────────────────────
    APP_ENV: str = "development"


settings = Settings()
