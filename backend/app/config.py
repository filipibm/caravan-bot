from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_docs"
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "claude-sonnet-4-6"
    top_k: int = 10
    min_score: float = 0.30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
