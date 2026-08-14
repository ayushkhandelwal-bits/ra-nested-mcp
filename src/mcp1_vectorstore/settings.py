from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local sentence-transformers model — no API key required.
    embedding_model: str = "all-MiniLM-L6-v2"
    mcp1_port: int = 8001


settings = Settings()  # type: ignore[call-arg]
