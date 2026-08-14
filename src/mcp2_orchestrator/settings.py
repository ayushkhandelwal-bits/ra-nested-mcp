from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq free tier — https://console.groq.com (no credit card required)
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # URL of the MCP 1 HTTP/SSE server
    mcp1_url: str = "http://localhost:8001"
    mcp2_port: int = 8002


settings = Settings()  # type: ignore[call-arg]
