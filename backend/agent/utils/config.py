"""
Configuration settings for the Citizen Support Agent.
Uses pydantic-settings for environment variable management.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    OPENAI_API_KEY: str

    GROQ_API_KEY: str

    REDIS_URL: str = "redis://localhost:6379"

    KNOWLEDGE_DIR: Path = Path(__file__).parent.parent.parent / "data" / "knowledge"

    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
