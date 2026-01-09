"""
Configuration settings for the Citizen Support Agent.
Uses pydantic-settings for environment variable management.
"""

from pathlib import Path
from typing import Optional
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

    # OpenAI API Configuration
    OPENAI_API_KEY: str

    # Groq API Configuration
    GROQ_API_KEY: str

    # LiveKit Configuration
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_URL: str = "wss://localhost:7880"

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"

    # Knowledge Base Directory
    KNOWLEDGE_DIR: Path = Path(__file__).parent.parent.parent / "data" / "knowledge"

    # Optional: Additional settings
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure knowledge directory exists
        self.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
