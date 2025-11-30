# config/config.py
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Gemini / LLM
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
    GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

    # Database
    DB_PATH: str = os.getenv("DB_PATH", "hotel_bookings.db")

    # SMTP / Email
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    FROM_EMAIL: str = os.getenv("FROM_EMAIL", "")


settings = Settings()
