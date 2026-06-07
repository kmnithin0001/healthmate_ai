"""
config/settings.py
──────────────────
Houses global configuration parameters, API keys, paths, and constants.
"""

import os
from pathlib import Path

class Settings:
    # Google Gemini model parameters
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GENERATION_TEMPERATURE: float = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1500"))

    # Active API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "test_key_for_unit_tests")

    # Storage Paths
    WORKSPACE_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = WORKSPACE_ROOT / "data"
    
    # Persistent JSON memory store path
    default_store = str(DATA_DIR / "healthmate_memory.json")
    MEMORY_STORE_PATH: Path = Path(os.getenv("MEMORY_STORE_PATH", default_store))

    # Bounded in-memory turn limits
    SESSION_MEMORY_LIMIT: int = 15

    # Static healthcare safety disclaimers
    DISCLAIMER: str = (
        "\n\n⚕️ *Disclaimer: HealthMate AI provides general health information only. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider for medical concerns.*"
    )

settings = Settings()
