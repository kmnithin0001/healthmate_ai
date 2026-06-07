"""
utils/gemini_client.py
──────────────────────
Thin wrapper around google-genai SDK that all agents use.
Supports automatic fallback to legacy google-generativeai SDK.
"""

from __future__ import annotations
import time
from typing import List, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    """
    Singleton wrapper around the Gemini client.
    """
    _instance: Optional[GeminiClient] = None

    def __new__(cls) -> GeminiClient:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        
        try:
            # Attempt using the new google-genai SDK
            from google import genai
            from google.genai import types as genai_types
            self._sdk = "new"
            self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            self._genai_types = genai_types
            logger.info(f"GeminiClient loaded (google-genai SDK, model={settings.GEMINI_MODEL})")
        except (ImportError, Exception):
            try:
                # Fallback to legacy google.generativeai SDK
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=settings.GOOGLE_API_KEY)
                self._sdk = "legacy"
                self._model_legacy = genai_legacy.GenerativeModel(
                    model_name=settings.GEMINI_MODEL,
                    generation_config=genai_legacy.types.GenerationConfig(
                        temperature=settings.GENERATION_TEMPERATURE,
                        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
                        candidate_count=1,
                    )
                )
                logger.info(f"GeminiClient loaded (legacy SDK, model={settings.GEMINI_MODEL})")
            except (ImportError, Exception) as exc:
                self._sdk = "error"
                logger.warning(f"No Gemini SDK available. Running in offline/mock mode only: {exc}")
                
        self._initialized = True

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        retries: int = 2,
        backoff: float = 1.5
    ) -> str:
        """Call the Gemini API with retry logic and return the text output."""
        full_prompt = f"{system_prompt}\n\n---\nUser message: {user_message}"
        
        if self._sdk == "error" or not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "test_key_for_unit_tests":
            return "Mock AI response for testing."

        for attempt in range(1, retries + 1):
            try:
                if self._sdk == "new":
                    response = self._client.models.generate_content(
                        model=settings.GEMINI_MODEL,
                        contents=full_prompt,
                        config=self._genai_types.GenerateContentConfig(
                            temperature=settings.GENERATION_TEMPERATURE,
                            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
                        )
                    )
                    return response.text.strip()
                elif self._sdk == "legacy":
                    response = self._model_legacy.generate_content(full_prompt)
                    return response.text.strip()
            except Exception as exc:
                logger.warning(f"Gemini call attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    time.sleep(backoff * attempt)

        return "I'm sorry, I'm having trouble connecting to my AI backend. Please try again."
