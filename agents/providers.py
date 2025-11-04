import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderConfig:
    openai_api_key: Optional[str]
    groq_api_key: Optional[str]
    ollama_base_url: Optional[str]
    ngrok_authtoken: Optional[str]
    composer_enabled: bool


def load_provider_config() -> ProviderConfig:
    """Load provider configuration from environment variables.

    This module centralizes provider wiring so agents can route to different
    backends (Composer/Groq/Ollama) without touching call sites.
    """
    return ProviderConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        ngrok_authtoken=os.getenv("NGROK_AUTHTOKEN"),
        composer_enabled=os.getenv("COMPOSER_ENABLED", "true").lower() == "true",
    )


class ClientRouter:
    """Simple router facade for provider clients.

    Note: This intentionally avoids importing heavyweight SDKs. Integrations can
    be added later (e.g., OpenAI, Groq, Ollama HTTP) behind these methods.
    """

    def __init__(self, config: Optional[ProviderConfig] = None) -> None:
        self.config = config or load_provider_config()

    def use_composer(self) -> bool:
        return self.config.composer_enabled

    def get_openai_headers(self) -> dict:
        if not self.config.openai_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.openai_api_key}"}

    def get_groq_headers(self) -> dict:
        if not self.config.groq_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.groq_api_key}"}

    def get_ollama_base_url(self) -> Optional[str]:
        return self.config.ollama_base_url


