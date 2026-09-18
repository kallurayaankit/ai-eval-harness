"""Minimal Ollama client. Same shape as the security suite."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


class OllamaClient:
    """Talks to a locally-running Ollama server."""

    def __init__(self, base_url=None, model=None, system=None):
        self.base_url = base_url or os.getenv("OLLAMA_URL", DEFAULT_URL)
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.system = system

    def generate(self, prompt, timeout=300, temperature=None):
        options = {}
        if temperature is not None:
            options["temperature"] = temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if self.system:
            payload["system"] = self.system

        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["response"]

    def is_available(self):
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=2)
            return True
        except Exception:
            return False