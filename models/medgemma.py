from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from .base import ModelClient


class MedGemmaClient(ModelClient):
    """
    MedGemma served via UIS Ollama endpoint.
    """

    def __init__(
        self,
        model_id: str = "MedAIBase/MedGemma1.5:4b",
        api_url: str = "https://ollama.ux.uis.no/api/generate",
        api_key: Optional[str] = None,
        timeout_s: int = 180,
    ):
        self.model_id = model_id
        self.api_url = api_url
        self.timeout_s = timeout_s

        # Hent fra env hvis ikke sendt inn
        self.api_key = api_key or os.getenv("OLLAMA_UIS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing API key for MedGemmaClient. Set env var OLLAMA_UIS_API_KEY or pass api_key=..."
            )

    def _build_prompt(self, system_prompt: str, user_text: str) -> str:
        # Ollama /generate tar én prompt-streng; vi “fletter” system + user deterministisk.
        system_prompt = (system_prompt or "").strip()
        user_text = (user_text or "").strip()

        if system_prompt:
            return f"{system_prompt}\n\nUSER:\n{user_text}\n\nASSISTANT:"
        return f"USER:\n{user_text}\n\nASSISTANT:"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.api_url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"HTTP {r.status_code} from {self.api_url}: {r.text[:800]}") from e
        return r.json()

    def generate(self, system_prompt: str, user_text: str, max_new_tokens: int = 512) -> str:
        prompt = self._build_prompt(system_prompt, user_text)

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            # Mapp max_new_tokens til Ollama sin token-limit
            "options": {"num_predict": int(max_new_tokens)},
        }

        data = self._post(payload)
        return (data.get("response") or "").strip()
