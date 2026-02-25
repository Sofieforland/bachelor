# models/base.py

# 
from abc import ABC, abstractmethod

class ModelClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_text: str, max_new_tokens: int = 512) -> str:
        """Return model output text."""
        raise NotImplementedError