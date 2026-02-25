# models/qwen_client.py
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .base import ModelClient

class QwenClient(ModelClient):
    def __init__(self, model_id: str, torch_dtype="auto", device_map="auto"):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _build_messages(self, system_prompt: str, user_text: str):
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_text: str, max_new_tokens: int = 512) -> str:
        messages = self._build_messages(system_prompt, user_text)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return text