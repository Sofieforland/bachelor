# models/llama_client.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import ModelClient

class LlamaClient(ModelClient):
    def __init__(self, model_id: str, torch_dtype="auto", device_map="auto", use_fast=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )
        self.model.eval()

    def _build_messages(self, system_prompt: str, user_text: str):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_text: str, max_new_tokens: int = 512) -> str:
        messages = self._build_messages(system_prompt, user_text)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated[:, prompt_len:]
        return self.tokenizer.decode(trimmed[0], skip_special_tokens=True).strip()
    