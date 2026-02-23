import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Laster tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else \
         "cpu"

print(f"Bruker device: {device}")

dtype = torch.bfloat16 if device == "cuda" else torch.float16

print("Laster modell... (første gang tar det litt tid)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto"
)

prompt = "Svar på norsk: Hva er forskjellen mellom overfitting og underfitting?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Genererer svar...\n")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
