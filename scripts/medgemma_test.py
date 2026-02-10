from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/medgemma-4b-pt"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

print("MedGemma loaded successfully!")
