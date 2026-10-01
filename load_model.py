# Run this on your LOCAL machine (not Kaggle)
from transformers import AutoTokenizer, AutoModel

model_name = "facebook/galactica-6.7b"
save_path = "./galactica-6.7b-local"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

print("Downloading model...")
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(save_path)

print(f"✅ Saved to {save_path}")
