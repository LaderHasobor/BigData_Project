import torch
from transformers import RobertaForSequenceClassification, PhobertTokenizer
import os

# ============================
# 🔧 Load model + tokenizer
# ============================
MODEL_DIR = os.path.join("models", "sentiment_model")

tokenizer = PhobertTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ============================
# 🔍 Hàm phân tích cảm xúc
# ============================
def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "neutral"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    return "positive" if predicted_class_id == 1 else "negative"
