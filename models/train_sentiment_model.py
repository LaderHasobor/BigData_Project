# train_sentiment_model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from transformers import PhobertTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

# 1. Load dữ liệu
df = pd.read_csv("train.csv")
df = df[["text", "label"]]

# 2. Encode label (positive → 1, negative → 0)
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# 3. Tách train/test
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# 4. Load PhoBERT tokenizer
model_name = "vinai/phobert-base"
tokenizer = PhobertTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# 5. Chuyển sang HuggingFace Dataset
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# 6. Load PhoBERT model
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 7. Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 8. Metric đánh giá
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# 10. Train!
trainer.train()
trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

print("✅ Mô hình đã được huấn luyện và lưu tại ./sentiment_model")
print("🎉 Hoàn tất huấn luyện mô hình cảm xúc!")
# Note: Ensure you have the necessary libraries installed:
# pip install pandas scikit-learn torch transformers datasets