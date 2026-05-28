import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")

train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds  = test_ds.rename_column("label", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted")
    }

args = TrainingArguments(
    output_dir="models/roberta-injection",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="logs",
    logging_steps=20,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

print("\nStarting RoBERTa training...")
trainer.train()

print("\nFinal evaluation:")
results = trainer.evaluate()
print(results)

preds_output = trainer.predict(test_ds)
preds  = preds_output.predictions.argmax(-1)
labels = preds_output.label_ids

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Safe", "Injection"]))

trainer.save_model("models/roberta-injection/final")
tokenizer.save_pretrained("models/roberta-injection/final")
print("\nModel saved to models/roberta-injection/final")