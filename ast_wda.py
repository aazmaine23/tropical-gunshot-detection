# Imports
import os
import json
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Config
DATASET_ROOT = "/content/drive/MyDrive"
TRAIN_CHUNKS_DIR = os.path.join(DATASET_ROOT, "preprocessed_train")
VAL_CHUNKS_DIR = os.path.join(DATASET_ROOT, "preprocessed_validation")

OUTPUT_PATH = os.path.join(DATASET_ROOT, "results")
LOGS_DIR = "./logs"

NUM_LABELS = 2
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
FINAL_MODEL_PATH = os.path.join(OUTPUT_PATH, "final_model")

# Load datasets
def load_preprocessed_chunks(split_dir):
    chunks = []
    for chunk_file in sorted(os.listdir(split_dir)):
        if chunk_file.startswith("chunk_"):
            chunk = load_from_disk(os.path.join(split_dir, chunk_file))
            chunks.append(chunk)
    return concatenate_datasets(chunks)

encoded_dataset = DatasetDict({
    "train": load_preprocessed_chunks(TRAIN_CHUNKS_DIR),
    "validation": load_preprocessed_chunks(VAL_CHUNKS_DIR)
})

# Model
model = AutoModelForAudioClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, pos_label=1),
        "recall": recall_score(labels, predictions, pos_label=1),
        "f1": f1_score(labels, predictions, pos_label=1),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist()
    }

# Training setup
training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=LOGS_DIR,
    logging_steps=10,
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0))

# Train
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

def print_eval_results(eval_results):
    print("\n" + "="*40)
    print("Model Evaluation Results")
    print("="*40)
    print(f"Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall:    {eval_results['eval_recall']:.4f}")
    print(f"F1 Score:  {eval_results['eval_f1']:.4f}")
    print("\nConfusion Matrix:")
    cm = eval_results['eval_confusion_matrix']
    print(cm)
    print("="*40)

print_eval_results(eval_results)

# Best model
best_model_path = os.path.join(OUTPUT_PATH, "checkpoint-5250")  # update if needed
best_model = AutoModelForAudioClassification.from_pretrained(best_model_path)

# Predictions
from torch.utils.data import DataLoader

eval_dataloader = DataLoader(
    encoded_dataset["validation"],
    batch_size=4,
    collate_fn=lambda x: {k: torch.stack([torch.tensor(d[k]) for d in x]) for k in ['input_values', 'label']}
)

predictions, labels = [], []
for batch in eval_dataloader:
    with torch.no_grad():
        outputs = best_model(input_values=batch['input_values'])
    predicted_labels = np.argmax(outputs.logits.cpu().numpy(), axis=-1)
    predictions.extend(predicted_labels)
    labels.extend(batch['label'].cpu().numpy())

from sklearn.metrics import classification_report
report = classification_report(labels, predictions)
print(report)
