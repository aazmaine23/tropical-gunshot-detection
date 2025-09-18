import os
import numpy as np
import torch
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    SwinForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Configuration
# --------------------------
MODEL_CHECKPOINT = "microsoft/swin-base-patch4-window7-224"
BASE_DATA_DIR = "/content/drive/MyDrive/sda-dataset-copy"   # Update this path
OUTPUT_DIR = "/content/drive/MyDrive/SwinFineTuning_Audio_CompetitiveRun"

LABELS = ["background", "gunshot"]
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in enumerate(LABELS)}

# --------------------------
# Device setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# Dataset loading
# --------------------------
def load_dataset(data_dir):
    """Load train/validation splits with .npy spectrograms."""
    data = {"train": [], "validation": []}

    for split in ["train", "validation"]:
        split_path = os.path.join(data_dir, split)
        for label, idx in label2id.items():
            class_path = os.path.join(split_path, label)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.endswith(".npy"):
                    data[split].append({
                        "file_path": os.path.join(class_path, fname),
                        "labels": idx
                    })
    return data

dataset_files = load_dataset(BASE_DATA_DIR)
train_ds = Dataset.from_list(dataset_files["train"])
val_ds   = Dataset.from_list(dataset_files["validation"])
raw_datasets = DatasetDict({"train": train_ds, "validation": val_ds})

print(f"Train size: {len(train_ds)}, Validation size: {len(val_ds)}")

# --------------------------
# Preprocessing
# --------------------------
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

def preprocess_fn(examples):
    images = []
    for path in examples["file_path"]:
        arr = np.load(path).astype(np.float32)

        # Normalize to [0, 1]
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())

        # Convert to RGB PIL
        if arr.ndim == 2:
            img = Image.fromarray((arr * 255).astype(np.uint8), "L").convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 1:
            img = Image.fromarray((arr[:, :, 0] * 255).astype(np.uint8), "L").convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
        else:
            raise ValueError(f"Unsupported shape {arr.shape} for {path}")
        images.append(img)

    processed = image_processor(images, return_tensors="pt")
    return {"pixel_values": processed["pixel_values"]}

tokenized = raw_datasets.map(
    preprocess_fn,
    batched=True,
    remove_columns=["file_path"]
).with_format("torch")

# --------------------------
# Model setup
# --------------------------
model = SwinForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(device)

# --------------------------
# Metrics
# --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    pos = label2id["gunshot"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=pos, zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision_gunshot": precision,
        "recall_gunshot": recall,
        "f1_gunshot": f1,
    }

# --------------------------
# Training setup
# --------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=25,
    learning_rate=3e-5,
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_gunshot",
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="tensorboard",
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=image_processor,
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7, early_stopping_threshold=1e-4)],
)

# --------------------------
# Train
# --------------------------
print("Starting training...")
trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final_best_model"))
print("Training complete.")

# --------------------------
# Evaluate & threshold tuning
# --------------------------
print("Evaluating...")
preds = trainer.predict(tokenized["validation"])
logits = preds.predictions
labels = preds.label_ids

probs = torch.softmax(torch.tensor(logits), dim=-1)[:, label2id["gunshot"]].numpy()

thresholds = np.linspace(0.01, 0.99, 197)
best_thresh, best_f1 = 0.5, -1
metrics_at_best = {}

for t in thresholds:
    pred_labels = (probs >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred_labels, average="binary", pos_label=1)
    if f1 > best_f1 and p >= 0.94 and r >= 0.95:  # your custom targets
        best_f1, best_thresh = f1, t
        metrics_at_best = {"precision": p, "recall": r, "f1": f1, "threshold": t}

print(f"Best threshold found: {best_thresh:.3f}, metrics: {metrics_at_best}")

# --------------------------
# Save final artifacts
# --------------------------
model_path = os.path.join(OUTPUT_DIR, "final_model")
processor_path = os.path.join(OUTPUT_DIR, "image_processor")
trainer.model.save_pretrained(model_path)
image_processor.save_pretrained(processor_path)

print("Done.")