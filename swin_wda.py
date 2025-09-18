import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration - update these paths to match your setup
MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
DATASET_PATH = "/teamspace/studios/this_studio/dataset"
OUTPUT_PATH = "/teamspace/studios/this_studio/output"
AUDIO_SAMPLE_RATE = 8000
CLIP_DURATION = 4  # seconds

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("GPU not available, using CPU. Training will be slower.")
    device = torch.device("cpu")

def scan_audio_dataset(dataset_root):
    """
    Scan directory structure for audio files and create dataset mappings
    Expected structure: dataset_root/[training|validation]/[background|gunshot]/*.wav
    """
    print(f"Loading dataset from: {dataset_root}")
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset directory '{dataset_root}' does not exist.")

    dataset_splits = {"train": [], "validation": []}
    class_names = ['background', 'gunshot']
    class_to_id = {"background": 0, "gunshot": 1}

    for split_name in ["training", "validation"]:
        split_key = "train" if split_name == "training" else "validation"
        split_folder = os.path.join(dataset_root, split_name)
        
        if not os.path.isdir(split_folder):
            print(f"Warning: Split directory '{split_folder}' not found. Skipping.")
            continue
            
        for class_name, class_id in class_to_id.items():
            class_folder = os.path.join(split_folder, class_name)
            if not os.path.isdir(class_folder):
                print(f"Warning: Category directory '{class_folder}' not found. Skipping.")
                continue
                
            file_count = 0
            for filename in os.listdir(class_folder):
                if filename.endswith(".wav"):
                    dataset_splits[split_key].append({
                        "file_path": os.path.join(class_folder, filename),
                        "labels": class_id
                    })
                    file_count += 1
            print(f"Found {file_count} .wav files in {class_folder}")

    if not dataset_splits["train"]:
        raise ValueError(f"No training data found in {os.path.join(dataset_root, 'training')}")
    if not dataset_splits["validation"]:
        raise ValueError(f"No validation data found in {os.path.join(dataset_root, 'validation')}")

    return dataset_splits

audio_files = scan_audio_dataset(DATASET_PATH)
training_dataset = Dataset.from_list(audio_files["train"])
validation_dataset = Dataset.from_list(audio_files["validation"])
datasets = DatasetDict({
    "train": training_dataset,
    "validation": validation_dataset
})

print(f"Datasets loaded: {datasets}")
print(f"Number of training samples: {len(datasets['train'])}")
print(f"Number of validation samples: {len(datasets['validation'])}")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

def convert_audio_to_spectrogram(batch):
    """
    Convert audio files to mel-spectrograms suitable for Swin transformer
    Handles resampling, duration normalization, and format conversion
    """
    processed_images = []
    
    for audio_path in batch["file_path"]:
        # Load audio file
        audio_signal, original_sr = torchaudio.load(audio_path)
        
        # Resample to target sample rate if needed
        if original_sr != AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(original_sr, AUDIO_SAMPLE_RATE)
            audio_signal = resampler(audio_signal)

        # Normalize audio duration to fixed length
        target_samples = CLIP_DURATION * AUDIO_SAMPLE_RATE
        if audio_signal.shape[1] > target_samples:
            # Trim if too long
            audio_signal = audio_signal[:, :target_samples]
        else:
            # Pad if too short
            padding_needed = target_samples - audio_signal.shape[1]
            audio_signal = torch.nn.functional.pad(audio_signal, (0, padding_needed))

        # Generate mel-spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIO_SAMPLE_RATE,
            n_fft=512,
            hop_length=160,
            n_mels=64,
            power=1.0
        )
        mel_spectrogram = mel_transform(audio_signal)

        # Convert to log scale and normalize
        log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        log_mel = (log_mel + 80) / 80  # Normalize to [0,1] range

        # Convert to RGB format for Swin (repeat grayscale across 3 channels)
        rgb_spectrogram = log_mel.repeat(3, 1, 1)
        
        # Resize to 224x224 for Swin input requirements
        resized_spec = F.interpolate(
            rgb_spectrogram.unsqueeze(0), 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

        # Convert to PIL Image format
        spec_numpy = resized_spec.permute(1, 2, 0).numpy()  # CHW -> HWC
        spec_uint8 = (spec_numpy * 255).astype(np.uint8)
        pil_image = Image.fromarray(spec_uint8, mode='RGB')
        processed_images.append(pil_image)

    # Process batch through image processor
    batch_processed = processor(processed_images, return_tensors="pt")
    return {"pixel_values": batch_processed["pixel_values"]}

processed_datasets = datasets.map(
    convert_audio_to_spectrogram,
    batched=True,
    remove_columns=["file_path"]
)
processed_datasets.set_format("torch")

print(f"Processed datasets: {processed_datasets}")
print(f"Example processed train sample shape: {processed_datasets['train'][0]['pixel_values'].shape}")

class_labels = ["background", "gunshot"]
id_to_label = {i: label for i, label in enumerate(class_labels)}
label_to_id = {label: i for i, label in enumerate(class_labels)}

swin_model = SwinForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(class_labels),
    id2label=id_to_label,
    label2id=label_to_id,
    ignore_mismatched_sizes=True
).to(device)

print(f"Model loaded with {len(class_labels)} labels: {class_labels}")

collator = DefaultDataCollator()

def calculate_metrics(evaluation_predictions):
    """
    Calculate performance metrics focusing on gunshot detection
    Returns accuracy, precision, recall, and F1 for the positive class
    """
    logits, true_labels = evaluation_predictions
    predicted_labels = np.argmax(logits, axis=-1)
    gunshot_label_id = label_to_id["gunshot"]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, 
        average='binary', 
        pos_label=gunshot_label_id, 
        zero_division=0
    )
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    return {
        "accuracy": accuracy,
        "precision_gunshot": precision,
        "recall_gunshot": recall,
        "f1_gunshot": f1,
    }

# Training hyperparameters
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_EVAL = 16
EPOCHS = 25
LR = 3e-5
GRAD_ACCUM_STEPS = 2
WEIGHT_DECAY_RATE = 0.05
WARMUP_PROPORTION = 0.1
SCHEDULER_TYPE = "cosine"

training_config = TrainingArguments(
    output_dir=os.path.join(OUTPUT_PATH, "checkpoints"),
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY_RATE,
    lr_scheduler_type=SCHEDULER_TYPE,
    warmup_ratio=WARMUP_PROPORTION,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_gunshot",
    greater_is_better=True,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="tensorboard",
    dataloader_num_workers=2,
)

early_stop_callback = EarlyStoppingCallback(
    early_stopping_patience=7,
    early_stopping_threshold=0.0001
)

trainer = Trainer(
    model=swin_model,
    args=training_config,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    processing_class=processor,
    data_collator=collator,
    compute_metrics=calculate_metrics,
    callbacks=[early_stop_callback]
)

print("Starting training...")
try:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    training_results = trainer.train()
    trainer.save_model(os.path.join(OUTPUT_PATH, "final_best_model"))
    trainer.log_metrics("train", training_results.metrics)
    trainer.save_metrics("train", training_results.metrics)
    trainer.save_state()
    print("Training finished.")
    if hasattr(training_results, 'metrics') and training_results.metrics.get("train_loss") is None:
        print("Note: Training may have stopped early. Check logs.")
except Exception as e:
    print(f"Training error: {e}")
    raise e

print("\nEvaluating the best model on the validation set...")
default_eval_results = trainer.evaluate(processed_datasets["validation"])
print("Evaluation results (default argmax threshold):")
for metric_name, metric_value in default_eval_results.items():
    print(f"  {metric_name}: {metric_value:.4f}")

print("\nPerforming threshold optimization. Target: P>0.94, R>0.95. Ideal: P>=0.97, R>=0.95.")
prediction_results = trainer.predict(processed_datasets["validation"])
model_logits = prediction_results.predictions
ground_truth_labels = prediction_results.label_ids
gunshot_class_id = label_to_id["gunshot"]

gunshot_probabilities = torch.softmax(torch.tensor(model_logits), dim=-1)[:, gunshot_class_id].numpy()

MIN_PRECISION_TARGET = 0.94
MIN_RECALL_TARGET = 0.95
IDEAL_PRECISION_TARGET = 0.97
IDEAL_RECALL_TARGET = 0.95

print(f"\nSearching for threshold to achieve P>{MIN_PRECISION_TARGET*100}% and R>={MIN_RECALL_TARGET*100}% for '{class_labels[gunshot_class_id]}'.")
print(f"Ideally aiming for P>={IDEAL_PRECISION_TARGET*100}% and R>={IDEAL_RECALL_TARGET*100}%.")

optimal_threshold = -1
best_f1_score = -1
best_metrics = {}
target_threshold_found = False

threshold_range = np.linspace(0.01, 0.99, 197)
precision_scores, recall_scores, f1_scores = [], [], []

for threshold_val in threshold_range:
    threshold_predictions = (gunshot_probabilities >= threshold_val).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth_labels, threshold_predictions, 
        average='binary', 
        pos_label=gunshot_class_id, 
        zero_division=0
    )
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Check if this threshold meets our targets
    if precision > MIN_PRECISION_TARGET and recall >= MIN_RECALL_TARGET:
        meets_ideal = (precision >= IDEAL_PRECISION_TARGET and recall >= IDEAL_RECALL_TARGET)
        
        if not target_threshold_found:
            target_threshold_found = True
            best_f1_score = f1
            optimal_threshold = threshold_val
        elif meets_ideal:
            # Prefer ideal thresholds
            prev_was_ideal = (best_metrics.get("precision_gunshot", 0) >= IDEAL_PRECISION_TARGET and
                             best_metrics.get("recall_gunshot", 0) >= IDEAL_RECALL_TARGET)
            if not prev_was_ideal or f1 > best_f1_score:
                best_f1_score = f1
                optimal_threshold = threshold_val
        elif f1 > best_f1_score and not (best_metrics.get("precision_gunshot", 0) >= IDEAL_PRECISION_TARGET and
                                        best_metrics.get("recall_gunshot", 0) >= IDEAL_RECALL_TARGET):
            best_f1_score = f1
            optimal_threshold = threshold_val

        # Update metrics if this is our new best threshold
        if optimal_threshold == threshold_val:
            accuracy = accuracy_score(ground_truth_labels, threshold_predictions)
            best_metrics = {
                "threshold": threshold_val,
                "accuracy": accuracy,
                "precision_gunshot": precision,
                "recall_gunshot": recall,
                "f1_gunshot": f1
            }

if target_threshold_found:
    print(f"\nThreshold found that surpasses P={MIN_PRECISION_TARGET*100}%, R={MIN_RECALL_TARGET*100}%:")
    for metric_key, metric_val in best_metrics.items():
        print(f"  {metric_key}: {metric_val:.4f}")
    if best_metrics["precision_gunshot"] >= IDEAL_PRECISION_TARGET and best_metrics["recall_gunshot"] >= IDEAL_RECALL_TARGET:
        print("This threshold also meets the ideal targets (P>=97%, R>=95%)!")

    # Generate confusion matrix
    confusion_labels = [label_to_id['background'], label_to_id['gunshot']]
    confusion_mat = confusion_matrix(
        ground_truth_labels, 
        (gunshot_probabilities >= optimal_threshold).astype(int), 
        labels=confusion_labels
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_labels[l] for l in confusion_labels],
                yticklabels=[class_labels[l] for l in confusion_labels])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix at Threshold {optimal_threshold:.3f}')
    matrix_save_path = os.path.join(OUTPUT_PATH, "confusion_matrix_chosen_threshold.png")
    plt.savefig(matrix_save_path)
    print(f"Confusion matrix saved to {matrix_save_path}")
else:
    print(f"\nCould not find a threshold to surpass P>{MIN_PRECISION_TARGET*100}% AND R>={MIN_RECALL_TARGET*100}%.")
    if f1_scores:
        best_f1_idx = np.argmax(f1_scores)
        fallback_threshold = threshold_range[best_f1_idx]
        print(f"\nBest F1 during threshold scan: {f1_scores[best_f1_idx]:.4f} at threshold {fallback_threshold:.3f}")
        print(f"  P: {precision_scores[best_f1_idx]:.4f}, R: {recall_scores[best_f1_idx]:.4f}")
        fallback_acc = accuracy_score(ground_truth_labels, (gunshot_probabilities >= fallback_threshold).astype(int))
        print(f"  Acc: {fallback_acc:.4f}")

model_save_path = os.path.join(OUTPUT_PATH, "final_best_f1_model")
print(f"\nSaving the best model to: {model_save_path}")
trainer.model.save_pretrained(model_save_path)

processor_save_path = os.path.join(OUTPUT_PATH, "final_image_processor")
print(f"Saving the image processor to: {processor_save_path}")
processor.save_pretrained(processor_save_path)

print("\nTraining and evaluation completed!")
if target_threshold_found:
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print("Performance metrics at this threshold:", best_metrics)
    print(f"Model saved at {model_save_path} is based on best F1_gunshot during training.")
    print(f"Apply threshold {optimal_threshold:.4f} during inference for optimal results.")
else:
    print(f"\nCould not achieve target performance (P={MIN_PRECISION_TARGET*100}%, R={MIN_RECALL_TARGET*100}%).")
    print(f"Model saved at {model_save_path} is based on best F1_gunshot during training.")
    print("Consider adjusting training parameters or collecting more data.")