import os
import glob
import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer

# Configuration - update paths to match your data structure
DATASET_ROOT = '/teamspace/studios/this_studio/dummy_dataset'
TRAIN_DIR = os.path.join(DATASET_ROOT, 'training')
VAL_DIR = os.path.join(DATASET_ROOT, 'validation')
OUTPUT_PATH = './outputs'
PROCESSED_DATA_DIR = os.path.join(OUTPUT_PATH, 'encoded_datasets')
MODELS_DIR = os.path.join(OUTPUT_PATH, 'models')
PLOTS_DIR = os.path.join(OUTPUT_PATH, 'plots')

# Create output directories
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model configuration
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_NAME = PRETRAINED_MODEL.split("/")[-1] + "-finetuned-gunshot"
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}-final")

CLASS_NAMES = ["Background", "Gunshot"]
label_to_id = {label: i for i, label in enumerate(CLASS_NAMES)}
id_to_label = {i: label for i, label in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
GUNSHOT_LABEL_ID = label_to_id["Gunshot"]

# Training hyperparameters
LR = 5e-5
BATCH_SIZE = 8
EPOCHS = 3
WEIGHT_DECAY_RATE = 0.01

# Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scan_spectrogram_files(data_dir, class_mapping):
    """
    Scan directory for .npy spectrogram files and create labels
    Expected structure: data_dir/[Background|Gunshot]/*.npy
    """
    all_files = []
    all_labels = []
    
    for class_name, class_id in class_mapping.items():
        class_folder = os.path.join(data_dir, class_name)
        spectrogram_files = glob.glob(os.path.join(class_folder, '*.npy'))
        
        for file_path in spectrogram_files:
            all_files.append(file_path)
            all_labels.append(class_id)
    
    return all_files, all_labels

def process_spectrograms_for_ast(batch, feature_extractor):
    """
    Process mel-spectrograms for AST model input
    Handles normalization, padding, and format conversion
    """
    processed_batch = []
    
    for file_path in batch['file_path']:
        # Load spectrogram from .npy file
        spectrogram = np.load(file_path).astype(np.float32)
        spectrogram = spectrogram.T  # Transpose for AST format
        
        # Apply normalization if required by feature extractor
        if feature_extractor.do_normalize:
            spectrogram = (spectrogram - feature_extractor.mean) / (feature_extractor.std + 1e-5)
        
        # Handle sequence length requirements
        current_length = spectrogram.shape[0]
        target_length = feature_extractor.max_length
        
        if current_length > target_length:
            # Trim from center if too long
            start_idx = (current_length - target_length) // 2
            spectrogram = spectrogram[start_idx : start_idx + target_length, :]
        elif current_length < target_length:
            # Pad with zeros if too short
            padding_length = target_length - current_length
            padding = np.zeros((padding_length, spectrogram.shape[1]), dtype=np.float32)
            spectrogram = np.concatenate((spectrogram, padding), axis=0)
        
        processed_batch.append(spectrogram)
    
    # Convert to tensor format
    input_tensor = torch.tensor(np.array(processed_batch), dtype=torch.float32)
    return {
        "input_values": input_tensor,
        "labels": torch.tensor(batch["label"], dtype=torch.long)
    }

def prepare_datasets():
    """
    Load spectrogram files and prepare datasets for AST training
    """
    # Scan for spectrogram files
    train_files, train_labels = scan_spectrogram_files(TRAIN_DIR, label_to_id)
    val_files, val_labels = scan_spectrogram_files(VAL_DIR, label_to_id)
    
    print(f"Found {len(train_files)} training files and {len(val_files)} validation files.")
    if not train_files or not val_files:
        print("ERROR: No .npy files found. Please check your TRAIN_DIR and VAL_DIR paths.")
        return
    
    # Create HuggingFace datasets
    train_data = {'file_path': train_files, 'label': train_labels}
    val_data = {'file_path': val_files, 'label': val_labels}
    
    training_dataset = Dataset.from_dict(train_data)
    validation_dataset = Dataset.from_dict(val_data)
    
    datasets = DatasetDict({
        'train': training_dataset,
        'validation': validation_dataset
    })
    
    # Load AST feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL)
    
    # Process datasets for AST input format
    processed_datasets = datasets.map(
        lambda batch: process_spectrograms_for_ast(batch, feature_extractor),
        batched=True,
        batch_size=100,
        remove_columns=['file_path']
    )
    
    # Save processed datasets
    processed_datasets.save_to_disk(PROCESSED_DATA_DIR)
    print(f"Processed datasets saved to: {PROCESSED_DATA_DIR}")
    feature_extractor.save_pretrained(PROCESSED_DATA_DIR)
    print(f"Feature extractor saved to: {PROCESSED_DATA_DIR}")

def calculate_metrics(evaluation_predictions):
    """
    Calculate performance metrics focusing on gunshot detection
    Returns accuracy, F1, precision, and recall for binary classification
    """
    try:
        logits, true_labels = evaluation_predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Load evaluation metrics
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        
        # Calculate metrics
        accuracy = accuracy_metric.compute(predictions=predicted_classes, references=true_labels)["accuracy"]
        f1_score = f1_metric.compute(predictions=predicted_classes, references=true_labels, 
                                   average="binary", pos_label=GUNSHOT_LABEL_ID)["f1"]
        precision = precision_metric.compute(predictions=predicted_classes, references=true_labels, 
                                           average="binary", pos_label=GUNSHOT_LABEL_ID)["precision"]
        recall = recall_metric.compute(predictions=predicted_classes, references=true_labels, 
                                     average="binary", pos_label=GUNSHOT_LABEL_ID)["recall"]
        
        return {"accuracy": accuracy, "f1": f1_score, "precision": precision, "recall": recall}
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        raise

def train_ast_model():
    """
    Train AST model for gunshot detection using processed datasets
    """
    # Load processed datasets and feature extractor
    try:
        processed_datasets = load_from_disk(PROCESSED_DATA_DIR)
        feature_extractor = AutoFeatureExtractor.from_pretrained(PROCESSED_DATA_DIR)
    except Exception as e:
        print(f"Error loading datasets or feature extractor: {e}")
        return
    
    # Load pre-trained AST model
    try:
        ast_model = AutoModelForAudioClassification.from_pretrained(
            PRETRAINED_MODEL,
            num_labels=NUM_CLASSES,
            label2id=label_to_id,
            id2label=id_to_label,
            ignore_mismatched_sizes=True
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Configure training parameters
    training_config = TrainingArguments(
        output_dir=MODEL_NAME,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        save_total_limit=2,
    )
    
    # Initialize trainer
    try:
        trainer = Trainer(
            model=ast_model,
            args=training_config,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["validation"],
            tokenizer=feature_extractor,
            compute_metrics=calculate_metrics,
        )
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return
    
    # Start training
    print("Starting AST model training...")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    try:
        training_results = trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Save trained model and feature extractor
    try:
        trainer.save_model(FINAL_MODEL_PATH)
        feature_extractor.save_pretrained(FINAL_MODEL_PATH)
        print(f"Fine-tuned AST model and feature extractor saved to: {FINAL_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # Display training results
    print("\nTraining completed successfully!")
    print("Training Results Summary:")
    print(training_results)
    print(f"\nModel saved to: {FINAL_MODEL_PATH}")
    print("Check the output directory for tensorboard logs and model checkpoints.")

if __name__ == "__main__":
    print("Starting AST model training pipeline...")
    print(f"Using device: {device}")
    
    # Step 1: Prepare datasets
    print("\n=== Step 1: Preparing datasets ===")
    prepare_datasets()
    
    # Step 2: Train model
    print("\n=== Step 2: Training AST model ===")
    train_ast_model()
    
    print("\n=== Pipeline completed ===")
    print("Check the outputs directory for trained models and logs.")