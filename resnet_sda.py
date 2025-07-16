import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.image import resize
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Dataset paths - update these to match your data structure
TRAIN_DIR = '/teamspace/studios/this_studio/dataset/training'
VAL_DIR = '/teamspace/studios/this_studio/dataset/validation'
CLASS_NAMES = ['gunshot', 'background']

def load_spectrogram_data(data_dir, class_names):
    """
    Load mel-spectrograms from .npy files and prepare for ResNet50
    Handles resizing and format conversion for transfer learning
    """
    spectrograms = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Class directory '{class_folder}' not found.")
            continue

        print(f"Processing class: {class_name}")
        npy_files = [f for f in os.listdir(class_folder) if f.endswith('.npy')]
        print(f"Found {len(npy_files)} .npy files in {class_folder}")

        for filename in npy_files:
            file_path = os.path.join(class_folder, filename)
            try:
                # Load spectrogram data
                spectrogram = np.load(file_path)
                
                # Add channel dimension if needed (grayscale to single channel)
                if spectrogram.ndim == 2:
                    spectrogram = np.expand_dims(spectrogram, axis=-1)
                
                # Resize to ResNet input size if needed
                if spectrogram.shape[:2] != (128, 128):
                    spectrogram = resize(spectrogram, (128, 128))
                
                spectrograms.append(spectrogram)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error processing '{file_path}': {e}")

    print(f"Total data samples loaded: {len(spectrograms)}")
    return np.array(spectrograms), np.array(labels)

train_spectrograms, train_labels = load_spectrogram_data(TRAIN_DIR, CLASS_NAMES)
val_spectrograms, val_labels = load_spectrogram_data(VAL_DIR, CLASS_NAMES)

# Convert labels to one-hot encoding
train_labels_onehot = to_categorical(train_labels, num_classes=len(CLASS_NAMES))
val_labels_onehot = to_categorical(val_labels, num_classes=len(CLASS_NAMES))

# Convert grayscale spectrograms to RGB for ResNet50 compatibility
train_rgb = tf.repeat(train_spectrograms, 3, axis=-1)
val_rgb = tf.repeat(val_spectrograms, 3, axis=-1)

# Apply ResNet50 preprocessing
train_processed = preprocess_input(train_rgb)
val_processed = preprocess_input(val_rgb)

# Handle class imbalance with weighted training
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(np.argmax(train_labels_onehot, axis=1)),
    y=np.argmax(train_labels_onehot, axis=1)
)
class_weights_dict = dict(enumerate(weights))

# Training callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=25, 
                              verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_recall',
                            save_best_only=True, mode='max', verbose=1)

def build_resnet_classifier():
    """
    Build ResNet50-based classifier for audio classification
    Uses transfer learning with frozen base layers initially
    """
    # Load pre-trained ResNet50 without top classification layers
    backbone = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )
    backbone.trainable = False  # Freeze base layers for initial training

    # Add custom classification head
    pooled_features = GlobalAveragePooling2D()(backbone.output)
    dense_layer = Dense(128, activation='relu')(pooled_features)
    predictions = Dense(len(CLASS_NAMES), activation='softmax')(dense_layer)
    
    return Model(inputs=backbone.input, outputs=predictions)

model = build_resnet_classifier()

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("Starting initial training with frozen backbone...")
initial_history = model.fit(
    train_processed, train_labels_onehot,
    epochs=200,
    batch_size=32,
    validation_data=(val_processed, val_labels_onehot),
    class_weight=class_weights_dict,
    callbacks=[early_stopping, checkpoint]
)

def enable_fine_tuning(model, layers_to_unfreeze=10):
    """
    Enable fine-tuning by unfreezing top layers of ResNet50
    Keeps early layers frozen to preserve learned features
    """
    backbone = model.layers[1]  # ResNet50 backbone
    backbone.trainable = True
    
    # Keep early layers frozen, only train last N layers
    for layer in backbone.layers[:-layers_to_unfreeze]:
        layer.trainable = False
    
    return model

print("Preparing for fine-tuning...")
model = enable_fine_tuning(model, layers_to_unfreeze=10)

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("Starting fine-tuning...")
finetune_history = model.fit(
    train_processed, train_labels_onehot,
    epochs=60,
    batch_size=32,
    validation_data=(val_processed, val_labels_onehot),
    class_weight=class_weights_dict,
    callbacks=[early_stopping, checkpoint]
)

print("\nEvaluating final model...")
final_loss, final_acc, final_prec, final_rec = model.evaluate(val_processed, val_labels_onehot)
print(f"Validation Metrics:\n  Loss: {final_loss:.4f}\n  Accuracy: {final_acc:.4f}"
      f"\n  Precision: {final_prec:.4f}\n  Recall: {final_rec:.4f}")

# Generate predictions for detailed evaluation
predictions = model.predict(val_processed)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(val_labels_onehot, axis=1)

print("\nClassification Report:")
detailed_report = classification_report(true_classes, predicted_classes,
                                       target_names=CLASS_NAMES)
print(detailed_report)

print("\nConfusion Matrix:")
confusion_mat = confusion_matrix(true_classes, predicted_classes)
print(confusion_mat)

print("\nSaving trained model...")
model.save('best_model.keras')

# Save to models directory
models_dir = "/teamspace/studios/this_studio/models"
os.makedirs(models_dir, exist_ok=True)
final_model_path = os.path.join(models_dir, "resnet50_gunshot_model.keras")
model.save(final_model_path)

print(f"Training completed! Model saved to: {final_model_path}")
print("Check the evaluation metrics above for detailed performance analysis.")