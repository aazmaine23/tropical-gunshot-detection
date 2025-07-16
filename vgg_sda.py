import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Dataset paths - update these to match your data structure
TRAIN_DIR = '/teamspace/studios/this_studio/dataset/training' 
VAL_DIR = '/teamspace/studios/this_studio/dataset/validation'
RESULTS_DIR = 'evaluation_results'
CLASS_NAMES = ['gunshot', 'background']

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_audio_files(data_dir, class_names):
    """
    Scan directories for .npy audio files and create labels
    Returns file paths and corresponding numeric labels
    """
    all_files = []
    all_labels = []
    
    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_folder):
            print(f"Warning: Missing directory {class_folder}")
            continue
            
        # Find all numpy files in this class folder
        npy_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) 
                    if f.endswith('.npy') and os.path.isfile(os.path.join(class_folder, f))]
        
        if not npy_files:
            print(f"Warning: No NPY files found in {class_folder}")
            
        all_files.extend(npy_files)
        all_labels.extend([idx] * len(npy_files))
    
    return all_files, all_labels

print("Loading training data...")
train_files, train_labels = load_audio_files(TRAIN_DIR, CLASS_NAMES)
print(f"Found {len(train_files)} training samples")

print("\nLoading validation data...")
val_files, val_labels = load_audio_files(VAL_DIR, CLASS_NAMES)
print(f"Found {len(val_files)} validation samples")

def build_tf_dataset(file_paths, labels, batch_size=32):
    """
    Create TensorFlow dataset from file paths and labels
    Handles mel-spectrogram loading and preprocessing for VGG16
    """
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.shuffle(buffer_size=max(len(file_paths), 1))
    
    def preprocess_audio(file_path, label):
        def load_and_normalize_spectrogram(path):
            # Load numpy array from file
            path_str = path.numpy().decode('utf-8')
            spectrogram = np.load(path_str)
            
            # Min-max normalization to 0-255 range
            spec_min = np.min(spectrogram)
            spec_max = np.max(spectrogram)
            if spec_max > spec_min:
                spectrogram = 255 * (spectrogram - spec_min) / (spec_max - spec_min)
            else:
                spectrogram = np.zeros_like(spectrogram)
            return spectrogram.astype(np.float32)

        # Process the spectrogram
        processed_spec = tf.py_function(load_and_normalize_spectrogram, [file_path], tf.float32)
        processed_spec.set_shape([224, None])  # Height fixed, width variable
        
        # Resize to square image and convert to RGB for VGG16
        processed_spec = tf.image.resize(processed_spec[..., tf.newaxis], (224, 224))
        processed_spec = tf.repeat(processed_spec, 3, axis=-1)  # Grayscale to RGB
        processed_spec = tf.keras.applications.vgg16.preprocess_input(processed_spec)
        
        return processed_spec, tf.one_hot(label, len(CLASS_NAMES))

    ds = ds.map(preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("\nCreating training dataset...")
train_ds = build_tf_dataset(train_files, train_labels)
print("Creating validation dataset...")
val_ds = build_tf_dataset(val_files, val_labels)

def build_vgg_classifier(input_shape=(224, 224, 3)):
    """
    Build VGG16-based classifier for audio classification
    Uses transfer learning with frozen base layers
    """
    # Load pre-trained VGG16 without top classification layers
    backbone = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    backbone.trainable = False  # Freeze base layers initially

    # Add custom classification head
    inputs = Input(shape=input_shape)
    features = backbone(inputs)
    pooled = GlobalAveragePooling2D()(features)
    dense_layer = Dense(256, activation='relu')(pooled)
    dropout_layer = Dropout(0.5)(dense_layer)
    predictions = Dense(len(CLASS_NAMES), activation='softmax')(dropout_layer)

    return Model(inputs, predictions)

print("\nInitializing model...")
model = build_vgg_classifier()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Training callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_vgg_model.keras',
                                  monitor='val_recall',
                                  mode='max',
                                  save_best_only=True,
                                  verbose=1)

# Handle class imbalance with weighted training
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(weights))

print("\nStarting initial training...")
initial_history = model.fit(
    train_ds,
    epochs=25,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, model_checkpoint]
)

def enable_fine_tuning(model):
    """
    Unfreeze top layers of VGG16 for fine-tuning
    Keeps bottom layers frozen to preserve learned features
    """
    model.layers[1].trainable = True  # Enable training for VGG backbone
    # Keep early layers frozen, only train last 8 layers
    for layer in model.layers[1].layers[:-8]:
        layer.trainable = False
    return model

print("\nPreparing for fine-tuning...")
model = enable_fine_tuning(model)
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

print("\nStarting fine-tuning...")
finetune_history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, model_checkpoint]
)

print("\nLoading best model for evaluation...")
best_model = load_model('best_vgg_model.keras')

# Collect predictions for evaluation metrics
true_labels = []
predicted_labels = []
prediction_scores = []

print("\nRunning predictions...")
for batch_x, batch_y in val_ds:
    # Get true labels from one-hot encoding
    batch_true = np.argmax(batch_y.numpy(), axis=1)
    true_labels.extend(batch_true)
    
    # Get model predictions
    batch_probs = best_model.predict(batch_x, verbose=0)
    prediction_scores.extend(batch_probs[:, 0])  # Gunshot class probability
    predicted_labels.extend(np.argmax(batch_probs, axis=1))

print("\nGenerating evaluation metrics...")

# Create confusion matrix visualization
confusion_mat = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
plt.close()

# Generate ROC curve
false_pos_rate, true_pos_rate, _ = roc_curve(true_labels, prediction_scores)
area_under_curve = auc(false_pos_rate, true_pos_rate)
plt.figure()
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=2, 
         label=f'ROC Curve (AUC = {area_under_curve:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
plt.close()

# Create precision-recall curve
prec_values, recall_values, _ = precision_recall_curve(true_labels, prediction_scores)
plt.figure()
plt.plot(recall_values, prec_values, color='blue', lw=2, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.png'))
plt.close()

# Save detailed classification report
detailed_report = classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES)
with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as report_file:
    report_file.write(detailed_report)

print(f"\nTraining completed! Results saved to: {RESULTS_DIR}")
print("Check the evaluation_results folder for detailed metrics and visualizations.")