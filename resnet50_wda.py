import os
import librosa
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

# Data directories
TRAIN_DIR = '/teamspace/studios/ast-credit-short/fake_dataset/training'
VAL_DIR = '/teamspace/studios/ast-credit-short/fake_dataset/validation'
CLASS_NAMES = ['gunshot', 'background']

def load_spectrograms(data_dir, class_names, shape=(128, 128)):
    spectrograms, labels = [], []
    for idx, name in enumerate(class_names):
        folder = os.path.join(data_dir, name)
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.endswith('.wav'):
                continue
            path = os.path.join(folder, fname)
            try:
                audio, sr = librosa.load(path, sr=None)
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=4000)
                mel = librosa.power_to_db(mel, ref=np.max)
                mel = resize(np.expand_dims(mel, axis=-1), shape).numpy()
                spectrograms.append(mel)
                labels.append(idx)
            except Exception:
                continue
    return np.array(spectrograms), np.array(labels)

train_x, train_y = load_spectrograms(TRAIN_DIR, CLASS_NAMES)
val_x, val_y = load_spectrograms(VAL_DIR, CLASS_NAMES)

train_y_cat = to_categorical(train_y, num_classes=len(CLASS_NAMES))
val_y_cat = to_categorical(val_y, num_classes=len(CLASS_NAMES))

train_rgb = tf.repeat(train_x, 3, axis=-1)
val_rgb = tf.repeat(val_x, 3, axis=-1)

train_x_proc = preprocess_input(train_rgb)
val_x_proc = preprocess_input(val_rgb)

weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
class_wt = dict(enumerate(weights))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_recall', save_best_only=True, mode='max')
]

def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(len(CLASS_NAMES), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model

model, base_model = build_model()
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model.fit(train_x_proc, train_y_cat, epochs=5, batch_size=32,
          validation_data=(val_x_proc, val_y_cat), class_weight=class_wt, callbacks=callbacks)

def fine_tune(model, base_model, layers=10):
    base_model.trainable = True
    for layer in base_model.layers[:-layers]:
        layer.trainable = False
    return model

model = fine_tune(model, base_model)
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model.fit(train_x_proc, train_y_cat, epochs=5, batch_size=32,
          validation_data=(val_x_proc, val_y_cat), class_weight=class_wt, callbacks=callbacks)

results = model.evaluate(val_x_proc, val_y_cat)
print("Validation metrics:", dict(zip(model.metrics_names, results)))

preds = model.predict(val_x_proc)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(val_y_cat, axis=1)

print("\nReport:")
print(classification_report(true_labels, pred_labels, target_names=CLASS_NAMES))
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))
