import json
import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from PIL import Image
import numpy as np

# ===== Constants =====
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = r"C:\projects\scripts\Mushrooms"
EPOCHS = 20
FINE_TUNE_AT = 100
SEED = 42

# Ustaw seed dla reprodukowalności
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===== Data Augmentation (zastosowana PRZED preprocessingiem) =====
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),  # Zmniejszone - grzyby mają orientację
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2)
], name='data_augmentation')

# ===== Clean dataset =====


def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        # Ponowne otwarcie po verify()
        img = Image.open(file_path)
        img.load()
        return True
    except Exception as e:
        print(f"Invalid image: {file_path} - {e}")
        return False


def clean_directory(directory):
    removed_count = 0
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if not is_valid_image(file_path):
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
                removed_count += 1
    print(f"Removed {removed_count} corrupted images")


print("Cleaning dataset...")
clean_directory(DATA_DIR)

# ===== Load dataset =====
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'  # Explicit label mode
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nKlasy: {class_names}")
print(f"Liczba klas: {num_classes}")

# Policz przykłady w każdej klasie
class_counts = {}
for images, labels in train_ds:
    for label in labels.numpy():
        class_counts[label] = class_counts.get(label, 0) + 1

print("\nRozkład klas w zbiorze treningowym:")
for i, count in sorted(class_counts.items()):
    print(f"  {class_names[i]}: {count}")

# ===== Preprocessing pipeline (POPRAWNA KOLEJNOŚĆ) =====
AUTOTUNE = tf.data.AUTOTUNE


def prepare_dataset(ds, is_training=False):
    """Poprawna kolejność: augmentacja -> preprocessing -> cache/prefetch"""
    if is_training:
        # Augmentacja na obrazach 0-255
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Preprocessing MobileNetV2 (skalowanie do -1, 1)
    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)

    # Cache i prefetch dla wydajności
    if is_training:
        ds = ds.cache().prefetch(AUTOTUNE)
    else:
        ds = ds.cache().prefetch(AUTOTUNE)

    return ds


train_ds = prepare_dataset(train_ds, is_training=True)
val_ds = prepare_dataset(val_ds, is_training=False)

# ===== Build model =====


def build_model(num_classes, fine_tune=False):
    """Buduje model z opcją fine-tuningu"""

    # Base model
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        pooling='avg'  # Użyj wbudowanego poolingu
    )

    base_model.trainable = fine_tune

    if fine_tune:
        # Odmroź tylko ostatnie warstwy
        for layer in base_model.layers[:FINE_TUNE_AT]:
            layer.trainable = False
        print(
            f"Fine-tuning: {len([l for l in base_model.layers if l.trainable])} warstw odmrożonych")

    # Classification head
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    # Zawsze False dla BN podczas inferencji
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ===== Phase 1: Train classification head =====
print("\n" + "="*60)
print("PHASE 1: Training classification head")
print("="*60)

model = build_model(num_classes, fine_tune=False)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,  # Więcej epok - early stopping zatrzyma gdy potrzeba
    callbacks=callbacks_phase1,
    verbose=1
)

# ===== Phase 2: Fine-tuning =====
print("\n" + "="*60)
print("PHASE 2: Fine-tuning")
print("="*60)

# Przebuduj model z odmrożonymi warstwami
model = build_model(num_classes, fine_tune=True)

# Załaduj wagi z fazy 1
temp_model = build_model(num_classes, fine_tune=False)
temp_model.set_weights(model.get_weights())

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # Więcej epok - early stopping zatrzyma gdy potrzeba
    callbacks=callbacks_phase2,
    verbose=1
)

# ===== Save model =====
model_name = "mushrooms_mobilenet_v2.h5"
model.save(model_name)
print(f"\n✔ Model zapisany jako {model_name}")

# Zapisz również class_names
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("✔ Nazwy klas zapisane jako class_names.json")

# ===== Evaluation =====
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

# Zbierz wszystkie dane walidacyjne
val_images_list = []
val_labels_list = []
for batch_images, batch_labels in val_ds:
    val_images_list.append(batch_images)
    val_labels_list.append(batch_labels)

val_images = tf.concat(val_images_list, axis=0)
val_labels = tf.concat(val_labels_list, axis=0)

# Predykcje
predictions = model.predict(val_images, verbose=0)
predicted_classes = tf.argmax(predictions, axis=1).numpy()
true_classes = val_labels.numpy()

# Classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes,
                            target_names=class_names, digits=3))

# ===== Plots =====
# 1. Training history
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Val Accuracy', linewidth=2)
plt.axvline(x=len(history1.history['accuracy']), color='r',
            linestyle='--', label='Fine-tuning starts')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Val Loss', linewidth=2)
plt.axvline(x=len(history1.history['loss']), color='r',
            linestyle='--', label='Fine-tuning starts')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(12, 10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix", fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# 3. Sample predictions
print("\nGenerowanie przykładowych predykcji...")
raw_val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=False
)

# Weź losowe próbki
all_samples = list(raw_val_ds.unbatch().take(100))
random_samples = random.sample(all_samples, min(9, len(all_samples)))

plt.figure(figsize=(12, 12))
for i, (img, label) in enumerate(random_samples):
    # Preprocessing
    img_array = tf.expand_dims(img, axis=0)
    img_processed = preprocess_input(img_array)

    # Predykcja
    prediction = model.predict(img_processed, verbose=0)
    predicted_idx = np.argmax(prediction)
    confidence = prediction[0][predicted_idx] * 100

    # Wyświetl
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(img.numpy().astype("uint8"))

    true_label = class_names[label]
    pred_label = class_names[predicted_idx]
    color = 'green' if true_label == pred_label else 'red'

    plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%",
              color=color, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.show()

print("\n✔ Training complete!")
print(f"✔ Model: {model_name}")
print(f"✔ Classes: class_names.json")
print(f"✔ Plots: training_history.png, confusion_matrix.png, sample_predictions.png")
