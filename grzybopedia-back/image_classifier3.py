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
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import numpy as np

# ===== Constants =====
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = r"C:\projects\scripts\Mushrooms"
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===== SILNIEJSZA augmentacja =====
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),  # Również pionowo
    layers.RandomRotation(0.3),  # Więcej rotacji
    layers.RandomZoom(0.3),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3),
    # Dodatkowe augmentacje
    layers.GaussianNoise(0.1),
], name='data_augmentation')

# ===== Clean dataset =====


def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        img = Image.open(file_path)
        img.load()
        return True
    except Exception as e:
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
                print(f"Removing: {file_path}")
                os.remove(file_path)
                removed_count += 1
    print(f"Usunięto {removed_count} uszkodzonych obrazów\n")


print("Czyszczenie datasetu...")
clean_directory(DATA_DIR)

# ===== Load dataset =====
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
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
print(f"Klasy: {class_names}")
print(f"Liczba klas: {num_classes}\n")

# ===== OBLICZ CLASS WEIGHTS (kluczowe!) =====
print("Obliczanie wag klas...")
all_labels = []
for _, labels in train_ds:
    all_labels.extend(labels.numpy())

class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
class_weights = dict(enumerate(class_weights_array))

print("\nRozkład klas i wagi:")
class_counts = {i: all_labels.count(i) for i in range(num_classes)}
for i, count in sorted(class_counts.items()):
    print(
        f"  {class_names[i]}: {count} obrazów (waga: {class_weights[i]:.2f})")

# ===== Preprocessing =====
AUTOTUNE = tf.data.AUTOTUNE


def prepare_dataset(ds, is_training=False):
    if is_training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)

    return ds.cache().prefetch(AUTOTUNE)


train_ds = prepare_dataset(train_ds, is_training=True)
val_ds = prepare_dataset(val_ds, is_training=False)

# ===== Build model z MOCNIEJSZĄ regularyzacją =====


def build_model(num_classes, fine_tune=False, dropout_rate=0.5):

    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    base_model.trainable = fine_tune

    if fine_tune:
        # Odmroź tylko ostatnie 50 warstw (mniej niż wcześniej)
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        print(
            f"Fine-tuning: {len([l for l in base_model.layers if l.trainable])} warstw\n")

    # SILNIEJSZA regularyzacja w głowie
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = base_model(inputs, training=False)

    # Więcej dropoutu i regularyzacji L2
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        128,  # Mniejsza warstwa (było 256)
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.02)  # Silniejsza L2
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ===== FAZA 1: Trening głowy (z class weights!) =====
print("="*70)
print("FAZA 1: Trening classification head z class weights")
print("="*70 + "\n")

model = build_model(num_classes, fine_tune=False, dropout_rate=0.6)
model.compile(
    optimizer=Adam(learning_rate=5e-4),  # Mniejszy LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # Bardziej agresywna redukcja
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks_phase1,
    class_weight=class_weights,  # KLUCZOWE!
    verbose=1
)

# ===== FAZA 2: Fine-tuning =====
print("\n" + "="*70)
print("FAZA 2: Fine-tuning (mniej warstw, mniejszy dropout)")
print("="*70 + "\n")

# Przebuduj z fine-tuningiem i mniejszym dropoutem
model_ft = build_model(num_classes, fine_tune=True, dropout_rate=0.4)

# Kopiuj wagi z fazy 1
for i, layer in enumerate(model_ft.layers):
    if i < len(model.layers):
        try:
            layer.set_weights(model.layers[i].get_weights())
        except:
            pass

model_ft.compile(
    optimizer=Adam(learning_rate=1e-5),  # Bardzo mały LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-8,
        verbose=1
    )
]

history2 = model_ft.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks_phase2,
    class_weight=class_weights,  # KLUCZOWE!
    verbose=1
)

model = model_ft  # Użyj fine-tuned model

# ===== Save =====
model.save("mushrooms_mobilenet_v2.h5")
print("\n✔ Model zapisany: mushrooms_mobilenet_v2.h5")

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("✔ Klasy zapisane: class_names.json\n")

# ===== Evaluation =====
print("="*70)
print("EWALUACJA")
print("="*70 + "\n")

val_images_list = []
val_labels_list = []
for batch_images, batch_labels in val_ds:
    val_images_list.append(batch_images)
    val_labels_list.append(batch_labels)

val_images = tf.concat(val_images_list, axis=0)
val_labels = tf.concat(val_labels_list, axis=0)

predictions = model.predict(val_images, verbose=0)
predicted_classes = tf.argmax(predictions, axis=1).numpy()
true_classes = val_labels.numpy()

print("Classification Report:")
print(classification_report(true_classes, predicted_classes,
                            target_names=class_names, digits=3))

# Analiza pewności predykcji
confidences = np.max(predictions, axis=1)
print(f"\nStatystyki pewności predykcji:")
print(f"  Średnia: {np.mean(confidences):.3f}")
print(f"  Mediana: {np.median(confidences):.3f}")
print(f"  Min: {np.min(confidences):.3f}")
print(f"  Max: {np.max(confidences):.3f}")
print(
    f"  % predykcji > 70%: {(confidences > 0.7).sum() / len(confidences) * 100:.1f}%")

# ===== Wykresy =====
# 1. Training history
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, acc, label='Train Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Val Accuracy', linewidth=2)
plt.axvline(x=len(history1.history['accuracy']), color='r',
            linestyle='--', alpha=0.7, label='Fine-tuning starts')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(epochs_range, loss, label='Train Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Val Loss', linewidth=2)
plt.axvline(x=len(history1.history['loss']), color='r',
            linestyle='--', alpha=0.7, label='Fine-tuning starts')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(confidences, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=0.7, color='r', linestyle='--', label='70% threshold')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Prediction Confidence Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Normalizowana macierz (%)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Zwykła macierz
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp1.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax1)
ax1.set_title("Confusion Matrix (Counts)", fontsize=14)

# Znormalizowana
disp2 = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized, display_labels=class_names)
disp2.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax2, values_format='.2f')
ax2.set_title("Confusion Matrix (Normalized)", fontsize=14)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. Sample predictions z rozkładem prawdopodobieństwa
print("\nGenerowanie przykładowych predykcji...\n")
raw_val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=1,
    shuffle=True
)

all_samples = list(raw_val_ds.unbatch().take(200))
random_samples = random.sample(all_samples, min(9, len(all_samples)))

fig = plt.figure(figsize=(15, 12))
for i, (img, label) in enumerate(random_samples):
    img_array = tf.expand_dims(img, axis=0)
    img_processed = preprocess_input(img_array)

    prediction = model.predict(img_processed, verbose=0)[0]
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx] * 100

    # Top 3 predykcje
    top3_idx = prediction.argsort()[-3:][::-1]

    # Subplot dla obrazu
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(img.numpy().astype("uint8"))

    true_label = class_names[label]
    pred_label = class_names[predicted_idx]
    color = 'green' if true_label == pred_label else 'red'

    # Tytuł z top 3
    title = f"True: {true_label}\n"
    title += f"Pred: {pred_label} ({confidence:.1f}%)\n"
    title += f"Top3: "
    for idx in top3_idx[:3]:
        title += f"{class_names[idx][:4]} {prediction[idx]*100:.0f}% "

    plt.title(title, color=color, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("✔ Training zakończony!")
print("="*70)
print(f"Pliki:")
print(f"  - Model: mushrooms_mobilenet_v2.h5")
print(f"  - Klasy: class_names.json")
print(f"  - Wykresy: training_history.png, confusion_matrix.png, sample_predictions.png")
print("="*70)
