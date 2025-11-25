# model_runner.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps  # <-- ZMIANA: Importujemy ImageOps
import numpy as np
import io
import os

# === SEKCJA KONFIGURACJI ===
print("--- Ładowanie modelu: grzybopedia_saved_model.h5 ---")
BASE_DIR = os.path.dirname(os.path.abspath(
    __file__))  # folder of model_runner.py
model_path = os.path.join(BASE_DIR, 'mushrooms_mobilenet_v2.h5')
model = load_model(model_path)
print("--- Model załadowany pomyślnie ---")

CLASS_NAMES_EN = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius',
                  'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']
CLASS_NAMES_PL = {
    'Agaricus': 'Pieczarka', 'Amanita': 'Muchomor', 'Boletus': 'Borowik',
    'Cortinarius': 'Zasłonak', 'Entoloma': 'Dzwonkówka', 'Hygrocybe': 'Wilgotnica',
    'Lactarius': 'Mleczaj', 'Russula': 'Gołąbek', 'Suillus': 'Maślak'
}

# === GŁÓWNA FUNKCJA ===


def predict_mushroom(image_bytes):
    """
    Przyjmuje obraz w formie bajtów, przetwarza go i zwraca
    listę 5 najlepszych predykcji.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # === KLUCZOWA ZMIANA: Automatycznie obracamy obraz zgodnie z metadanymi EXIF ===
    img = ImageOps.exif_transpose(img)

    # Konwersję na RGB robimy PO ewentualnym obróceniu
    img = img.convert('RGB')

    # Przetwarzanie obrazu
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predykcja
    predictions = model.predict(img_array)[0]

    # Przygotowanie 5 najlepszych wyników
    top_indices = predictions.argsort()[-5:][::-1]

    results = []
    for i in top_indices:
        class_name_en = CLASS_NAMES_EN[i]
        results.append({
            "classNameEN": class_name_en,
            "classNamePL": CLASS_NAMES_PL.get(class_name_en, class_name_en),
            "confidence": float(predictions[i])
        })

    return results
