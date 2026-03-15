"""
Веб-сервер для распознавания рукописных цифр.
Запуск: uvicorn src.app:app --reload
"""

from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import io
import joblib
import os

# Константы
# Путь к модели (универсальный)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'svm_mnist.pkl')

# Загружаем модель при старте сервера
model = joblib.load(MODEL_PATH)
print(f"✅ Модель загружена из {MODEL_PATH}")

# Создаём приложение
app = FastAPI(title="MNIST Classifier",
              description="Распознавание рукописных цифр методом опорных векторов")

# Функция подготовки изображения для веб-сервера
def prepare_image(image_bytes):
    """
    Подготавливает картинку из байтов для модели
    """
    # Открываем картинку из байтов
    img = Image.open(io.BytesIO(image_bytes))

    # Превращаем в ч/б
    img = img.convert('L')

    # Меняем размер на 28x28
    img = img.resize((28, 28))

    # Превращаем в массив numpy
    img_array = np.array(img)

    # Автоматическая инверсия (делаем цифру светлее фона)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Превращаем в плоский массив из 784 чисел и нормализуем
    img_flat = img_array.reshape(784) / 255.0

    return img_flat

# Маршруты
@app.get("/")
def root():
    return {
        "message": "MNIST classifier API",
        "usage": "Отправь POST запрос на /predict с картинкой"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Загрузи картинку с цифрой, получи предсказание
    """
    try:
        # Читаем загруженный файл
        contents = await file.read()

        # Подготавливаем картинку
        img_array = prepare_image(contents)

        # Предсказываем
        prediction = model.predict([img_array])[0]

        return {
            "filename": file.filename,
            "prediction": int(prediction),
            "status": "success"
        }
    except Exception as e:
        return {
            "filename": file.filename,
            "error": str(e),
            "status": "error"
        }

# Запуск
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)