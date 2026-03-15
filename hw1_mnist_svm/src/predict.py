"""
Скрипт для предсказания цифр на реальных картинках.
Загружает модель, обрабатывает картинки из папки test_images и показывает результаты.
"""

import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Константы
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'svm_mnist.pkl')
# Папка с тестовыми картинками
TEST_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_images')

# Функция загрузки модели
def load_model():
    """Загружает сохранённую модель из файла."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print(f"Модель загружена из {MODEL_PATH}")
    return model

# Функция подготовки изображения
def prepare_image(image_path):
    """
    Подготавливает картинку для модели:
    - ч/б
    - размер 28x28
    - превращает в массив 784 чисел
    - нормализует
    """
    # Открываем картинку
    img = Image.open(image_path)
    img_original = img.copy()  # сохраняем оригинал для показа

    # Превращаем в ч/б
    img = img.convert('L')

    # Меняем размер на 28x28
    img = img.resize((28, 28))

    # Превращаем в массив numpy
    img_array = np.array(img)

    # Автоматическая инверсия: делаем так, чтобы цифра была светлее фона
    # Модель обучена на белых цифрах на чёрном фоне
    if np.mean(img_array) > 127:  # если фон светлый
        img_array = 255 - img_array  # инвертируем в тёмный фон, светлую цифру

    # Превращаем в плоский массив из 784 чисел
    img_flat = img_array.reshape(784)

    # Нормализуем
    img_normalized = img_flat / 255.0

    return img_normalized, img_original


# Функция предсказания для одной картинки
def predict_single(model, image_array):
    """Принимает модель и массив из 784 чисел, возвращает предсказанную цифру."""
    prediction = model.predict([image_array])[0]
    return prediction


#  Основная программа
if __name__ == "__main__":

    print("ЗАГРУЗКА МОДЕЛИ...")

    # Загружаем модель
    model = load_model()

    # Ищем все картинки в папке test_images
    image_files = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.png')) + \
                  glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))

    if not image_files:
        print(f"В папке {TEST_IMAGES_DIR} нет PNG или JPG файлов")
        print("Создай папку и добавь туда картинки с цифрами")
        exit()

    print(f"\nНайдено картинок: {len(image_files)}")

    # Собираем результаты
    results = []
    for img_file in image_files:
        img_array, img_original = prepare_image(img_file)
        pred = predict_single(model, img_array)

        # Пытаемся извлечь истинную цифру из имени файла
        name = os.path.basename(img_file)
        true_label = None
        digits = [int(c) for c in name if c.isdigit()]
        if digits:
            true_label = digits[0]

        results.append((img_original, pred, true_label, img_file))

    # Итоговая статистика
    correct = sum(1 for _, pred, true, _ in results if true is not None and pred == true)
    total_with_true = sum(1 for _, _, true, _ in results if true is not None)
    if total_with_true > 0:
        accuracy = (correct / total_with_true * 100) if total_with_true > 0 else 0
        print(f"\nТочность на {total_with_true} картинках: {correct}/{total_with_true} = {accuracy:.1f}%")


    # Визуализация результатов
    print("\nОтображаем результаты...")

    n_images = len(results)
    cols = 5
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, (img, pred, true_label, img_file) in enumerate(results):
        # Показываем картинку
        axes[i].imshow(img, cmap='gray')

        # Определяем цвет и текст
        if true_label is not None:
            color = 'green' if pred == true_label else 'red'
            title = f'И: {true_label}\nП: {pred}'
        else:
            color = 'black'
            title = f'Пред: {pred}'

        axes[i].set_title(title, color=color, fontsize=12)
        axes[i].axis('off')

        # Имя файла мелко внизу
        axes[i].text(0.5, -0.1, os.path.basename(img_file),
                     transform=axes[i].transAxes, fontsize=8, ha='center')

    # Прячем лишние оси
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Результаты распознавания реальных цифр', fontsize=16)
    plt.tight_layout()
    plt.show()

    print("\nГотово!")