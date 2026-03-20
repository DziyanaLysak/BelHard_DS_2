"""
Модуль для загрузки и подготовки датасета CIFAR-10.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets


# Функция загрузки данных
def load_cifar10():
    """
    Загружает датасет CIFAR-10.
    """
    # Загружаем данные. keras скачивает данные в папку .keras/datasets/
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    print(f"Размер обучающей выборки: {x_train.shape}")   # (50000, 32, 32, 3)
    print(f"Размер тестовой выборки: {x_test.shape}")   # (10000, 32, 32, 3)
    print(f"Минимальное значение пикселя до нормализации: {x_train.min()}")
    print(f"Максимальное значение пикселя до нормализации: {x_train.max()}")

    return (x_train, y_train), (x_test, y_test)

# Функция нормализации
def normalize_images(images):
    """
    Нормализует значения пикселей в диапазон [0, 1].
    """
    return images.astype('float32') / 255.0

# Функция визуализации
def show_sample_images(x_train, y_train, num_samples=5):
    """
    Показывает несколько примеров изображений из датасета,
    чтобы убедиться, что данные загрузились правильно.

    Args:
        x_train: массив изображений
        y_train: массив меток
        num_samples: количество изображений для показа
    """
    # Названия классов в CIFAR-10 (по порядку от 0 до 9)
    class_names = ['самолет', 'автомобиль', 'птица', 'кошка', 'олень',
                   'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

    # Создаём картинку с несколькими графиками в ряд
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_train[i])  # показываем картинку
        plt.title(class_names[y_train[i][0]])  # подписываем класс
        plt.axis('off')  # убираем оси
    plt.show()


if __name__ == "__main__":
    # Тестовый запуск
    print("Загрузка CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Нормализуем данные
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    print(f"После нормализации - min: {x_train.min()}, max: {x_train.max()}")
    print("Загрузка завершена!")

    # Показываем примеры изображений
    show_sample_images(x_train, y_train)
