"""
Скрипт для загрузки MNIST и сохранения в SQLite базу данных.
"""

import sqlite3
import numpy as np
from tensorflow.keras.datasets import mnist
import os

# Константы
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'mnist.db')  # путь к базе данных
IMAGE_SIZE = 28 * 28  # 784 пикселя

# Функция загрузки и подготовки данных
def load_and_prepare_data():
    """
    Загружает MNIST, объединяет train и test, превращает 28x28 в плоский массив.
    Возвращает X (признаки) и y (метки).
    """
    # Загружаем данные (keras сам разделяет на train/test)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Объединяем train и test в один массив (все 70000 картинок)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Превращаем каждую картинку 28x28 в ряд из 784 чисел
    # X.shape был (70000, 28, 28) -> станет (70000, 784)
    n_samples = X.shape[0]
    X = X.reshape(n_samples, IMAGE_SIZE)

    return X, y

# Функция создания базы данных
def create_database(X, y):
    """
    Создаёт SQLite базу и заполняет её данными.
    """
    # Создаём папку data, если её нет
    os.makedirs('data', exist_ok=True)

    # Если база уже есть — удалим (чтобы не было дублей)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Подключаемся к базе
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Создаём таблицу с колонками pixel_1 ... pixel_784 и label
    column_names = [f'pixel_{i}' for i in range(1, IMAGE_SIZE + 1)]
    columns_definition = 'id INTEGER PRIMARY KEY AUTOINCREMENT, ' + \
                         ', '.join([f'{col} INTEGER' for col in column_names]) + \
                         ', label INTEGER'

    cursor.execute(f'CREATE TABLE mnist_data ({columns_definition})')

    # Вставляем данные построчно
    for i in range(X.shape[0]):
        # Берём строку из 784 пикселей и добавляем метку
        row = X[i].tolist() + [int(y[i])]

        # Создаём строку с плейсхолдерами (?, ?, ..., ?)
        placeholders = ','.join(['?' for _ in range(IMAGE_SIZE + 1)])
        cursor.execute(f'INSERT INTO mnist_data ({", ".join(column_names + ["label"])}) VALUES ({placeholders})', row)

    conn.commit()
    conn.close()
    print(f"База данных создана: {DB_PATH}")
    print(f"Загружено {X.shape[0]} изображений")


if __name__ == "__main__":
    print("Загружаем MNIST...")
    X, y = load_and_prepare_data()
    print(f"Загружено {X.shape[0]} изображений. Формат X: {X.shape}, формат y: {y.shape}")

    print("Создаём базу данных...")
    create_database(X, y)
    print("Готово!")