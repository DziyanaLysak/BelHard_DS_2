"""
Скрипт для обучения SVM на данных MNIST из SQLite базы.
"""

import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Константы
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'mnist.db')
TEST_SIZE = 0.15  # 15% данных оставим для теста
RANDOM_STATE = 42  # для воспроизводимости (чтобы при каждом запуске разделение было одинаковым)
MODEL_PATH = os.path.join('models', 'svm_mnist.pkl')  # куда сохраним обученную модель

# Функция загрузки данных
def load_data_from_db():
    """
    Загружает данные из SQLite базы.
    Возвращает X (признаки) и y (метки).
    """
    # Подключаемся к базе
    conn = sqlite3.connect(DB_PATH)

    # Читаем все данные из таблицы
    query = "SELECT * FROM mnist_data"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Извлекаем признаки и метки:
    # - id (колонка 0) нам не нужен
    # - пиксели (колонки с 1 по предпоследнюю)
    # - label (последняя колонка)
    X = df.iloc[:, 1:-1].values  # все строки, колонки с 1 по предпоследнюю
    y = df.iloc[:, -1].values    # все строки, последняя колонка

    X = X / 255.0  # нормализация: приводим пиксели из диапазона 0-255 к 0-1

    return X, y

#  Основная программа
if __name__ == "__main__":
    print("Загружаем данные из базы...")
    X, y = load_data_from_db()
    print(f"Загружено {X.shape[0]} изображений")
    print(f"Размер X: {X.shape}, размер y: {y.shape}")
    print(f"Минимальное значение пикселя: {X.min()}, максимальное: {X.max()}")

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # чтобы пропорции цифр сохранились
    )

    print(f"Обучающая выборка: {X_train.shape[0]} картинок")
    print(f"Тестовая выборка: {X_test.shape[0]} картинок")

    # Создаём и обучаем модель SVM
    print("Обучаем модель SVM...")
    model = SVC(kernel='rbf', gamma='scale', verbose=True)
    model.fit(X_train, y_train)

    # Предсказываем на тестовых данных
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy:.4f}")

    # 1. Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Предсказанная цифра')
    plt.ylabel('Истинная цифра')
    plt.title('Матрица ошибок')
    plt.tight_layout()
    plt.show()

    # 2. Примеры цифр из тестовой выборки
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(10):
        # Берём первую картинку для каждой цифры (для примера)
        idx = np.where(y_test == i)[0]
        if len(idx) > 0:
            img = X_test[idx[0]].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Цифра: {i}')
        axes[i].axis('off')

    plt.suptitle('Примеры цифр из тестовой выборки', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 3. Несколько случайных картинок с предсказаниями
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.ravel()

    # Берём случайные индексы
    np.random.seed(42)
    indices = np.random.choice(len(X_test), 15, replace=False)

    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        # Зелёный если угадала, красный если ошиблась
        color = 'green' if y_pred[idx] == y_test[idx] else 'red'
        axes[i].set_title(f'И: {y_test[idx]} П: {y_pred[idx]}', color=color)
        axes[i].axis('off')

    plt.suptitle('Предсказания на тестовых картинках (зелёный - верно, красный - ошибка)', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Сохраняем модель
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")

    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")

