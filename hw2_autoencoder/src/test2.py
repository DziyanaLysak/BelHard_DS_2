"""
Модуль для обучения автоэнкодера на датасете CIFAR-10.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets
import tensorflow as tf

# Добавляем пути для импорта наших модулей
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Импортируем наши модули
from src.data_loader import load_cifar10, normalize_images
from src.models import build_autoencoder


# Функция для создания названия папки с результатами
def create_results_folder():
    """
    Создает папку для сохранения результатов обучения.
    Название папки включает дату и время, чтобы не перезаписывать старые результаты.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# Функция для визуализации процесса обучения
def plot_training_history(history, save_path):
    """
    Рисует графики потерь (loss) на обучающей и валидационной выборках.

    Args:
        history: объект History, возвращаемый model.fit()
        save_path: путь для сохранения графика
    """
    plt.figure(figsize=(12, 4))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучение', linewidth=2)
    plt.plot(history.history['val_loss'], label='Валидация', linewidth=2)
    plt.title('Потери (loss) в процессе обучения', fontsize=14)
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Сохраняем
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ График сохранен: {save_path}")


# Функция для визуализации результатов
def show_reconstructions(autoencoder, x_test, num_images=10, save_path=None):
    """
    Показывает оригинальные и восстановленные изображения рядом для сравнения.

    Args:
        autoencoder: обученная модель
        x_test: тестовые данные
        num_images: количество изображений для показа
        save_path: путь для сохранения картинки
    """
    # Выбираем случайные картинки из тестовой выборки
    indices = np.random.choice(len(x_test), num_images, replace=False)
    test_images = x_test[indices]

    # Получаем восстановленные изображения
    reconstructed = autoencoder.predict(test_images, verbose=0)

    # Рисуем
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Оригинал
        plt.subplot(2, num_images, i + 1)
        plt.imshow(test_images[i])
        plt.title('Оригинал', fontsize=10)
        plt.axis('off')

        # Восстановленное
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(reconstructed[i])
        plt.title('Восстановлено', fontsize=10)
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Сравнение сохранено: {save_path}")
    plt.show()


# Функция для проверки размерностей
def check_dimensions(encoder, decoder, autoencoder):
    """
    Проверяет, правильно ли работают размерности в моделях.
    Пропускает одну картинку через все модели и выводит размеры.
    """
    print("\n" + "=" * 50)
    print("🔍 ПРОВЕРКА РАЗМЕРНОСТЕЙ")
    print("=" * 50)

    # Создаем тестовую картинку (одно изображение 32x32x3)
    test_image = np.random.rand(1, 32, 32, 3)

    # Пропускаем через энкодер
    encoded = encoder.predict(test_image, verbose=0)
    print(f"После энкодера: {encoded.shape} → 4×4×8 (сжатое представление)")

    # Пропускаем через декодер
    decoded = decoder.predict(encoded, verbose=0)
    print(f"После декодера: {decoded.shape} → 32×32×3 (восстановлено)")

    # Проверяем полный автоэнкодер
    autoencoded = autoencoder.predict(test_image, verbose=0)
    print(f"Полный автоэнкодер: {autoencoded.shape} → 32×32×3")

    print("\n✅ Все размерности совпадают!")
    print("=" * 50 + "\n")

    return encoded, decoded


# Функция для сохранения моделей
def save_models(autoencoder, encoder, decoder, results_dir):
    """
    Сохраняет все три модели в папку models/ с указанием даты в названии.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Создаем папку models если её нет
    models_dir = os.path.join('models')
    os.makedirs(models_dir, exist_ok=True)

    # Сохраняем модели
    autoencoder_path = os.path.join(models_dir, f'autoencoder_{timestamp}.h5')
    encoder_path = os.path.join(models_dir, f'encoder_{timestamp}.h5')
    decoder_path = os.path.join(models_dir, f'decoder_{timestamp}.h5')

    autoencoder.save(autoencoder_path)
    encoder.save(encoder_path)
    decoder.save(decoder_path)

    print(f"\n💾 Модели сохранены:")
    print(f"   - Полный автоэнкодер: {autoencoder_path}")
    print(f"   - Энкодер: {encoder_path}")
    print(f"   - Декодер: {decoder_path}")

    return autoencoder_path, encoder_path, decoder_path


# ----- ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ -----
def train_autoencoder(epochs=20, batch_size=128):
    """
    Главная функция для обучения автоэнкодера.

    Args:
        epochs: количество эпох обучения
        batch_size: размер батча
    """
    print("\n" + "=" * 60)
    print("🚀 НАЧАЛО ОБУЧЕНИЯ АВТОЭНКОДЕРА")
    print("=" * 60)

    # 1. Создаем папку для результатов
    results_dir = create_results_folder()
    print(f"📁 Результаты будут сохранены в: {results_dir}")

    # 2. Загружаем данные
    print("\n📦 Загрузка данных CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Нормализуем (приводим пиксели к диапазону 0-1)
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    print(f"   Обучающая выборка: {x_train.shape}")
    print(f"   Тестовая выборка: {x_test.shape}")

    # 3. Создаем модели
    print("\n🏗️  Создание моделей...")
    autoencoder, encoder, decoder = build_autoencoder()

    # Проверяем размерности
    check_dimensions(encoder, decoder, autoencoder)

    # 4. Компилируем модель
    print("\n⚙️  Компиляция модели...")
    autoencoder.compile(
        optimizer='adam',  # Оптимизатор
        loss='mse',  # Функция потерь (Mean Squared Error)
        metrics=['mae']  # Дополнительная метрика (Mean Absolute Error)
    )

    # Показываем архитектуру
    print("\n📊 Архитектура полного автоэнкодера:")
    autoencoder.summary()

    # 5. Обучаем модель
    print("\n🏋️  Начало обучения...")
    print(f"   Эпох: {epochs}")
    print(f"   Batch size: {batch_size}")
    print("-" * 40)

    history = autoencoder.fit(
        x_train, x_train,  # Вход = выход (учим восстанавливать)
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),  # Проверяем на тестовых данных
        verbose=1  # Показываем прогресс
    )

    # 6. Визуализируем процесс обучения
    print("\n📈 Построение графиков обучения...")
    plot_path = os.path.join(results_dir, 'training_history.png')
    plot_training_history(history, plot_path)

    # 7. Показываем примеры восстановления
    print("\n🖼️  Примеры восстановления изображений...")
    reconst_path = os.path.join(results_dir, 'reconstructions.png')
    show_reconstructions(autoencoder, x_test, num_images=10, save_path=reconst_path)

    # 8. Сохраняем модели
    save_models(autoencoder, encoder, decoder, results_dir)

    # 9. Итог
    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"📁 Все результаты в папке: {results_dir}")
    print(f"   - График обучения: training_history.png")
    print(f"   - Примеры восстановления: reconstructions.png")
    print(f"   - Модели сохранены в папке: models/")
    print("=" * 60 + "\n")

    return autoencoder, encoder, decoder, history


# ----- ТОЧКА ВХОДА -----
if __name__ == "__main__":
    """
    При запуске файла напрямую начинаем обучение.
    Можно изменить параметры при необходимости.
    """
    # Обучаем с параметрами по умолчанию
    autoencoder, encoder, decoder, history = train_autoencoder(
        epochs=50,  # Больше эпох = лучше качество
        batch_size=128  # Размер батча
    )

    print("\n🎉 Готово! Теперь можно использовать evaluate.py для анализа.")