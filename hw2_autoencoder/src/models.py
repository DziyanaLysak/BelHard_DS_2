"""
Модуль с архитектурой автоэнкодера.
"""

from tensorflow.keras import layers, models


# Функция создания энкодера (глубокая версия)
def build_encoder(input_shape=(32, 32, 3)):
    """
    Создает encoder с увеличенной глубиной.

    Архитектура:
    1. 32×32×3  → Conv2D(32) + Pooling → 16×16×32  (ищем базовые признаки)
    2. 16×16×32 → Conv2D(32) + Pooling → 8×8×32    (ищем сложные признаки)
    3. 8×8×32   → Conv2D(32) (без Pooling) → 8×8×32 (углубляем понимание)

    Returns:
        encoder: модель для сжатия изображений
    """
    encoder = models.Sequential(name='encoder')

    # Слой 1: Первичная обработка
    # Вход: 32×32×3 (RGB картинка)
    # Выход: 16×16×32 (32 признака, размер уменьшен вдвое)
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='conv_1'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same', name='pool_1' ))

    # Слой 2: Поиск сложных признаков
    # Вход: 16×16×32
    # Выход: 8×8×32 (снова уменьшаем размер, сохраняем глубину)
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same', name='pool_2'))

    # Слой 3: Углубление признаков (без сжатия размера)
    # Вход: 8×8×32
    # Выход: 8×8×32 (размер тот же, но признаки стали "глубже")
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_3'))
    # Здесь НЕТ пулинга - сохраняем размер 8×8

    return encoder


# Функция создания декодера (под глубокий энкодер)
def build_decoder():
    """
    Создает decoder для восстановления из 8×8×32.

    Архитектура (зеркально энкодеру):
    1. 8×8×32  → Conv2DTranspose → 16×16×32  (увеличиваем размер)
    2. 16×16×32 → Conv2DTranspose → 32×32×32  (еще увеличиваем)
    3. 32×32×32 → Conv2DTranspose → 32×32×3   (получаем RGB)

    Returns:
        decoder: модель для восстановления изображений
    """
    decoder = models.Sequential(name='decoder')

    # Слой 1: Первое увеличение
    # Вход: 8×8×32 (сжатое представление)
    # Выход: 16×16×32 (увеличили размер в 2 раза)
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same', name='deconv_1'))

    # Слой 2: Второе увеличение
    # Вход: 16×16×32
    # Выход: 32×32×32 (достигли исходного размера)
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same', name='deconv_2'))

    # Слой 3: Восстановление цвета
    # Вход: 32×32×32
    # Выход: 32×32×3 (превращаем 32 признака в RGB каналы)
    decoder.add(layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='deconv_3'))

    return decoder


# Функция сборки полного автоэнкодера
def build_autoencoder():
    """
    Собирает полный автоэнкодер из энкодера и декодера.

    Returns:
        autoencoder: полная модель для обучения
        encoder: отдельно энкодер (для сжатия)
        decoder: отдельно декодер (для восстановления)
    """
    encoder = build_encoder()
    decoder = build_decoder()

    # Склеиваем энкодер и декодер в одну последовательную модель
    autoencoder = models.Sequential([encoder, decoder], name='autoencoder')

    return autoencoder, encoder, decoder


if __name__ == "__main__":
    """
    При запуске файла напрямую показываем архитектуру моделей.
    Это помогает проверить, что все размерности совпадают.
    """
    # Создаем модели
    autoencoder, encoder, decoder = build_autoencoder()

    # Выводим архитектуру
    print("Архитектура энкодера:")
    encoder.summary()
    print("\nАрхитектура декодера:")
    decoder.summary()
    print("\nАрхитектура автоэнкодера:")
    autoencoder.summary()

    # Дополнительная информация
    print("\nХарактеристики модели:")

    # Считаем степень сжатия
    input_size = 32 * 32 * 3  # 3072
    encoded_size = 8 * 8 * 32  # 2048
    compression_ratio = input_size / encoded_size

    print(f"Входное изображение: 32×32×3 = {input_size} чисел")
    print(f"Сжатое представление: 8×8×32 = {encoded_size} чисел")
    print(f"Коэффициент сжатия: {compression_ratio:.2f} (сжатие в {compression_ratio:.1f} раз)")

    if compression_ratio > 1:
        print("✅ Это настоящее сжатие!")
    else:
        print("⚠️  Внимание: это не сжатие, а раздутие!")
