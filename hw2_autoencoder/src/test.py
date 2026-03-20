# tr модель 2-3                                                                                          _
 Функция создания энкодера
def build_encoder(input_shape=(32, 32, 3)):
    """
    Создает encoder - часть, которая сжимает изображение.
    На входе: 32x32x3 (цветная картинка)
    На выходе: сжатое представление (8x8x16) - меньше сжатие!
    """
    encoder = models.Sequential(name='encoder')

    # Первый сверточный слой: 32x32x3 -> 16x16x32
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))

    # Второй сверточный слой: 16x16x32 -> 8x8x16
    encoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))

    # Третий сверточный слой: 8x8x16 -> 8x8x16 (УБИРАЕМ MaxPooling!)
    encoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    # MaxPooling2D удалили - размер остается 8×8

    return encoder

# Функция создания декодера
def build_decoder():
    """
    Создает decoder - часть, которая восстанавливает изображение.
    На входе: сжатое представление (8x8x16)
    На выходе: 32x32x3 (восстановленная картинка)
    """
    decoder = models.Sequential(name='decoder')


    # # Первый слой: 4x4x8 -> 8x8x16
    # decoder.add(layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same'))

    # Первый слой: 8×8×16 -> 16×16×32
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'))
    # strides=2 увеличит с 8 до 16

    # # Второй слой: 8x8x16 -> 16x16x32
    # decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'))

    # Второй слой: 16x16x32 -> 32x32x3
    decoder.add(layers.Conv2DTranspose(3, (3, 3), strides=2, activation='sigmoid', padding='same'))

    # # Третий слой: 16x16x32 -> 32x32x3
    # decoder.add(layers.Conv2DTranspose(3, (3, 3), strides=2, activation='sigmoid', padding='same'))

    return decoder

# tr модель 4                                                                                             _

def build_encoder(input_shape=(32, 32, 3)):
    encoder = models.Sequential(name='encoder')

    # 32x32x3 -> 16x16x32
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))

    # 16x16x32 -> 16x16x16 (Conv без пулинга!)
    encoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    # Размер НЕ меняется!

    return encoder  # Выход: 16×16×16


def build_decoder():
    decoder = models.Sequential(name='decoder')

    # Вход: 16×16×16 (от энкодера)

    # Слой 1: 16×16×16 -> 32×32×32
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'))
    # strides=2 увеличит с 16 до 32!

    # Слой 2: 32×32×32 -> 32×32×3
    decoder.add(layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same'))
    # Без strides=2 — размер не меняем, только каналы

    return decoder

#--------------------------------------------------------------------------------------
# Функция сборки автоэнкодера
def build_autoencoder():
    """
    Собирает полный автоэнкодер: encoder + decoder.
    Возвращает три модели: полный автоэнкодер, отдельно энкодер, отдельно декодер.
    Это нужно чтобы:
    - полный автоэнкoder использовать для обучения
    - энкодер отдельно — для сжатия картинок
    - декодер отдельно — для восстановления из сжатого представления
    """
    encoder = build_encoder()
    decoder = build_decoder()

    # Входные данные -> encoder -> decoder -> выход
    autoencoder = models.Sequential([encoder, decoder], name='autoencoder')

    return autoencoder, encoder, decoder


if __name__ == "__main__":
    # Проверяем, что модель создается без ошибок
    # Если запустить этот файл напрямую, то увидим архитектуру:
    # сколько слоев, какие размеры, сколько параметров
    # Это помогает убедиться, что все слои соединены правильно
    autoencoder, encoder, decoder = build_autoencoder()

    print("Encoder:")
    encoder.summary()
    print("\nDecoder:")
    decoder.summary()
    print("\nПолный автоэнкодер:")
    autoencoder.summary()