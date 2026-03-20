"""
Утилиты для автоэнкодера: визуализация, метрики, подготовка данных.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras import models
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)



# Функция 1: Визуализация скрытого пространства (latent space)
def plot_latent_space(encoder, x_test, y_test, save_path=None):
    """
    Визуализирует, как энкодер распределяет картинки в сжатом пространстве.
    Использует t-SNE для уменьшения размерности до 2D.

    Args:
        encoder: обученная модель энкодера
        x_test: тестовые картинки
        y_test: метки классов (0-9 для CIFAR-10)
        save_path: путь для сохранения графика
    """
    print("Сжимаем картинки энкодером...")
    # Пропускаем тестовые картинки через энкодер
    encoded_imgs = encoder.predict(x_test[:1000], verbose=0)

    # Уменьшаем размерность до 2D с помощью t-SNE
    print("Уменьшаем размерность t-SNE...")
    encoded_flat = encoded_imgs.reshape(len(encoded_imgs), -1)
    tsne = TSNE(n_components=2, random_state=42)
    encoded_2d = tsne.fit_transform(encoded_flat)

    # Рисуем
    plt.figure(figsize=(12, 8))

    # Названия классов для CIFAR-10
    class_names = ['самолет', 'автомобиль', 'птица', 'кошка', 'олень',
                   'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

    # Раскрашиваем по классам
    scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1],
                          c=y_test[:1000].flatten(), cmap='tab10', alpha=0.6, s=30)

    # Подписываем цвета названиями классов
    cbar = plt.colorbar(scatter, ticks=range(10), label='Класс')
    cbar.ax.set_yticklabels(class_names)
    cbar.ax.tick_params(labelsize=9)

    plt.title('Скрытое пространство (latent space) энкодера', fontsize=14)
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ График latent space сохранен: {save_path}")

    plt.show()

    return encoded_2d


# Функция 2: Расчет PSNR (пиковое отношение сигнала к шуму)
def calculate_psnr_score(original, reconstructed):
    """
    Считает PSNR между оригиналом и восстановленной картинкой.
    Чем выше PSNR, тем лучше качество (типично 20-40 dB).

    Args:
        original: оригинальное изображение
        reconstructed: восстановленное изображение

    Returns:
        psnr_value: значение PSNR в dB
    """
    # PSNR требует значения в [0, 255], а у нас [0, 1]
    orig_255 = (original * 255).astype(np.uint8)
    recon_255 = (reconstructed * 255).astype(np.uint8)

    # PSNR для цветных картинок (multichannel=True)
    psnr_value = psnr(orig_255, recon_255, data_range=255)
    return psnr_value


# Функция 3: Расчет SSIM (структурное сходство)
def calculate_ssim_score(original, reconstructed):
    """
    Считает SSIM между оригиналом и восстановленной картинкой.
    1.0 = идеально, 0.0 = никакого сходства.

    Args:
        original: оригинальное изображение
        reconstructed: восстановленное изображение

    Returns:
        ssim_value: значение SSIM
    """
    # SSIM требует значения в [0, 255] и win_size нечетный
    orig_255 = (original * 255).astype(np.uint8)
    recon_255 = (reconstructed * 255).astype(np.uint8)

    # Для цветных картинок используем channel_axis=-1
    ssim_value = ssim(orig_255, recon_255,
                      channel_axis=-1,
                      data_range=255,
                      win_size=3)  # маленькое окно для 32x32

    return ssim_value


# Функция 4: Сравнение всех моделей по метрикам
def compare_models_metrics(models_dict, x_test, y_test, save_path=None):
    """
    Сравнивает все модели по PSNR и SSIM.

    Args:
        models_dict: словарь {название_модели: модель}
        x_test: тестовые картинки
        y_test: метки (не используются, но для совместимости)
        save_path: путь для сохранения графика
    """
    results = {}

    for name, autoencoder in models_dict.items():
        print(f"Оцениваем модель: {name}")

        # Восстанавливаем картинки
        reconstructed = autoencoder.predict(x_test[:100], verbose=0)

        # Считаем метрики
        psnr_scores = []
        ssim_scores = []

        for i in range(len(x_test[:100])):
            psnr_val = calculate_psnr_score(x_test[i], reconstructed[i])
            ssim_val = calculate_ssim_score(x_test[i], reconstructed[i])
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

        results[name] = {
            'PSNR_mean': np.mean(psnr_scores),
            'PSNR_std': np.std(psnr_scores),
            'SSIM_mean': np.mean(ssim_scores),
            'SSIM_std': np.std(ssim_scores)
        }

    # Выводим таблицу
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ ПО МЕТРИКАМ")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  PSNR: {metrics['PSNR_mean']:.2f} ± {metrics['PSNR_std']:.2f} dB")
        print(f"  SSIM: {metrics['SSIM_mean']:.3f} ± {metrics['SSIM_std']:.3f}")

    # Рисуем график
    if save_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        names = list(results.keys())
        psnr_means = [results[n]['PSNR_mean'] for n in names]
        psnr_stds = [results[n]['PSNR_std'] for n in names]
        ssim_means = [results[n]['SSIM_mean'] for n in names]
        ssim_stds = [results[n]['SSIM_std'] for n in names]

        ax1.bar(names, psnr_means, yerr=psnr_stds, capsize=5, alpha=0.7)
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('PSNR по моделям')
        ax1.tick_params(axis='x', rotation=45)

        ax2.bar(names, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7, color='green')
        ax2.set_ylabel('SSIM')
        ax2.set_title('SSIM по моделям')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ График сравнения сохранен: {save_path}")

    return results


# Функция 5: Загрузка всех моделей из папки models
def load_all_models(models_dir=None):
    """
    Загружает все сохраненные модели из папки models/.

    Returns:
        dict: словарь {имя_файла: модель}
    """
    if models_dir is None:
        models_dir = os.path.join(BASE_DIR, 'models')

    models_dict = {}

    if os.path.exists(models_dir):
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.h5') and model_file.startswith('autoencoder_'):
                model_path = os.path.join(models_dir, model_file)
                try:
                    model = models.load_model(model_path, compile=False)
                    # Используем имя файла без расширения как ключ
                    name = model_file.replace('.h5', '')
                    models_dict[name] = model
                    print(f"✅ Загружена: {name}")
                except Exception as e:
                    print(f"❌ Ошибка загрузки {model_file}: {e}")

    return models_dict


if __name__ == "__main__":
    print("Тестирование утилит...")

    # 1. Загружаем тестовые данные CIFAR-10
    from tensorflow.keras import datasets

    print("Загружаем тестовые данные...")
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    x_test = x_test / 255.0  # нормализация

    # 2. Берем первую картинку для теста
    img = x_test[0]
    img_reconstructed = img * 0.9 + 0.05  # имитация восстановления (чуть хуже)

    # 3. Считаем PSNR и SSIM
    print("\nТЕСТ МЕТРИК:")
    psnr_val = calculate_psnr_score(img, img_reconstructed)
    ssim_val = calculate_ssim_score(img, img_reconstructed)
    print(f"PSNR: {psnr_val:.2f} dB (чем выше, тем лучше)")
    print(f"SSIM: {ssim_val:.3f} (1.0 = идеально)")

    # 4. Показываем картинки
    print("\nВИЗУАЛИЗАЦИЯ:")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Оригинал', fontsize=12)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_reconstructed)
    plt.title(f'Восстановлено', fontsize=12)
    plt.xlabel(f'PSNR={psnr_val:.1f} dB, SSIM={ssim_val:.3f}', fontsize=10)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    diff = np.abs(img - img_reconstructed)
    diff_enhanced = diff * 3  # усиливаем, чтобы было видно
    plt.imshow(diff_enhanced, cmap='hot')
    plt.title('Разница', fontsize=12)
    plt.axis('off')

    plt.suptitle('Демонстрация работы метрик PSNR и SSIM', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

    # 5. Latent space с лучшей моделью
    print("\nВИЗУАЛИЗАЦИЯ LATENT SPACE:")
    print("Загружаем обученную модель...")

    try:
        import os
        from tensorflow.keras import models

        models_dir = os.path.join(BASE_DIR, 'models')
        best_model_path = None

        # Ищем любую autoencoder модель
        for f in os.listdir(models_dir):
            if f.startswith('autoencoder_') and f.endswith('.h5'):
                best_model_path = os.path.join(models_dir, f)
                break

        if best_model_path:
            print(f"Найдена модель: {best_model_path}")
            autoencoder = models.load_model(best_model_path, compile=False)
            encoder = autoencoder.layers[0]
            encoder.compile(optimizer='adam', loss='mse')

            print("Модель загружена! Рисуем график...")
            plot_latent_space(encoder, x_test[:300], y_test[:300])
        else:
            print("Модель не найдена. Пропускаем визуализацию latent space.")

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")