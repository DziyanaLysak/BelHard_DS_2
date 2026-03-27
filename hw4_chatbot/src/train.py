"""
train.py — обучение нейросети для распознавания намерений.

Этот скрипт:
1. Загружает intents.json
2. Обрабатывает текст (токенизация, стемминг, bag-of-words)
3. Создаёт и обучает нейросеть
4. Сохраняет модель и графики обучения
5. Логирует все метрики в текстовый файл
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split

# Импортируем наши модули
from src.nltk_utils import tokenize, stem, bag_of_words
from src.model import NeuralNet
from src.utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, CURVES_DIR, BASE_DIR, ensure_dirs

# ==================== 1. ПОДГОТОВКА ПАПОК ====================
# Создаём все необходимые папки
ensure_dirs()

# ==================== 2. ЗАГРУЗКА ДАННЫХ ====================
# Путь к файлу с намерениями
intents_path = DATA_DIR / "intents.json"

print("=" * 60)
print("ОБУЧЕНИЕ НЕЙРОСЕТИ ДЛЯ ЧАТ-БОТА")
print("=" * 60)
print(f"Загрузка данных из: {intents_path}")

with open(intents_path, 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Выводим информацию о загруженных намерениях
print(f"\n📊 Загружено намерений: {len(intents['intents'])}")
for intent in intents['intents']:
    print(f"  - {intent['tag']}: {len(intent['patterns'])} паттернов")

# ==================== 3. ОБРАБОТКА ДАННЫХ ====================
# Списки для хранения:
# all_words — все слова из всех паттернов (после стемминга)
# tags — все уникальные теги (намерения)
# xy — список пар (токены_паттерна, тег)
all_words = []
tags = []
xy = []

# Проходим по каждому намерению
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        # Токенизируем паттерн (разбиваем на слова)
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))

# Символы, которые нужно игнорировать (знаки препинания)
ignore_words = ['?', '!', '.', ',', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}']

# Применяем стемминг ко всем словам и удаляем игнорируемые символы
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Убираем дубликаты и сортируем для воспроизводимости
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"\n📚 Статистика:")
print(f"  - Уникальных слов: {len(all_words)}")
print(f"  - Намерений: {len(tags)}")
print(f"  - Намерения: {', '.join(tags)}")

# ==================== 4. СОЗДАНИЕ BAG-OF-WORDS ВЕКТОРОВ ====================
# X — входные векторы (bag-of-words)
# y — метки классов (индексы тегов)
X = []
y = []

for (pattern_words, tag) in xy:
    # Создаём bag-of-words вектор для паттерна
    bag = bag_of_words(pattern_words, all_words)
    X.append(bag)

    # Метка — индекс тега в списке tags
    label = tags.index(tag)
    y.append(label)

# Превращаем списки в numpy массивы
X = np.array(X)
y = np.array(y)

print(f"\n📐 Размерность данных:")
print(f"  - Всего примеров: {X.shape[0]}")
print(f"  - Размер входного вектора: {X.shape[1]}")

# ==================== 5. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ВАЛИДАЦИОННУЮ ВЫБОРКИ ====================
# stratify=y — сохраняем пропорции классов в обеих выборках
# random_state=42 — для воспроизводимости результатов
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Разделение данных:")
print(f"  - Обучающая выборка: {len(X_train)} примеров")
print(f"  - Валидационная выборка: {len(X_val)} примеров")


# ==================== 6. КЛАСС DATASET ДЛЯ PYTORCH ====================
class ChatDataset(Dataset):
    """
    Класс для упаковки данных в формат, понятный PyTorch.
    Нужен для использования DataLoader.
    """

    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        """Возвращает одну пару (вход, метка) по индексу"""
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """Возвращает размер выборки"""
        return self.n_samples


# ==================== 7. ГИПЕРПАРАМЕТРЫ ====================
# Эти параметры можно менять для экспериментов
batch_size = 16  # Количество примеров в одном батче
hidden_size = 128  # Количество нейронов в скрытых слоях
output_size = len(tags)  # Количество классов = количество намерений
input_size = len(X_train[0])  # Размер входного вектора
learning_rate = 0.001  # Скорость обучения
num_epochs = 500  # Максимальное количество эпох. Можно оставить, early stopping сработает раньше
early_stopping_patience = 50  # Сколько эпох ждать улучшения
dropout = 0.3  # Вероятность отключения нейрона (регуляризация)

weight_decay = 1e-5   # добавили в оптимизатор

print(f"\n⚙️ Гиперпараметры:")
print(f"  - Размер батча: {batch_size}")
print(f"  - Скрытый слой: {hidden_size} нейронов")
print(f"  - Скорость обучения: {learning_rate}")
print(f"  - Максимум эпох: {num_epochs}")
print(f"  - Early stopping: {early_stopping_patience} эпох")
print(f"  - Dropout: {dropout}")

# ==================== 8. СОЗДАНИЕ DATALOADER ====================
# DataLoader умеет подавать данные батчами и перемешивать их
train_dataset = ChatDataset(X_train, y_train)
val_dataset = ChatDataset(X_val, y_val)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Перемешиваем для лучшего обучения
    num_workers=0  # Для Windows нужно 0
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,  # На валидации перемешивание не нужно
    num_workers=0
)

# ==================== 9. СОЗДАНИЕ МОДЕЛИ ====================
# Определяем устройство (GPU если есть, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n💻 Устройство: {device}")

# Создаём модель и переносим на устройство
model = NeuralNet(input_size, hidden_size, output_size, dropout=dropout).to(device)

# Функция потерь (кросс-энтропия — для многоклассовой классификации)
criterion = nn.CrossEntropyLoss()

# Оптимизатор Adam (адаптивный метод, хорошо работает на практике)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# ==================== 10. ОБУЧЕНИЕ ====================
print("\n🚀 Начинаем обучение...")
print("-" * 60)

# Списки для сохранения метрик
train_losses = []  # Потери на обучении
val_losses = []  # Потери на валидации
val_accuracies = []  # Точность на валидации

# Для early stopping
best_val_loss = float('inf')  # Лучшая потеря на валидации
patience_counter = 0  # Счётчик эпох без улучшения
best_model_state = None  # Состояние лучшей модели

for epoch in range(num_epochs):
    # ========== ФАЗА ОБУЧЕНИЯ ==========
    model.train()  # Переводим модель в режим обучения (dropout работает)
    epoch_train_loss = 0
    num_train_batches = 0

    for words, labels in train_loader:
        # Переносим данные на устройство
        words = words.to(device)
        labels = labels.to(device).long()  # метки должны быть целыми числами

        # Прямой проход: получаем предсказания
        outputs = model(words)

        # Вычисляем ошибку (loss)
        loss = criterion(outputs, labels)
        epoch_train_loss += loss.item()
        num_train_batches += 1

        # Обратный проход и обновление весов
        optimizer.zero_grad()  # Обнуляем градиенты
        loss.backward()  # Вычисляем градиенты
        optimizer.step()  # Обновляем веса

    avg_train_loss = epoch_train_loss / num_train_batches
    train_losses.append(avg_train_loss)

    # ========== ФАЗА ВАЛИДАЦИИ ==========
    model.eval()  # Переводим модель в режим оценки (dropout выключается)
    epoch_val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    num_val_batches = 0

    with torch.no_grad():  # Отключаем вычисление градиентов (экономия памяти)
        for words, labels in val_loader:
            words = words.to(device)
            labels = labels.to(device).long()

            outputs = model(words)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()
            num_val_batches += 1

            # Подсчёт точности
            _, predicted = torch.max(outputs, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_val_loss = epoch_val_loss / num_val_batches
    val_losses.append(avg_val_loss)
    val_accuracy = correct_predictions / total_predictions
    val_accuracies.append(val_accuracy)

    # ========== EARLY STOPPING ==========
    # Если потеря на валидации улучшилась — сохраняем модель
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()  # Копируем веса
    else:
        patience_counter += 1

    # Выводим информацию каждые 20 эпох
    if (epoch + 1) % 20 == 0:
        print(f'Эпоха [{epoch + 1:4d}/{num_epochs}] | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val Acc: {val_accuracy:.4f}')

    # Останавливаем обучение, если нет улучшений
    if patience_counter >= early_stopping_patience:
        print(f"\n⏹️ Early stopping на эпохе {epoch + 1}")
        break

# Загружаем лучшую модель
model.load_state_dict(best_model_state)

print("\n" + "-" * 60)
print(f"✅ Обучение завершено!")
print(f"  - Лучший loss на валидации: {best_val_loss:.4f}")
print(f"  - Лучшая точность на валидации: {max(val_accuracies):.4f}")

# ==================== 11. СОХРАНЕНИЕ МОДЕЛИ ====================
# Данные, которые нужно сохранить вместе с моделью
data = {
    "model_state": model.state_dict(),  # Веса модели
    "input_size": input_size,  # Размер входного вектора
    "hidden_size": hidden_size,  # Размер скрытого слоя
    "output_size": output_size,  # Количество классов
    "all_words": all_words,  # Словарь всех слов
    "tags": tags  # Список намерений
}

# Сохраняем модель в папку models с временной меткой
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = MODELS_DIR / f"chatbot_{timestamp}.pth"
torch.save(data, model_path)
print(f"\n💾 Модель сохранена в: {model_path}")

# Сохраняем также как data.pth в корень (для совместимости с оригиналом)
compat_path = BASE_DIR / "data.pth"
torch.save(data, compat_path)
print(f"💾 Модель (для совместимости) сохранена в: {compat_path}")

# ==================== 12. СОХРАНЕНИЕ ГРАФИКОВ ====================
# Создаём фигуру с двумя подграфиками
plt.figure(figsize=(12, 5))

# График 1: Loss (ошибка) на обучении и валидации
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses,
         label='Train Loss', alpha=0.7, linewidth=2)
plt.plot(range(1, len(val_losses) + 1), val_losses,
         label='Validation Loss', alpha=0.7, linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Кривая обучения (потери)')
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Точность на валидации
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies,
         color='green', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.title('Точность на валидационной выборке')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Сохраняем график
plot_path = CURVES_DIR / f"training_curves_{timestamp}.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"📊 График обучения сохранён в: {plot_path}")

# ==================== 13. СОХРАНЕНИЕ МЕТРИК ====================
# Сохраняем текстовый файл с подробными метриками
metrics_path = RESULTS_DIR / f"metrics_{timestamp}.txt"

with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("ОБУЧЕНИЕ НЕЙРОСЕТИ ДЛЯ ЧАТ-БОТА\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("ПАРАМЕТРЫ ОБУЧЕНИЯ:\n")
    f.write(f"- Количество эпох (фактическое): {len(train_losses)}\n")
    f.write(f"- Размер батча: {batch_size}\n")
    f.write(f"- Размер скрытого слоя: {hidden_size}\n")
    f.write(f"- Скорость обучения: {learning_rate}\n")
    f.write(f"- Dropout: {dropout}\n")
    f.write(f"- Early stopping patience: {early_stopping_patience}\n\n")

    f.write("ДАННЫЕ:\n")
    f.write(f"- Количество намерений: {len(tags)}\n")
    f.write(f"  Намерения: {', '.join(tags)}\n")
    f.write(f"- Количество уникальных слов: {len(all_words)}\n")
    f.write(f"- Размер обучающей выборки: {len(X_train)}\n")
    f.write(f"- Размер валидационной выборки: {len(X_val)}\n\n")

    f.write("РЕЗУЛЬТАТЫ:\n")
    f.write(f"- Лучший loss на валидации: {best_val_loss:.4f}\n")
    f.write(f"- Лучшая точность на валидации: {max(val_accuracies):.4f}\n")
    f.write(f"- Финальная точность: {val_accuracies[-1]:.4f}\n\n")

    f.write("СОХРАНЁННЫЕ ФАЙЛЫ:\n")
    f.write(f"- Модель (с меткой времени): {model_path}\n")
    f.write(f"- Модель (data.pth): {compat_path}\n")
    f.write(f"- График обучения: {plot_path}\n")

print(f"📝 Метрики сохранены в: {metrics_path}")

# ==================== 14. ТЕСТОВЫЙ ПРОГОН ====================
print("\n" + "=" * 60)
print("ТЕСТОВЫЙ ПРОГОН ОБУЧЕННОЙ МОДЕЛИ:")
print("=" * 60)

# Переводим модель в режим оценки
model.eval()

# Тестовые фразы для проверки
test_phrases = [
    "привет",
    "готово",
    "да",
    "нет",
    "а то компьютер заберу",
    "сделал уроки",
    "не сделал",
    "пока"
]

print("\nРезультаты распознавания:")
with torch.no_grad():
    for phrase in test_phrases:
        # Токенизируем и создаём bag-of-words вектор
        tokens = tokenize(phrase)
        bag = bag_of_words(tokens, all_words)
        bag_tensor = torch.tensor(bag, dtype=torch.float32).unsqueeze(0).to(device)

        # Предсказываем намерение
        output = model(bag_tensor)
        _, predicted = torch.max(output, dim=1)
        prob = torch.softmax(output, dim=1)[0][predicted.item()].item()
        tag = tags[predicted.item()]

        print(f'  "{phrase}" → {tag} (уверенность: {prob:.3f})')

print("\n" + "=" * 60)
print("✅ Обучение завершено успешно!")
print("=" * 60)