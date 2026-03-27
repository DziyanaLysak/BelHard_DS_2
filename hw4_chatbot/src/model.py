"""
model.py — определение архитектуры нейросети.

Нейросеть используется для классификации намерений (intents).
Она принимает на вход bag-of-words вектор и возвращает вероятности
для каждого из возможных намерений.
"""

import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    """
    Нейросеть для классификации намерений.

    Архитектура:
        Входной слой → Скрытый слой 1 → ReLU → Dropout
                     → Скрытый слой 2 → ReLU → Dropout
                     → Выходной слой (без активации)

    Почему без активации на выходе?
        Функция потерь CrossEntropyLoss внутри себя применяет softmax,
        поэтому на выходе мы получаем "сырые" логиты (raw logits).

    Параметры:
        input_size (int): Размер входного вектора (количество слов в словаре)
        hidden_size (int): Количество нейронов в скрытых слоях
        num_classes (int): Количество классов (намерений)
        dropout (float): Вероятность отключения нейрона в dropout (0.5 по умолчанию)
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.5):
        super(NeuralNet, self).__init__()

        # Первый полносвязный слой: input_size → hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)

        # Второй полносвязный слой: hidden_size → hidden_size
        self.l2 = nn.Linear(hidden_size, hidden_size)

        # Выходной слой: hidden_size → num_classes
        self.l3 = nn.Linear(hidden_size, num_classes)

        # Функция активации ReLU (выпрямленный линейный элемент)
        # ReLU(x) = max(0, x) — помогает сети обучаться нелинейным зависимостям
        self.relu = nn.ReLU()

        # Dropout — регуляризация: случайно отключает dropout% нейронов
        # Это предотвращает переобучение, так как сеть не может полагаться
        # на отдельные нейроны и вынуждена учить более общие признаки
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход данных через сеть.

        Аргументы:
            x (torch.Tensor): Входной тензор размера (batch_size, input_size)

        Возвращает:
            torch.Tensor: Выходной тензор размера (batch_size, num_classes)
                          — "сырые" логиты (не softmax)
        """
        # Слой 1 → ReLU → Dropout
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)

        # Слой 2 → ReLU → Dropout
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Выходной слой (без активации)
        out = self.l3(out)

        return out