"""
utils.py — вспомогательные функции для работы с путями.

Этот файл определяет универсальные пути к папкам проекта,
чтобы код работал независимо от того, из какой директории его запускают.
"""

import os
from pathlib import Path

# ==================== ОПРЕДЕЛЕНИЕ КОРНЕВОЙ ПАПКИ ====================
# BASE_DIR — это корневая папка проекта (hw4_chatbot)
# __file__ — путь к текущему файлу (src/utils.py)
# .resolve() — превращает относительный путь в абсолютный
# .parent — поднимаемся на один уровень вверх (из src/ в hw4_chatbot/)
# .parent — поднимаемся ещё раз, если utils.py лежит в src/
# Но в нашей структуре utils.py лежит в src/, поэтому нужно подняться на 1 уровень
BASE_DIR = Path(__file__).resolve().parent.parent

# ==================== ПАПКИ ПРОЕКТА ====================
# Папка с данными (intents.json)
DATA_DIR = BASE_DIR / "data"

# Папка для сохранения моделей (.pth файлы)
MODELS_DIR = BASE_DIR / "models"

# Папка для результатов (логи, графики, метрики)
RESULTS_DIR = BASE_DIR / "results"

# Внутри results создаём отдельные папки для логов и графиков
LOGS_DIR = RESULTS_DIR / "logs"  # Логи диалогов
CURVES_DIR = RESULTS_DIR / "training_curves"  # Графики обучения

# ==================== СОЗДАНИЕ ПАПОК ====================
# Создаём все необходимые папки, если их нет
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, CURVES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)  # exist_ok=True — не выдавать ошибку, если папка уже есть

# ==================== КОНСТАНТЫ ДЛЯ СОВМЕСТИМОСТИ ====================
# В оригинальном репозитории модель сохраняется как data.pth в корне
# Сохраняем этот путь для совместимости
DEFAULT_MODEL_FILE = "data.pth"
DEFAULT_MODEL_PATH = BASE_DIR / DEFAULT_MODEL_FILE


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_latest_model() -> Path:
    """
    Возвращает путь к самой свежей модели в папке MODELS_DIR.
    Если в папке нет моделей, возвращает путь к data.pth в корне.
    """
    # Ищем все .pth файлы в папке models
    models = list(MODELS_DIR.glob("*.pth"))

    if not models:
        # Если нет моделей, возвращаем путь по умолчанию
        return DEFAULT_MODEL_PATH

    # Возвращаем самый новый файл (по дате создания)
    return max(models, key=os.path.getctime)


def ensure_dirs() -> None:
    """
    Гарантирует, что все необходимые папки существуют.
    Эту функцию можно вызывать перед сохранением файлов.
    """
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, CURVES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)