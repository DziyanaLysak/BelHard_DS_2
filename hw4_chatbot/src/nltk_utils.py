"""
nltk_utils.py — функции для обработки текста с помощью NLTK.

Здесь реализованы:
- Токенизация (разбиение текста на слова)
- Стемминг (приведение слов к корневой форме)
- Преобразование текста в bag-of-words вектор
- Загрузка intents.json
"""

import nltk
import numpy as np
import ssl
from nltk.stem import SnowballStemmer

# ==================== ОБХОД SSL-ПРОБЛЕМЫ ====================
# При скачивании данных NLTK может возникнуть SSL-ошибка.
# Этот код обходит её, отключая проверку сертификата.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ==================== ЗАГРУЗКА ДАННЫХ NLTK ====================
# Для токенизации нужна модель 'punkt'. Скачиваем её, если ещё не скачана.
# 'punkt_tab' — это табличная версия, нужная для некоторых версий NLTK
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')  # на всякий случай скачиваем и обычную версию

# ==================== СТЕММЕР ДЛЯ РУССКОГО ЯЗЫКА ====================
# SnowballStemmer поддерживает множество языков, в том числе русский
stemmer = SnowballStemmer("russian")


def tokenize(sentence: str) -> list:
    """
    Разбивает предложение на отдельные слова (токены).

    Пример:
        tokenize("Привет, как дела?") -> ["Привет", ",", "как", "дела", "?"]

    Args:
        sentence (str): Входное предложение

    Returns:
        list: Список токенов (слов и знаков препинания)
    """
    # Проверяем, что предложение не пустое
    if not sentence or not sentence.strip():
        return []
    # Токенизируем с указанием языка (русский)
    return nltk.word_tokenize(sentence, language='russian')


def stem(word: str) -> str:
    """
    Приводит слово к его корневой форме (стемминг).

    Пример:
        stem("делала") -> "дела"
        stem("уроками") -> "урок"

    Args:
        word (str): Слово для стемминга

    Returns:
        str: Корневая форма слова
    """
    return stemmer.stem(word)


def bag_of_words(tokenized_sentence: list, all_words: list) -> np.ndarray:
    """
    Создаёт bag-of-words вектор для предложения.

    Bag-of-words — это способ представления текста в виде вектора,
    где каждый элемент соответствует слову из словаря (all_words).
    Значение 1 означает, что слово присутствует в предложении, 0 — отсутствует.

    Пример:
        all_words = ["мама", "мыла", "раму"]
        предложение = "мама мыла"
        результат = [1, 1, 0]

    Args:
        tokenized_sentence (list): Токенизированное предложение (список слов)
        all_words (list): Словарь всех известных слов

    Returns:
        np.ndarray: Вектор bag-of-words (массив из 0 и 1)
    """
    # Сначала применяем стемминг ко всем словам в предложении
    # Это нужно, чтобы сравнивать корни слов, а не точные формы
    stemmed_sentence = [stem(w) for w in tokenized_sentence]

    # Для быстрого поиска преобразуем список в множество
    sentence_set = set(stemmed_sentence)

    # Создаём вектор из нулей нужной длины
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Для каждого слова из словаря проверяем, есть ли оно в предложении
    for idx, w in enumerate(all_words):
        if w in sentence_set:
            bag[idx] = 1.0

    return bag


def load_intents(file_path: str) -> dict:
    """
    Загружает intents.json из указанного пути.

    Args:
        file_path (str): Путь к файлу intents.json

    Returns:
        dict: Словарь с намерениями
    """
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)