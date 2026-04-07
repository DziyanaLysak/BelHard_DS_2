FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости для pygame
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы проекта
COPY requirements.txt .
COPY models/ ./models/
COPY src/ ./src/

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Переменная для токена бота (устанавливается в Render)
ENV TELEGRAM_TOKEN=""

# Запуск бота
CMD ["python", "src/bot.py"]