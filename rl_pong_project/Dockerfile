FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Клонируем gymnasium-2048 напрямую из GitHub
RUN git clone https://github.com/Quentin18/gymnasium-2048.git
RUN cd gymnasium-2048 && pip install -e .

# Копируем остальной код проекта
COPY models/ ./models/
COPY src/ ./src/

ENV TELEGRAM_TOKEN=""

CMD ["python", "src/bot.py"]