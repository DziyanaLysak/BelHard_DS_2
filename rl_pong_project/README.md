#  2048 AI Battle

Telegram-бот с искусственным интеллектом, обученным играть в головоломку **2048** методом **Reinforcement Learning (PPO)**.

---
###  Доступ к боту

Бот доступен в Telegram: **[@my2048_ai_bot](https://t.me/my2048_ai_bot)**

Для запуска достаточно написать команду `/start`.

---

##  Что умеет бот

-  **Одиночная игра** — классическая 2048
-  **Игра против ИИ** — дуэль с нейросетью
-  **Три уровня сложности:**
  - 🟢 Новичок (1 млн шагов обучения)
  - 🟡 Любитель (2 млн шагов)
  - 🔴 Профи (3 млн шагов)
-  **Запись игры ИИ в GIF** — можно посмотреть, как думает машина
-  **Встроенные правила** и советы по стратегии

---

## 🎬 Демонстрация работы

[▶️ Смотреть видео на Google Диске](https://drive.google.com/file/d/1Jd2_KnGnBrWoFMni89TwLzSb7hxCEVmX/view?usp=sharing)

---

##  Как обучалась модель

| Этап | Шагов | Результат |
|------|-------|-----------|
| 🟢 Новичок | 1 000 000 | ~550 очков |
| 🟡 Любитель | 2 000 000 | ~700 очков |
| 🔴 Профи | 3 000 000 | ~980 очков |

**Алгоритм:** PPO (Proximal Policy Optimization)  
**Среда:** gymnasium-2048  
**Нейросеть:** MlpPolicy

Подробный отчёт — в ноутбуке [`notebooks/RL_Pong_2048_Final.ipynb`](notebooks/RL_Pong_2048_Final.ipynb)

---

##  Быстрый запуск

### 1. Клонировать репозиторий
```bash
git clone https://github.com/DziyanaLysak/BelHard_DS_2.git
cd BelHard_DS_2/rl_pong_project
```

### 2. Установить зависимости
```bash
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Добавить токен бота
Создать файл `.env` в корне проекта:
```
BOT_TOKEN=ваш_токен_от_BotFather
```

### 4. Запустить бота
```bash
python src/bot.py
```

---

##  Структура проекта

```
rl_pong_project/
├── models/                  # Обученные модели (.zip)
├── notebooks/               # Jupyter-отчёт
├── src/                     # Исходный код
│   ├── bot.py               # Telegram-бот
│   ├── train.py             # Обучение с нуля
│   ├── continue_train.py    # Дообучение
│   └── test_models.py       # Тестирование
├── .env                     # Токен (не в Git)
├── Dockerfile               # Для деплоя
├── requirements.txt         # Зависимости
└── README.md                # Этот файл
```

---

##  Технологии

- `Python 3.10`
- `stable-baselines3` (PPO)
- `gymnasium-2048`
- `python-telegram-bot`
- `PyTorch`
- `Docker`

---

**Dziyana Lysak**  
GitHub: [DziyanaLysak](https://github.com/DziyanaLysak)
```

