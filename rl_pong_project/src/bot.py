"""
Telegram-бот для игры 2048
Использует обученную модель PPO
"""

import os
from telegram import Update
from telegram.ext import Application, CommandHandler
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np


# ========== 1. ПРЕОБРАЗОВАНИЕ НАБЛЮДЕНИЙ ==========
# Среда 2048 выдаёт картинку (4,4,16) — 16 каналов 4x4
# Нейросеть MlpPolicy работает только с плоским вектором
# Этот класс превращает (4,4,16) в вектор из 256 чисел
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        from gymnasium import spaces
        # Новое пространство наблюдений: 256 чисел от 0 до 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4 * 4 * 16,), dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten()  # Превращаем картинку в строчку


# ========== 2. ЗАГРУЗКА МОДЕЛИ ==========
# Определяем путь к папке с проектом
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "2048_ai_test")

print(f"Загрузка модели из {MODEL_PATH}.zip")
model = PPO.load(MODEL_PATH)
print("Модель загружена")


# ========== 3. ОБРАБОТЧИК КОМАНДЫ /play ==========
async def play(update: Update, context):
    await update.message.reply_text("🎮 ИИ играет в 2048...")

    # Создаём среду
    env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")
    env = FlattenObservation(env)

    # Один эпизод игры
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        # Модель предсказывает действие
        action, _ = model.predict(obs, deterministic=True)
        # Выполняем действие в среде
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1
        if step > 499:  # Защита от бесконечной игры
            break

    env.close()
    await update.message.reply_text(f"📊 Счёт: {int(total_reward)} очков за {step} ходов")


# ========== 4. ЗАПУСК БОТА ==========
def main():
    # Токен от @BotFather (вставь свой)
    TOKEN = "8578815163:AAHgwVJzYyn9ruhYJ7wj0uIy7-SoeLPtPhE"

    # Создаём приложение
    app = Application.builder().token(TOKEN).build()
    # Добавляем обработчик команды /play
    app.add_handler(CommandHandler("play", play))

    print("✅ Бот запущен")
    app.run_polling()  # Запускаем бота


if __name__ == "__main__":
    main()