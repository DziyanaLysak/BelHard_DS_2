"""
ТЕСТОВЫЙ скрипт для проверки работоспособности
Обучение всего 1000 шагов (1-2 минуты)
Модель сохраняется в папку models/ (корень проекта)
"""

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os

# ========== 1. НАСТРАИВАЕМ ПУТИ ==========
# PROJECT_ROOT - это папка rl_pong_project (родительская для src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Создаём папку models, если её ещё нет
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 50)
print("🔧 ТЕСТОВЫЙ ЗАПУСК (1000 шагов)")
print("=" * 50)
print(f"📁 Корень проекта: {PROJECT_ROOT}")
print(f"📁 Модели будут в: {MODELS_DIR}")

# ========== 2. СОЗДАЁМ СРЕДУ ==========
# Средой называется игра, с которой ИИ взаимодействует
print("\n1. Создаём среду 2048...")
env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")


# Оборачиваем наблюдение: из (4,4,16) делаем плоский вектор (256,)
# Это нужно, чтобы нейросеть MlpPolicy могла его обработать
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        from gymnasium import spaces
        # 256 чисел (4*4*16) от 0 до 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4 * 4 * 16,), dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten()  # Превращаем картинку в строчку чисел


env = FlattenObservation(env)
print("   ✅ Среда создана и обёрнута")

# ========== 3. СОЗДАЁМ МОДЕЛЬ ==========
# PPO - алгоритм обучения с подкреплением
# MlpPolicy - простая нейросеть (не свёрточная)
print("\n2. Создаём модель PPO...")
model = PPO(
    "MlpPolicy",  # Тип нейросети (многослойный перцептрон)
    env,  # Среда, в которой играет ИИ
    verbose=1,  # Показывать прогресс обучения
    device="cpu",  # На ноутбуке используем процессор
    n_steps=512,  # Сколько шагов собрать перед обучением
    batch_size=32,  # Размер порции данных для обучения
    learning_rate=0.00025,  # Скорость обучения
    ent_coef=0.01  # Поощрение за исследование (чтобы не застревал)
)
print("   ✅ Модель создана")

# ========== 4. ТЕСТОВОЕ ОБУЧЕНИЕ ==========
print("\n3. Запускаем тестовое обучение (1000 шагов)...")
print("   ⏳ Это займёт 1-2 минуты...")
model.learn(total_timesteps=1000)  # Всего 1000 действий ИИ

# ========== 5. СОХРАНЯЕМ МОДЕЛЬ ==========
print("\n4. Сохраняем тестовую модель...")
model_path = os.path.join(MODELS_DIR, "2048_ai_test")
model.save(model_path)
print(f"   ✅ Модель сохранена: {model_path}.zip")

# ========== 6. ПРОВЕРЯЕМ, ЧТО МОДЕЛЬ РАБОТАЕТ ==========
print("\n5. Проверяем модель на 1 эпизоде...")
obs, _ = env.reset()  # Начинаем новую игру
done = False
total_reward = 0
step = 0

while not done:
    # Модель предсказывает лучшее действие
    action, _ = model.predict(obs, deterministic=True)
    # Выполняем действие в среде
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated
    step += 1
    if step > 100:  # Защита от бесконечной игры
        break

print(f"   📊 Результат: {int(total_reward)} очков за {step} ходов")

env.close()
print("\n" + "=" * 50)
print("✅ ТЕСТ ПРОЙДЕН! Модель в папке models/")
print("=" * 50)