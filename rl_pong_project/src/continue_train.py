"""
Дообучение ИИ игре 2048 на дополнительные 1 000 000 шагов
Запускать на мощном ПК с видеокартой RTX 3060
"""

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os

# ========== 1. НАСТРАИВАЕМ ПУТИ ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 50)
print("🚀 ДООБУЧЕНИЕ МОДЕЛИ 2048 (+1 000 000 шагов)")
print("=" * 50)
print(f"📁 Корень проекта: {PROJECT_ROOT}")
print(f"📁 Модели в: {MODELS_DIR}")

# ========== 2. СОЗДАЁМ СРЕДУ (ТОЧНО КАК В ОБУЧЕНИИ) ==========
print("\n1. Создаём среду 2048...")
env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")

class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4*4*16,), dtype=np.float32
        )
    def observation(self, obs):
        return obs.flatten()

env = FlattenObservation(env)
print("   ✅ Среда создана")

# ========== 3. ЗАГРУЖАЕМ МОДЕЛЬ ==========
print("\n2. Загружаем модель models/2048_ai.zip...")
model = PPO.load(os.path.join(MODELS_DIR, "2048_ai"), env=env)
print("   ✅ Модель загружена")

# ========== 4. ДООБУЧАЕМ ==========
print("\n3. Дообучаем на 1 000 000 шагов...")
print("   ⏳ Время: ~2 часа")
model.learn(total_timesteps=1_000_000)

# ========== 5. СОХРАНЯЕМ ==========
print("\n4. Сохраняем обновлённую модель...")
model.save(os.path.join(MODELS_DIR, "2048_ai"))
print("   ✅ Модель сохранена")

# ========== 6. ПРОВЕРКА ==========
print("\n5. Проверяем модель на 1 эпизоде...")
obs, _ = env.reset()
done = False
total_reward = 0
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated
    step += 1
    if step > 500:
        break

print(f"   📊 Результат: {int(total_reward)} очков за {step} ходов")

env.close()
print("\n" + "=" * 50)
print("✅ ДООБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"📁 Модель: {MODELS_DIR}\\2048_ai.zip")
print("=" * 50)
