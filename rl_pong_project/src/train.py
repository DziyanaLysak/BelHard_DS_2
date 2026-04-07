"""
Полное обучение ИИ игре 2048
Запускать на мощном ПК с видеокартой RTX 3060
Время обучения: 1-2 часа (2 000 000 шагов)
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
print("🚀 ПОЛНОЕ ОБУЧЕНИЕ ИГРЕ 2048")
print("=" * 50)
print(f"📁 Корень проекта: {PROJECT_ROOT}")
print(f"📁 Модели будут в: {MODELS_DIR}")

# ========== 2. СОЗДАЁМ СРЕДУ ==========
print("\n1. Создаём среду 2048...")
env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")


# Оборачиваем наблюдение: из (4,4,16) делаем плоский вектор (256,)
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4 * 4 * 16,), dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten()


env = FlattenObservation(env)
print("   ✅ Среда создана")

# ========== 3. СОЗДАЁМ МОДЕЛЬ ==========
# PPO с нейросетью MlpPolicy (простой многослойный перцептрон)
print("\n2. Создаём модель PPO...")
model = PPO(
    "MlpPolicy",  # Тип нейросети
    env,  # Среда
    verbose=1,  # Показывать прогресс
    device="cuda",  # Используем видеокарту RTX 3060
    n_steps=2048,  # Шагов на сбор данных (стандарт для PPO)
    batch_size=64,  # Размер батча для обучения
    learning_rate=0.00025,  # Скорость обучения
    ent_coef=0.01  # Поощрение за исследование
)
print("   ✅ Модель создана на GPU")

# ========== 4. ЗАПУСКАЕМ ОБУЧЕНИЕ ==========
print("\n3. Запускаем полное обучение...")
print("   ⏳ Время: 1-2 часа (2 000 000 шагов)")
print("   💡 Можно оставить компьютер и заниматься другими делами")
print("=" * 50)

model.learn(total_timesteps=2_000_000)

# ========== 5. СОХРАНЯЕМ МОДЕЛЬ ==========
print("\n4. Сохраняем обученную модель...")
model_path = os.path.join(MODELS_DIR, "2048_ai")
model.save(model_path)
print(f"   ✅ Модель сохранена: {model_path}.zip")

# ========== 6. БЫСТРАЯ ПРОВЕРКА ==========
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
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"📁 Модель: {MODELS_DIR}\\2048_ai.zip")
print("=" * 50)