import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os


# ========== ОБЁРТКА ДЛЯ СРЕДЫ ==========
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(256,), dtype=np.float32)

    def observation(self, obs):
        return obs.flatten()


def test_model(model_path, model_name, games=5, max_steps=1000):
    """Тестирует модель и выводит статистику"""
    print("=" * 50)
    print(f"ТЕСТИРОВАНИЕ: {model_name}")
    print(f"Путь: {model_path}")
    print(f"Лимит ходов: {max_steps}")

    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл не найден!")
        return []

    model = PPO.load(model_path)
    scores = []
    steps_list = []
    finished_games = 0

    for i in range(games):
        env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0")
        env = FlattenObservation(env)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

        if done:
            finished_games += 1
            status = "завершена"
        else:
            status = "лимит ходов"

        scores.append(int(total_reward))
        steps_list.append(step)
        env.close()
        print(f"   Партия {i + 1}: {int(total_reward):5d} очков | {step:4d} ходов | {status}")

    print("-" * 40)
    print(f"Средний счёт:    {np.mean(scores):.0f} ± {np.std(scores):.0f}")
    print(f"Среднее ходов:   {np.mean(steps_list):.0f}")
    print(f"Максимум:        {max(scores)}")
    print(f"Минимум:         {min(scores)}")
    print(f"Завершено игр:   {finished_games}/{games}")
    print("=" * 50)

    return scores


# ========== ЗАПУСК ТЕСТОВ ==========
if __name__ == "__main__":
    # Путь к моделям (на уровень выше, так как скрипт в src/)
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    MAX_STEPS = 1000  # лимит ходов

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ 2048 AI")
    print("=" * 60 + "\n")

    # Тест НОВИЧКА
    novice_scores = test_model(
        os.path.join(MODELS_DIR, "2048_ai_1M.zip"),
        "НОВИЧОК (1M шагов)",
        games=5,
        max_steps=MAX_STEPS
    )

    # Тест ЛЮБИТЕЛЯ
    amateur_scores = test_model(
        os.path.join(MODELS_DIR, "2048_ai_2M.zip"),
        "ЛЮБИТЕЛЬ (2M шагов)",
        games=5,
        max_steps=MAX_STEPS
    )

    # Тест ПРОФИ
    pro_scores = test_model(
        os.path.join(MODELS_DIR, "2048_ai_3M.zip"),
        "ПРОФИ (3M шагов)",
        games=5,
        max_steps=MAX_STEPS
    )

    # ИТОГОВАЯ ТАБЛИЦА
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 60)

    if novice_scores:
        print(
            f"НОВИЧОК (1M):   среднее {np.mean(novice_scores):.0f} ± {np.std(novice_scores):.0f} | макс {max(novice_scores)}")
    if amateur_scores:
        print(
            f"ЛЮБИТЕЛЬ (2M):  среднее {np.mean(amateur_scores):.0f} ± {np.std(amateur_scores):.0f} | макс {max(amateur_scores)}")
    if pro_scores:
        print(f"ПРОФИ (3M):     среднее {np.mean(pro_scores):.0f} ± {np.std(pro_scores):.0f} | макс {max(pro_scores)}")

    print("=" * 60)