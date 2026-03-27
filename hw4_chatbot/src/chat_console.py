"""
chat_console.py — консольная версия чат-бота.

Этот скрипт запускает диалог в терминале.
Нейросеть используется только для распознавания простых команд:
- приветствие
- готово
- да / подтверждение
- нет / отказ

Вся остальная логика (выбор предметов, ввод оценок, расчёт балла)
реализована обычным кодом на Python.
"""

import random
import re
import torch
from datetime import datetime

# Импортируем наши модули
from src.model import NeuralNet
from src.nltk_utils import bag_of_words, tokenize, load_intents
from src.utils import DATA_DIR, get_latest_model, ensure_dirs, LOGS_DIR

# ==================== 1. ЗАГРУЗКА МОДЕЛИ ====================
# Создаём папки, если их нет
ensure_dirs()

# Определяем устройство (GPU если есть, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем намерения из файла intents.json
intents_path = DATA_DIR / "intents.json"
intents = load_intents(intents_path)

# Загружаем последнюю обученную модель (автоматически из папки models/)
model_path = get_latest_model()
print(f"Загрузка модели из: {model_path}")

data = torch.load(model_path, map_location=device)

# Извлекаем параметры модели, сохранённые при обучении
input_size = data["input_size"]  # размер входного вектора (124 слова)
hidden_size = data["hidden_size"]  # количество нейронов в скрытых слоях
output_size = data["output_size"]  # количество намерений (6)
all_words = data['all_words']  # словарь всех слов
tags = data['tags']  # список намерений
model_state = data["model_state"]  # веса модели

# Создаём модель и загружаем веса
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # Переводим в режим оценки (выключаем dropout)

bot_name = "КиберПатруль"


# ==================== 2. ФУНКЦИИ ДЛЯ РАБОТЫ БОТА ====================

def get_subject_in_prepositional(subject: str) -> str:
    """
    Возвращает название предмета в предложном падеже (для вопроса "по ...").

    Примеры:
        "математика" -> "математике"
        "русский язык" -> "русскому языку"
        "человек и мир" -> "человеку и миру"
    """
    subjects_dict = {
        "математика": "математике",
        "русский язык": "русскому языку",
        "английский язык": "английскому языку",
        "белорусский язык": "белорусскому языку",
        "история": "истории",
        "литература": "литературе",
        "человек и мир": "человеку и миру"
    }
    return subjects_dict.get(subject, subject)


def predict_intent(text: str) -> tuple:
    """
    Определяет намерение пользователя с помощью нейросети.

    Args:
        text (str): Текст пользователя

    Returns:
        tuple: (намерение, уверенность)
    """
    # Токенизируем текст и создаём bag-of-words вектор
    tokens = tokenize(text)
    bag = bag_of_words(tokens, all_words)
    bag_tensor = torch.tensor(bag, dtype=torch.float32).unsqueeze(0).to(device)

    # Предсказываем намерение
    with torch.no_grad():
        output = model(bag_tensor)
        _, predicted = torch.max(output, dim=1)
        prob = torch.softmax(output, dim=1)[0][predicted.item()].item()
        tag = tags[predicted.item()]

    return tag, prob


def assign_tasks(avg_score: float) -> list:
    """
    Назначает дополнительные задания в зависимости от среднего балла.

    Правила:
        - avg_score >= 9: бонус, нет заданий
        - avg_score >= 7: 1 бытовое задание
        - avg_score >= 5: 2 задания (бытовое + учебное)
        - avg_score < 5: 3 задания (2 бытовых + 1 учебное)
        - avg_score is None (нет оценок): 2 задания (бытовое + учебное)
    """
    house_tasks = [
        "🔄 Пропылесосить комнату",
        "🔄 Протереть пыль на полках",
        "🔄 Сложить чистую одежду в шкаф",
        "🔄 Вынести мусор",
        "🔄 Протереть зеркала",
        "🔄 Разобрать письменный стол",
        "🔄 Заправить кровать",
        "🔄 Почистить обувь"
    ]

    study_tasks = [
        "📚 Решить 5 задач из учебника по математике",
        "📚 Прочитать 20 страниц внеклассного чтения",
        "📚 Сделать 5 упражнений по английскому языку",
        "📚 Сделать 5 упражнений по русскому языку",
        "📚 Выучить 10 новых английских слов",
        "📚 Написать сочинение-миниатюру (5–7 предложений)"
    ]

    tasks = []

    if avg_score is None:
        # Нет оценок — среднее количество заданий
        tasks.append(random.choice(house_tasks))
        tasks.append(random.choice(study_tasks))
    elif avg_score >= 9:
        # Отлично — никаких дополнительных заданий
        return []
    elif avg_score >= 7:
        # Хорошо — одно лёгкое задание
        tasks.append(random.choice(house_tasks))
    elif avg_score >= 5:
        # Удовлетворительно — два задания
        tasks.append(random.choice(house_tasks))
        tasks.append(random.choice(study_tasks))
    else:
        # Плохо — три задания
        tasks.append(random.choice(house_tasks))
        tasks.append(random.choice(house_tasks))
        tasks.append(random.choice(study_tasks))

    return tasks


def log_conversation(user_message: str, bot_response: str) -> None:
    """
    Сохраняет диалог в лог-файл.
    Файл создаётся в папке results/logs/ с именем log_YYYYMMDD.txt
    """
    log_file = LOGS_DIR / f"log_{datetime.now().strftime('%Y%m%d')}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | Пользователь: {user_message}\n")
        f.write(f"{timestamp} | Бот: {bot_response}\n\n")


def get_response_by_tag(tag: str) -> str:
    """
    Возвращает случайный ответ из intents.json по тегу намерения.
    """
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Я вас не понял."


# ==================== 3. ОСНОВНОЙ ДИАЛОГ ====================

def main():
    """Главная функция для запуска консольного чата."""

    print("\n" + "=" * 60)
    print(f"🤖 {bot_name} — помощник для получения пароля от компьютера")
    print("=" * 60)
    print("Введите 'выход', чтобы завершить диалог.\n")

    # Приветствие из intents.json
    greeting = get_response_by_tag("приветствие")
    print(f"{bot_name}: {greeting}")
    log_conversation("(начало диалога)", greeting)

    # ==================== СОСТОЯНИЕ ДИАЛОГА ====================
    stage = "main_tasks"  # текущий этап диалога
    lessons = []  # список выбранных предметов
    grades = {}  # словарь оценок {предмет: оценка}
    current_subject_index = 0  # индекс текущего предмета для ввода оценки

    # Список всех возможных предметов
    SUBJECTS = [
        "математика",
        "русский язык",
        "английский язык",
        "история",
        "литература",
        "белорусский язык",
        "человек и мир"
    ]

    # ==================== ОСНОВНОЙ ЦИКЛ ====================
    while True:
        user_input = input("Ты: ").strip()

        # Пропускаем пустой ввод
        if not user_input:
            continue

        # Команда выхода
        if user_input.lower() == "выход":
            farewell = get_response_by_tag("выход")
            print(f"{bot_name}: {farewell}")
            log_conversation(user_input, farewell)
            break

        # Распознаём намерение с помощью нейросети
        tag, prob = predict_intent(user_input)

        # ========== ЭТАП 1: ГЛАВНЫЕ ЗАДАНИЯ ==========
        if stage == "main_tasks":
            if tag == "готово" and prob > 0.75:
                response = get_response_by_tag("готово")
                print(f"{bot_name}: {response}")
                log_conversation(user_input, response)

                # Переходим к выбору предметов
                print(f"{bot_name}: Теперь выбери предметы, которые были сегодня.")
                print(f"{bot_name}: Введи номера через пробел (например: 1 2 4):")
                for i, subj in enumerate(SUBJECTS, 1):
                    print(f"  {i}. {subj}")
                stage = "subjects"

            elif tag == "отказ" and prob > 0.75:
                response = get_response_by_tag("отказ")
                print(f"{bot_name}: {response}")
                log_conversation(user_input, response)
                print(f"{bot_name}: Напиши 'готово', когда выполнишь все главные задания:\n\n"
                      "1. Поесть\n2. Сложить школьную одежду в шкаф\n3. Помыть посуду\n4. Сделать уроки")
            else:
                print(f"{bot_name}: Напиши 'готово', когда выполнишь все главные задания:\n\n"
                      "1. Поесть\n2. Сложить школьную одежду в шкаф\n3. Помыть посуду\n4. Сделать уроки")

        # ========== ЭТАП 2: ВЫБОР ПРЕДМЕТОВ ==========
        elif stage == "subjects":
            try:
                # Извлекаем все числа из ввода
                numbers = [int(x) for x in user_input.split() if x.isdigit()]
                # Оставляем только валидные номера (от 1 до количества предметов)
                valid_numbers = [n for n in numbers if 1 <= n <= len(SUBJECTS)]

                if not valid_numbers:
                    print(f"{bot_name}: Введи номера из списка (1–{len(SUBJECTS)}).")
                    continue

                # Сохраняем выбранные предметы
                lessons = [SUBJECTS[n - 1] for n in valid_numbers]
                print(f"{bot_name}: Ты выбрал: {', '.join(lessons)}")
                print("Всё верно? (да/нет)")
                stage = "confirm_subjects"

            except ValueError:
                print(f"{bot_name}: Введи номера через пробел, например: 1 2 4")

        # ========== ПОДТВЕРЖДЕНИЕ СПИСКА ПРЕДМЕТОВ ==========
        elif stage == "confirm_subjects":
            if tag == "подтверждение" and prob > 0.75:
                response = get_response_by_tag("подтверждение")
                print(f"{bot_name}: {response}")
                log_conversation(user_input, response)

                # Проверяем, что список не пуст
                if not lessons:
                    print(f"{bot_name}: Ты не выбрал ни одного предмета. Давай заново.")
                    stage = "subjects"
                    continue

                # Переходим к вводу оценок
                stage = "grades"
                current_subject_index = 0
                first_subject = lessons[current_subject_index]
                prep_subject = get_subject_in_prepositional(first_subject)
                print(f"{bot_name}: Какая оценка по {prep_subject}? (если оценки не было, введи 'нет')")

            elif tag == "отказ" and prob > 0.75:
                response = get_response_by_tag("отказ")
                print(f"{bot_name}: {response}")
                log_conversation(user_input, response)
                print(f"{bot_name}: Напиши номера предметов заново через пробел.")
                lessons = []  # Очищаем список предметов
                stage = "subjects"
            else:
                print(f"{bot_name}: Ответь 'да' или 'нет'.")

        # ========== ЭТАП 3: ОЦЕНКИ ==========
        elif stage == "grades":
            subject = lessons[current_subject_index]

            # Проверка на "нет оценки" (принимаем только "нет" и "не было")
            if user_input.lower() in ["нет", "не было"]:
                prep_subject = get_subject_in_prepositional(subject)
                print(f"{bot_name}: Принято. Оценки по {prep_subject} не было.")
                grades[subject] = None  # Сохраняем None как признак отсутствия оценки
                current_subject_index += 1

                if current_subject_index < len(lessons):
                    next_subject = lessons[current_subject_index]
                    prep_subject = get_subject_in_prepositional(next_subject)
                    print(f"{bot_name}: Какая оценка по {prep_subject}? (если оценки не было, введи 'нет')")
                else:
                    # Все предметы обработаны — выводим список и сразу переходим к расчёту
                    print(f"{bot_name}: Спасибо! Вот что я записал:")
                    for s, g in grades.items():
                        if g is None:
                            print(f"  {s}: нет оценки")
                        else:
                            print(f"  {s}: {g}")
                    log_conversation(user_input, f"Записаны оценки: {grades}")

                    # Собираем только числовые оценки (игнорируем None)
                    valid_grades = [g for g in grades.values() if g is not None]

                    # Расчёт среднего балла
                    if valid_grades:
                        avg_score = sum(valid_grades) / len(valid_grades)
                        print(f"{bot_name}: Твой средний балл сегодня: {avg_score:.1f}")

                        if avg_score >= 9:
                            print(f"{bot_name}: 🎉 Отлично! Ты молодец! Так держать!")
                        elif avg_score >= 7:
                            print(f"{bot_name}: 👍 Хорошо, но есть куда расти!")
                        elif avg_score >= 5:
                            print(f"{bot_name}: 📚 Неплохо, но нужно подтянуться!")
                        else:
                            print(f"{bot_name}: ⚠️ Низкий балл. Нужно исправляться!")
                    else:
                        avg_score = None
                        print(f"{bot_name}: Сегодня не было оценок.")
                        print("В следующий раз постарайся получить пятёрки!")

                    # Назначаем дополнительные задания
                    tasks = assign_tasks(avg_score)

                    if tasks:
                        print(f"{bot_name}: 📝 Дополнительные задания на сегодня:")
                        for i, task in enumerate(tasks, 1):
                            print(f"  {i}. {task}")
                    else:
                        print(f"{bot_name}: 🌟 Ты отлично справился! Дополнительных заданий нет!")

                    print(f"{bot_name}: Когда выполнишь всё, напиши 'готово'.")
                    stage = "password"
            else:
                # Обычная проверка цифр (оценка от 1 до 10)
                digits = re.findall(r'\b(10|[1-9])\b', user_input)

                if digits:
                    grade = int(digits[0])
                    if 1 <= grade <= 10:
                        grades[subject] = grade
                        current_subject_index += 1

                        if current_subject_index < len(lessons):
                            next_subject = lessons[current_subject_index]
                            prep_subject = get_subject_in_prepositional(next_subject)
                            print(f"{bot_name}: Какая оценка по {prep_subject}? (если оценки не было, введи 'нет')")
                        else:
                            # Все предметы обработаны — выводим список и сразу переходим к расчёту
                            print(f"{bot_name}: Спасибо! Вот что я записал:")
                            for s, g in grades.items():
                                if g is None:
                                    print(f"  {s}: нет оценки")
                                else:
                                    print(f"  {s}: {g}")
                            log_conversation(user_input, f"Записаны оценки: {grades}")

                            # Собираем только числовые оценки
                            valid_grades = [g for g in grades.values() if g is not None]

                            # Расчёт среднего балла
                            if valid_grades:
                                avg_score = sum(valid_grades) / len(valid_grades)
                                print(f"{bot_name}: Твой средний балл сегодня: {avg_score:.1f}")

                                if avg_score >= 9:
                                    print(f"{bot_name}: 🎉 Отлично! Ты молодец! Так держать!")
                                elif avg_score >= 7:
                                    print(f"{bot_name}: 👍 Хорошо, но есть куда расти!")
                                elif avg_score >= 5:
                                    print(f"{bot_name}: 📚 Неплохо, но нужно подтянуться!")
                                else:
                                    print(f"{bot_name}: ⚠️ Низкий балл. Нужно исправляться!")
                            else:
                                avg_score = None
                                print(f"{bot_name}: Сегодня не было оценок.")
                                print("В следующий раз постарайся получить хорошие оценки!")

                            # Назначаем дополнительные задания
                            tasks = assign_tasks(avg_score)

                            if tasks:
                                print(f"{bot_name}: 📝 Дополнительные задания на сегодня:")
                                for i, task in enumerate(tasks, 1):
                                    print(f"  {i}. {task}")
                            else:
                                print(f"{bot_name}: 🌟 Ты отлично справился! Дополнительных заданий нет!")

                            print(f"{bot_name}: Когда выполнишь всё, напиши 'готово'.")
                            stage = "password"
                    else:
                        print(f"{bot_name}: Оценка должна быть от 1 до 10.")
                else:
                    print(f"{bot_name}: Введи цифру от 1 до 10 или напиши 'нет'.")

        # ========== ЭТАП 4: ПАРОЛЬ ==========
        elif stage == "password":
            if tag == "готово" and prob > 0.75:
                response = get_response_by_tag("готово")
                print(f"{bot_name}: {response}")
                log_conversation(user_input, response)

                print(f"{bot_name}: 🎉 Молодец! Ты выполнил все задания!")
                print(f"{bot_name}: 🔑 Держи пароль: КОМПЬЮТЕР2026")
                print("Мама проверит вечером. Удачи!")

                farewell = get_response_by_tag("выход")
                print(f"{bot_name}: {farewell}")
                log_conversation("(завершение диалога)", farewell)
                break
            else:
                print(f"{bot_name}: Напиши 'готово', когда выполнишь все задания.")


if __name__ == "__main__":
    main()