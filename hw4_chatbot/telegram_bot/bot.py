"""
bot.py — Telegram-бот с кнопками для выбора предметов.

Нейросеть используется только для распознавания простых команд:
- приветствие
- готово
- да / подтверждение
- нет / отказ

Вся остальная логика реализована через состояния и кнопки.
"""

import os
import random
import re
import asyncio
import torch
from datetime import datetime
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# Импортируем наши модули
import sys
from pathlib import Path

# Добавляем путь к корневой папке проекта
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import NeuralNet
from src.nltk_utils import bag_of_words, tokenize, load_intents
from src.utils import DATA_DIR, get_latest_model, ensure_dirs, LOGS_DIR

# ==================== 1. ЗАГРУЗКА МОДЕЛИ ====================
# Создаём папки, если их нет
ensure_dirs()

# Определяем устройство (GPU если есть, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем намерения
intents_path = DATA_DIR / "intents.json"
intents = load_intents(intents_path)

# Загружаем последнюю обученную модель
model_path = get_latest_model()
print(f"Загрузка модели из: {model_path}")

data = torch.load(model_path, map_location=device)

# Извлекаем параметры модели
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Создаём модель и загружаем веса
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # Переводим в режим оценки

bot_name = "КиберПатруль"


# ==================== 2. ФУНКЦИИ ДЛЯ РАБОТЫ БОТА ====================

def predict_intent(text: str) -> tuple:
    """Определяет намерение пользователя с помощью нейросети."""
    tokens = tokenize(text)
    bag = bag_of_words(tokens, all_words)
    bag_tensor = torch.tensor(bag, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(bag_tensor)
        _, predicted = torch.max(output, dim=1)
        prob = torch.softmax(output, dim=1)[0][predicted.item()].item()
        tag = tags[predicted.item()]

    return tag, prob


def assign_tasks(avg_score: float) -> list:
    """
    Назначает дополнительные задания в зависимости от среднего балла.
    Главные задания уже выполнены на первом этапе.
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
        tasks.append(random.choice(house_tasks))
        tasks.append(random.choice(study_tasks))
    elif avg_score >= 9:
        return []
    elif avg_score >= 7:
        tasks.append(random.choice(house_tasks))
    elif avg_score >= 5:
        tasks.append(random.choice(house_tasks))
        tasks.append(random.choice(study_tasks))
    else:
        tasks.append(random.choice(house_tasks))
        tasks.append(random.choice(house_tasks))
        tasks.append(random.choice(study_tasks))

    return tasks


def get_grade_message(avg_score: float) -> str:
    """Возвращает сообщение в зависимости от среднего балла."""
    if avg_score >= 9:
        return "🎉 Отлично! Ты молодец! Так держать!"
    elif avg_score >= 7:
        return "👍 Хорошо, но есть куда расти!"
    elif avg_score >= 5:
        return "📚 Неплохо, но нужно подтянуться!"
    else:
        return "⚠️ Низкий балл. Нужно исправляться!"


def get_response_by_tag(tag: str) -> str:
    """Возвращает случайный ответ из intents.json по тегу."""
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Я вас не понял."


def log_conversation(user_id: int, user_message: str, bot_response: str) -> None:
    """Сохраняет диалог в лог-файл."""
    log_file = LOGS_DIR / f"log_{datetime.now().strftime('%Y%m%d')}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | Пользователь {user_id}: {user_message}\n")
        f.write(f"{timestamp} | Бот: {bot_response}\n\n")


# ==================== 3. КЛАСС ДЛЯ ХРАНЕНИЯ СОСТОЯНИЯ ====================

class UserState:
    """
    Хранит состояние диалога для каждого пользователя.
    Telegram-бот должен помнить, на каком этапе находится каждый пользователь.
    """

    def __init__(self):
        self.stage = "main_tasks"  # main_tasks -> subjects -> confirm_subjects -> grades -> password
        self.lessons = []  # Список выбранных предметов
        self.grades = {}  # Словарь оценок {предмет: оценка}
        self.current_subject_index = 0  # Индекс текущего предмета для ввода оценки
        self.subjects_confirmed = False  # Флаг: список предметов подтверждён
        self.tasks_assigned = False  # Флаг: дополнительные задания назначены


# Хранилище состояний всех пользователей
user_states = {}

# ==================== 4. КНОПКИ ДЛЯ ВЫБОРА ПРЕДМЕТОВ ====================

SUBJECTS = [
    "математика",
    "русский язык",
    "английский язык",
    "история",
    "литература",
    "белорусский язык",
    "человек и мир"
]


def get_subjects_keyboard() -> InlineKeyboardMarkup:
    """
    Создаёт клавиатуру с кнопками предметов.
    Кнопки с галочками показывают выбранные предметы.
    """
    keyboard = []
    row = []

    for i, subject in enumerate(SUBJECTS, 1):
        # Здесь мы не можем показать галочки, так как не знаем выбранные предметы
        # Галочки будут добавляться при редактировании сообщения
        row.append(InlineKeyboardButton(f"{i}. {subject}", callback_data=f"subj_{i}"))

        if i % 2 == 0:  # по 2 кнопки в ряд
            keyboard.append(row)
            row = []

    if row:
        keyboard.append(row)

    # Добавляем кнопку "Готово"
    keyboard.append([InlineKeyboardButton("✅ Готово", callback_data="subj_done")])

    return InlineKeyboardMarkup(keyboard)


def get_subjects_text_with_checks(selected_lessons: list) -> str:
    """
    Возвращает текст с отмеченными выбранными предметами.

    Args:
        selected_lessons (list): Список выбранных предметов

    Returns:
        str: Текст с галочками
    """
    text = "📚 Выбери предметы, которые были сегодня:\n\n"

    for i, subject in enumerate(SUBJECTS, 1):
        checked = "✅ " if subject in selected_lessons else "   "
        text += f"{checked}{i}. {subject}\n"

    text += "\nКогда закончишь, нажми 'Готово'."

    return text


# ==================== 5. ОБРАБОТЧИКИ КОМАНД ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды /start"""
    user_id = update.effective_user.id

    # Создаём новое состояние для пользователя
    user_states[user_id] = UserState()

    # Приветствие из intents.json
    greeting = get_response_by_tag("приветствие")
    await update.message.reply_text(greeting)
    log_conversation(user_id, "/start", greeting)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    user_id = update.effective_user.id
    text = update.message.text.strip()

    # Если пользователь новый, создаём состояние
    if user_id not in user_states:
        user_states[user_id] = UserState()

    state = user_states[user_id]

    # Распознаём намерение
    tag, prob = predict_intent(text)

    # ========== ЭТАП 1: ГЛАВНЫЕ ЗАДАНИЯ ==========
    if state.stage == "main_tasks":
        if tag == "готово" and prob > 0.75:
            response = get_response_by_tag("готово")
            await update.message.reply_text(response)
            log_conversation(user_id, text, response)

            # Показываем кнопки для выбора предметов
            await update.message.reply_text(
                "Теперь выбери предметы, которые были сегодня:",
                reply_markup=get_subjects_keyboard()
            )
            state.stage = "subjects"

        elif tag == "отказ" and prob > 0.75:
            response = get_response_by_tag("отказ")
            await update.message.reply_text(response)
            log_conversation(user_id, text, response)
            await update.message.reply_text(
                "Напиши 'готово', когда выполнишь все главные задания."
            )
        else:
            await update.message.reply_text(
                "Напиши 'готово', когда выполнишь все главные задания:\n\n"
                "1. Поесть\n2. Сложить школьную одежду в шкаф\n"
                "3. Помыть посуду\n4. Сделать уроки"
            )

    # ========== ЭТАП 2: ОЦЕНКИ ==========
    elif state.stage == "grades":
        # Проверяем, что есть предметы для оценки
        if not state.lessons:
            await update.message.reply_text(
                "Что-то пошло не так. Давай начнём заново с /start"
            )
            del user_states[user_id]
            return

        subject = state.lessons[state.current_subject_index]

        # Исправленное регулярное выражение: сначала ищем 10, потом 1-9
        digits = re.findall(r'\b(10|[1-9])\b', text)

        if digits:
            grade = int(digits[0])
            if 1 <= grade <= 10:
                state.grades[subject] = grade
                state.current_subject_index += 1

                if state.current_subject_index < len(state.lessons):
                    next_subject = state.lessons[state.current_subject_index]
                    await update.message.reply_text(f"Какая оценка по {next_subject}?")
                else:
                    # Все оценки собраны
                    result = "📊 Вот что я записал:\n"
                    for s, g in state.grades.items():
                        result += f"  {s}: {g}\n"
                    await update.message.reply_text(result)
                    log_conversation(user_id, text, result)

                    # Расчёт среднего балла
                    valid_grades = list(state.grades.values())

                    if valid_grades:
                        avg_score = sum(valid_grades) / len(valid_grades)
                        avg_msg = f"📈 Твой средний балл сегодня: {avg_score:.1f}\n"
                        avg_msg += get_grade_message(avg_score)
                        await update.message.reply_text(avg_msg)

                        # Назначаем дополнительные задания
                        if not state.tasks_assigned:
                            tasks = assign_tasks(avg_score)

                            if tasks:
                                tasks_msg = "📝 Дополнительные задания на сегодня:\n"
                                for i, task in enumerate(tasks, 1):
                                    tasks_msg += f"{i}. {task}\n"
                                tasks_msg += "\nКогда выполнишь всё, напиши 'готово'."
                                await update.message.reply_text(tasks_msg)
                                state.tasks_assigned = True
                                state.stage = "password"
                            else:
                                # Заданий нет — сразу выдаём пароль
                                await update.message.reply_text(
                                    "🎉 Молодец! Ты отлично справился!\n\n"
                                    "🔑 Держи пароль: **КОМПЬЮТЕР2026**\n\n"
                                    "Мама проверит вечером. Удачи!",
                                    parse_mode="Markdown"
                                )
                                # Очищаем состояние
                                del user_states[user_id]
                    else:
                        await update.message.reply_text(
                            "Сегодня не было оценок.\nВ следующий раз постарайся получить пятёрки!"
                        )
                        tasks = assign_tasks(None)
                        if tasks:
                            tasks_msg = "📝 Дополнительные задания на сегодня:\n"
                            for i, task in enumerate(tasks, 1):
                                tasks_msg += f"{i}. {task}\n"
                            tasks_msg += "\nКогда выполнишь всё, напиши 'готово'."
                            await update.message.reply_text(tasks_msg)
                            state.tasks_assigned = True
                        state.stage = "password"
            else:
                await update.message.reply_text("Оценка должна быть от 1 до 10.")
        else:
            await update.message.reply_text("Введи цифру от 1 до 10.")

    # ========== ЭТАП 3: ПАРОЛЬ ==========
    elif state.stage == "password":
        if tag == "готово" and prob > 0.75:
            response = get_response_by_tag("готово")
            await update.message.reply_text(response)
            log_conversation(user_id, text, response)

            await update.message.reply_text(
                "🎉 Молодец! Ты выполнил все задания!\n\n"
                "🔑 Держи пароль: **КОМПЬЮТЕР2026**\n\n"
                "Мама проверит вечером. Удачи!",
                parse_mode="Markdown"
            )
            # Очищаем состояние
            del user_states[user_id]
        else:
            await update.message.reply_text(
                "Напиши 'готово', когда выполнишь все дополнительные задания."
            )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на кнопки"""
    query = update.callback_query
    user_id = update.effective_user.id
    data = query.data

    # Если пользователь новый, создаём состояние
    if user_id not in user_states:
        user_states[user_id] = UserState()

    state = user_states[user_id]

    # ========== ЕСЛИ СПИСОК УЖЕ ПОДТВЕРЖДЁН ==========
    if state.subjects_confirmed:
        await query.answer(
            "✅ Список предметов уже подтверждён! Введи оценки цифрами.",
            show_alert=True
        )
        return

    # ========== ВЫБОР ПРЕДМЕТОВ ==========
    if state.stage == "subjects":
        if data == "subj_done":
            if state.lessons:
                lessons_list = ", ".join(state.lessons)
                # Показываем кнопки подтверждения
                await query.message.reply_text(
                    f"Ты выбрал: {lessons_list}\n\nВсё верно?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("✅ Да", callback_data="confirm_yes")],
                        [InlineKeyboardButton("❌ Нет", callback_data="confirm_no")]
                    ])
                )
                state.stage = "confirm_subjects"
            else:
                await query.answer("Выберите хотя бы один предмет!", show_alert=True)
                return

        elif data.startswith("subj_"):
            num = int(data.split("_")[1])
            subject = SUBJECTS[num - 1]

            if subject in state.lessons:
                state.lessons.remove(subject)
            else:
                state.lessons.append(subject)

            # Обновляем сообщение с отмеченными предметами
            text = get_subjects_text_with_checks(state.lessons)
            await query.edit_message_text(text, reply_markup=get_subjects_keyboard())

    # ========== ПОДТВЕРЖДЕНИЕ СПИСКА ==========
    elif state.stage == "confirm_subjects":
        if data == "confirm_yes":
            state.subjects_confirmed = True

            await query.message.reply_text(
                f"✅ Выбранные предметы: {', '.join(state.lessons)}\n\n"
                "Теперь введи оценки (цифрой от 1 до 10)."
            )
            state.stage = "grades"
            state.current_subject_index = 0
            await query.message.reply_text(f"📝 Какая оценка по {state.lessons[0]}?")

            # Удаляем сообщение с кнопками подтверждения
            await query.message.delete()

        elif data == "confirm_no":
            state.lessons = []
            state.stage = "subjects"

            # Отправляем новое сообщение с кнопками
            text = get_subjects_text_with_checks(state.lessons)
            await query.message.reply_text(text, reply_markup=get_subjects_keyboard())
            # Удаляем сообщение с кнопками подтверждения
            await query.message.delete()

    await query.answer()


# ==================== 6. ЗАПУСК БОТА ====================

def main():
    """Главная функция для запуска Telegram-бота"""

    # Загружаем токен из .env
    load_dotenv()
    token = os.getenv("BOT_TOKEN")

    if not token:
        print("❌ Ошибка: токен не найден. Проверь файл .env")
        return

    print(f"🤖 Запуск Telegram-бота {bot_name}...")

    # Создаём приложение
    app = Application.builder().token(token).build()

    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Бот запущен! Нажми Ctrl+C для остановки.")

    # Запускаем бота
    app.run_polling()


if __name__ == "__main__":
    main()