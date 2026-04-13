import os
import random
import logging
import imageio
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from stable_baselines3 import PPO
import gymnasium as gym
from dotenv import load_dotenv

# ========== ЗАГРУЗКА ТОКЕНА И НАСТРОЙКА ЛОГИРОВАНИЯ ==========
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

if TOKEN is None:
    raise ValueError("❌ Токен не найден! Создайте файл .env с BOT_TOKEN=ваш_токен")

# Определяем корень проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Папка для логов
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "bot.log")

# Настройка логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.INFO)

logger.info("=" * 50)
logger.info("🚀 БОТ ЗАПУСКАЕТСЯ")
logger.info(f"📁 Корень проекта: {PROJECT_ROOT}")
logger.info(f"📁 Логи: {LOG_FILE}")
logger.info("=" * 50)

# ========== ПУТИ К МОДЕЛЯМ ==========
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
logger.info(f"📁 Папка с моделями: {MODELS_DIR}")

MODELS = {
    "novice": os.path.join(MODELS_DIR, "2048_ai_1M.zip"),
    "amateur": os.path.join(MODELS_DIR, "2048_ai_2M.zip"),
    "pro": os.path.join(MODELS_DIR, "2048_ai_3M.zip"),
}

# Папка для временных GIF
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"📁 Папка для временных файлов: {TEMP_DIR}")

# ========== ЗАГРУЗКА МОДЕЛЕЙ ==========
logger.info("=" * 50)
logger.info("🚀 ЗАГРУЗКА МОДЕЛЕЙ")

models = {}
for level, path in MODELS.items():
    logger.info(f"📁 {level}: {path}")
    if os.path.exists(path):
        try:
            models[level] = PPO.load(path)
            logger.info(f"   ✅ Загружена!")
        except Exception as e:
            logger.error(f"   ❌ Ошибка загрузки: {e}")
            models[level] = None
    else:
        logger.error(f"   ❌ Файл не найден!")
        models[level] = None

ai_scores = {}
ai_levels = {}
games = {}


# ========== ОБЁРТКА ДЛЯ СРЕДЫ (ДЛЯ ИИ) ==========
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(256,), dtype=np.float32)

    def observation(self, obs):
        return obs.flatten()


# ========== ИГРА ДЛЯ ЧЕЛОВЕКА ==========
class Game2048:
    def __init__(self, mode="solo"):
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.moves = 0
        self.mode = mode
        self.add_tile()
        self.add_tile()

    def add_tile(self):
        empty = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        if empty:
            i, j = random.choice(empty)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def move(self, direction):
        old_board = [row[:] for row in self.board]

        if direction == 'left':
            for i in range(4):
                row = self.board[i]
                row = [x for x in row if x != 0]
                for j in range(len(row) - 1):
                    if row[j] == row[j + 1]:
                        row[j] *= 2
                        self.score += row[j]
                        row[j + 1] = 0
                row = [x for x in row if x != 0]
                row += [0] * (4 - len(row))
                self.board[i] = row

        elif direction == 'right':
            for i in range(4):
                row = self.board[i][::-1]
                row = [x for x in row if x != 0]
                for j in range(len(row) - 1):
                    if row[j] == row[j + 1]:
                        row[j] *= 2
                        self.score += row[j]
                        row[j + 1] = 0
                row = [x for x in row if x != 0]
                row += [0] * (4 - len(row))
                self.board[i] = row[::-1]

        elif direction == 'up':
            for j in range(4):
                col = [self.board[i][j] for i in range(4)]
                col = [x for x in col if x != 0]
                for i in range(len(col) - 1):
                    if col[i] == col[i + 1]:
                        col[i] *= 2
                        self.score += col[i]
                        col[i + 1] = 0
                col = [x for x in col if x != 0]
                col += [0] * (4 - len(col))
                for i in range(4):
                    self.board[i][j] = col[i]

        elif direction == 'down':
            for j in range(4):
                col = [self.board[i][j] for i in range(4)][::-1]
                col = [x for x in col if x != 0]
                for i in range(len(col) - 1):
                    if col[i] == col[i + 1]:
                        col[i] *= 2
                        self.score += col[i]
                        col[i + 1] = 0
                col = [x for x in col if x != 0]
                col += [0] * (4 - len(col))
                col = col[::-1]
                for i in range(4):
                    self.board[i][j] = col[i]

        if self.board != old_board:
            self.moves += 1
            self.add_tile()
            return True
        return False

    def game_over(self):
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    return False
                if j < 3 and self.board[i][j] == self.board[i][j + 1]:
                    return False
                if i < 3 and self.board[i][j] == self.board[i + 1][j]:
                    return False
        return True


# ========== РИСОВАНИЕ ПОЛЯ ==========
def draw_board(board):
    size = 320
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    colors = {2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
              16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 59),
              128: (237, 207, 114), 256: (237, 204, 97), 512: (237, 200, 80),
              1024: (237, 197, 63), 2048: (237, 194, 46)}

    for i in range(4):
        for j in range(4):
            x1, y1 = j * 80, i * 80
            x2, y2 = x1 + 80, y1 + 80
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=2)
            val = board[i][j]
            if val:
                draw.rectangle([x1 + 2, y1 + 2, x2 - 2, y2 - 2], fill=colors.get(val, (205, 193, 180)))
                text = str(val)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                draw.text((x1 + (80 - w) // 2, y1 + (80 - h) // 2), text, fill=(0, 0, 0), font=font)

    buf = BytesIO()
    img.save(buf, 'PNG')
    buf.seek(0)
    return buf


# ========== КЛАВИАТУРЫ ==========
def get_main_menu():
    keyboard = [
        [InlineKeyboardButton("🎮 Одиночная игра", callback_data="menu_human")],
        [InlineKeyboardButton("🤖 Игра с ИИ соперником", callback_data="menu_ai")],
        [InlineKeyboardButton("📖 Правила игры", callback_data="menu_rules")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_level_menu():
    keyboard = [
        [InlineKeyboardButton("🟢 НОВИЧОК (лёгкий)", callback_data="level_novice")],
        [InlineKeyboardButton("🟡 ЛЮБИТЕЛЬ (средний)", callback_data="level_amateur")],
        [InlineKeyboardButton("🔴 ПРОФИ (сложный)", callback_data="level_pro")],
        [InlineKeyboardButton("◀️ Назад в меню", callback_data="menu")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_game_keyboard():
    keyboard = [
        [InlineKeyboardButton("⬆️", callback_data="up")],
        [InlineKeyboardButton("⬅️", callback_data="left"), InlineKeyboardButton("➡️", callback_data="right")],
        [InlineKeyboardButton("⬇️", callback_data="down")],
        [InlineKeyboardButton("🔄 Новая игра", callback_data="new")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_after_ai_keyboard():
    keyboard = [
        [InlineKeyboardButton("🎮 Теперь моя очередь", callback_data="after_ai_play")]
    ]
    return InlineKeyboardMarkup(keyboard)


# ========== ПРАВИЛА ИГРЫ ==========
async def show_rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rules_text = """
📖 **ПРАВИЛА ИГРЫ 2048**

━━━━━━━━━━━━━━━━━━━━━━━

### 🎮 **ОДИНОЧНАЯ ИГРА**

👤 Классическая игра 2048, где вы играете сами за себя.

🎯 **Цель:**  
Соединять одинаковые плитки, чтобы получить **2048** (и выше!)

🕹 **Как играть:**

⬆️ ⬅️ ➡️ ⬇️  — двигайте плитки
2️⃣ + 2️⃣ = 4️⃣  — одинаковые сливаются
После каждого хода появляется новая плитка (2 или 4)

🏁 **Конец:** когда поле заполнено и ходов нет

━━━━━━━━━━━━━━━━━━━━━━━

### 🤖 **ИГРА С ИИ СОПЕРНИКОМ**

Сразитесь с искусственным интеллектом!

🎚 **Уровни сложности:**

🟢 НОВИЧОК     ▰▰▰▱▱▱▱▱▱▱  (лёгкий)
🟡 ЛЮБИТЕЛЬ    ▰▰▰▰▰▰▱▱▱▱  (средний)
🔴 ПРОФИ       ▰▰▰▰▰▰▰▰▰▰  (сложный)

⚡ **Как проходит игра:**

1️⃣ 🤖 ИИ играет первым → вы видите запись (GIF) и его счёт  
2️⃣ 👤 Ваша очередь → попробуйте набрать больше очков  
3️⃣ 🏆 Сравнение → объявление победителя

📌 **Важно:**
• У ИИ одна попытка на партию
• Кнопка «🔄 Новая игра» начинает заново

━━━━━━━━━━━━━━━━━━━━━━━

### 💎 **ПАРА СОВЕТОВ**

🔒 Держите большую плитку в углу
⬅️➡️ Двигайте в 2-3 направлениях
🧩 Не гонитесь за каждым слиянием

━━━━━━━━━━━━━━━━━━━━━━━

**УДАЧИ!** 🍀
"""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=rules_text,
        parse_mode="Markdown",
        reply_markup=get_main_menu()
    )
    if update.callback_query:
        await update.callback_query.answer()


# ========== МЕНЮ ВЫБОРА УРОВНЯ ==========
async def choose_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text="🤖 **ВЫБЕРИТЕ УРОВЕНЬ СЛОЖНОСТИ**\n\nС каким соперником хотите сразиться?",
        parse_mode="Markdown",
        reply_markup=get_level_menu()
    )
    await query.answer()


# ========== ИИ ИГРАЕТ ==========
async def play_ai(update: Update, context: ContextTypes.DEFAULT_TYPE, level: str = "amateur"):
    query = update.callback_query
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if context.user_data.get('ai_busy', False):
        await query.answer("⏳ ИИ уже играет! Подождите завершения партии.", show_alert=True)
        return

    await query.answer("ИИ начинает игру! Ожидайте...")

    context.user_data['ai_busy'] = True
    logger.info(f"🎮 ИИ начал игру | user={user_id} | level={level}")

    try:
        ai_levels[user_id] = level
        level_names = {
            "novice": "НОВИЧОК 🟢",
            "amateur": "ЛЮБИТЕЛЬ 🟡",
            "pro": "ПРОФИ 🔴"
        }

        await context.bot.send_message(
            chat_id,
            "🎮 **ИИ В ИГРЕ!**\n\n"
            f"🤖 Уровень: **{level_names[level]}**\n\n"
            "🤖 ▰▰▰▰▰▰▰▰▰▰ 0%\n"
            "🤖 ▰▰▰▰▰▰▱▱▱▱ 60% — анализирую ходы...\n"
            "🤖 ▰▰▰▰▰▰▰▰▰▱ 90% — записываю GIF...\n\n"
            "⏳ **Ожидайте около минуты...**\n ... я играю и записываю свою игру для вас!\n\n"
            "После просмотра GIF...\n ... кнопка «Теперь моя очередь» станет активной.✅\n\n"
            "Попробуйте побить мой результат! 🎯",
            parse_mode="Markdown"
        )

        model = models.get(level)
        if model is None:
            logger.error(f"❌ Модель уровня {level} не загружена!")
            await context.bot.send_message(chat_id, f"❌ Модель уровня {level} не загружена!")
            return

        env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", render_mode="rgb_array")
        env = FlattenObservation(env)

        frames = []
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        MAX_STEPS = 1000

        while not done and step < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
            frame = env.render()
            frames.append(frame)

        env.close()
        ai_scores[user_id] = int(total_reward)
        logger.info(f"🏁 ИИ завершил игру | user={user_id} | score={int(total_reward)} | steps={step}")

        gif_path = os.path.join(TEMP_DIR, f"ai_game_{user_id}.gif")
        imageio.mimsave(gif_path, frames, fps=8, duration=0.125)

        with open(gif_path, 'rb') as f:
            await context.bot.send_video(
                chat_id,
                f,
                caption=f"🤖 **ИИ завершил игру!**\n\n"
                        f"🎚 Уровень: **{level_names[level]}**\n"
                        f"📊 **Результат:** {int(total_reward)} очков\n"
                        f"🎬 **Ходов:** {step}\n\n"
                        f"Хотите попробовать побить рекорд?",
                parse_mode="Markdown",
                reply_markup=get_after_ai_keyboard()
            )

        try:
            os.remove(gif_path)
            logger.info(f"🗑️ Удалён временный файл: {gif_path}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось удалить GIF: {e}")

    except Exception as e:
        logger.error(f"❌ Ошибка в игре ИИ: {e}")
        await context.bot.send_message(chat_id, "❌ Произошла ошибка. Попробуйте позже.")
    finally:
        context.user_data['ai_busy'] = False


# ========== ИГРА С ИИ СОПЕРНИКОМ (человек) ==========
async def play_human(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    logger.info(f"👤 Человек начал игру | user={user_id} | mode=vs_ai")

    ai_result = ai_scores.get(user_id, 0)
    ai_level = ai_levels.get(user_id, "amateur")
    level_names = {
        "novice": "НОВИЧОК 🟢",
        "amateur": "ЛЮБИТЕЛЬ 🟡",
        "pro": "ПРОФИ 🔴"
    }

    if ai_result > 0:
        await context.bot.send_message(
            chat_id,
            f"🤖 **Напоминаю:** ИИ ({level_names[ai_level]}) набрал **{ai_result}** очков!\n\n"
            f"Попробуйте побить этот рекорд! 🎯",
            parse_mode="Markdown"
        )

    games[user_id] = Game2048(mode="vs_ai")
    game = games[user_id]

    await context.bot.send_photo(
        chat_id,
        draw_board(game.board),
        caption=f"🎮 **ИГРА С ИИ СОПЕРНИКОМ**\n\n"
                f"🤖 Уровень ИИ: **{level_names[ai_level]}**\n"
                f"🎯 Цель: набрать больше {ai_result} очков\n\n"
                f"Ход: {game.moves} | Счёт: {game.score}\n\n"
                f"💡 Используйте кнопки со стрелками",
        parse_mode="Markdown",
        reply_markup=get_game_keyboard()
    )

    await query.answer("Ваша очередь! Удачи! 🍀")


# ========== ОДИНОЧНАЯ ИГРА ==========
async def play_solo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    logger.info(f"👤 Человек начал одиночную игру | user={user_id}")

    games[user_id] = Game2048(mode="solo")
    game = games[user_id]

    await context.bot.send_photo(
        chat_id,
        draw_board(game.board),
        caption=f"🎮 **ОДИНОЧНАЯ ИГРА**\n\nХод: {game.moves} | Счёт: {game.score}\n\n💡 Используйте кнопки со стрелками",
        parse_mode="Markdown",
        reply_markup=get_game_keyboard()
    )

    await query.answer()


# ========== ПОКАЗАТЬ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ==========
async def show_vs_result(chat_id, user_id, context, human_score, human_moves):
    ai_score = ai_scores.get(user_id, 0)
    ai_level = ai_levels.get(user_id, "amateur")
    level_names = {
        "novice": "НОВИЧОК 🟢",
        "amateur": "ЛЮБИТЕЛЬ 🟡",
        "pro": "ПРОФИ 🔴"
    }

    if ai_score > human_score:
        winner = "AI"
        winner_text = "🤖 **ПОБЕДИЛ ИИ!** 😢"
    elif human_score > ai_score:
        winner = "HUMAN"
        winner_text = "🎉 **ВЫ ПОБЕДИЛИ!** 🏆"
    else:
        winner = "DRAW"
        winner_text = "🤝 **НИЧЬЯ!** 🎯"

    logger.info(f"🏆 Результат | user={user_id} | AI={ai_score} | Human={human_score} | Winner={winner}")

    result_text = f"""
🎮 **ИТОГОВЫЙ РЕЗУЛЬТАТ**

━━━━━━━━━━━━━━━━━━━━━━━

🤖 **ИИ ({level_names[ai_level]}):** {ai_score} очков

👤 **Вы:** {human_score} очков

━━━━━━━━━━━━━━━━━━━━━━━

{winner_text}

━━━━━━━━━━━━━━━━━━━━━━━
"""
    await context.bot.send_message(chat_id, result_text, parse_mode="Markdown")
    await context.bot.send_message(chat_id, "Выберите режим:", reply_markup=get_main_menu())


# ========== ОБРАБОТЧИК КНОПОК ==========
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    data = query.data

    if data == "menu":
        await context.bot.send_message(chat_id, "Выберите режим:", reply_markup=get_main_menu())
        return

    if data == "menu_human":
        await play_solo(update, context)
        return

    if data == "menu_ai":
        await choose_level(update, context)
        return

    if data == "menu_rules":
        await show_rules(update, context)
        return

    if data == "level_novice":
        await play_ai(update, context, level="novice")
        return
    if data == "level_amateur":
        await play_ai(update, context, level="amateur")
        return
    if data == "level_pro":
        await play_ai(update, context, level="pro")
        return

    if data == "after_ai_play":
        await play_human(update, context)
        return

    if user_id not in games:
        if data == "new":
            await context.bot.send_message(chat_id, "Выберите режим:", reply_markup=get_main_menu())
        else:
            try:
                await query.message.edit_text("❌ Игра не активна. Нажмите 'Новая игра'",
                                              reply_markup=get_game_keyboard())
            except Exception:
                pass
        return

    game = games[user_id]

    if data == "new":
        if game.mode == "vs_ai" and game.score > 0:
            human_score = game.score
            human_moves = game.moves
            await show_vs_result(chat_id, user_id, context, human_score, human_moves)
        else:
            await context.bot.send_message(chat_id, "Выберите режим:", reply_markup=get_main_menu())
        del games[user_id]
        return

    moved = game.move(data)
    if not moved:
        await query.answer("Этот ход невозможен")
        return

    img = draw_board(game.board)
    caption = f"🎮 Ход: {game.moves} | Счёт: {game.score}"

    if game.game_over():
        human_score = game.score
        human_moves = game.moves
        logger.info(f"🏁 Игра окончена | user={user_id} | mode={game.mode} | score={human_score} | moves={human_moves}")

        try:
            await query.message.edit_media(
                media=InputMediaPhoto(img, caption=f"❌ **ИГРА ОКОНЧЕНА!**\n\nХод: {human_moves} | Счёт: {human_score}",
                                      parse_mode="Markdown"),
                reply_markup=None
            )
        except Exception as e:
            logger.warning(f"Не удалось обновить сообщение: {e}")

        if game.mode == "vs_ai":
            await show_vs_result(chat_id, user_id, context, human_score, human_moves)
        else:
            await context.bot.send_message(chat_id, f"🎮 **ИГРА ОКОНЧЕНА!**\n\nХод: {human_moves} | Счёт: {human_score}",
                                           parse_mode="Markdown")
            await context.bot.send_message(chat_id, "Выберите режим:", reply_markup=get_main_menu())

        del games[user_id]
    else:
        try:
            await query.message.edit_media(
                media=InputMediaPhoto(img, caption=caption, parse_mode="Markdown"),
                reply_markup=get_game_keyboard()
            )
        except Exception as e:
            logger.warning(f"Не удалось обновить сообщение: {e}")


# ========== СТАРТ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or "unknown"
    logger.info(f"🆕 Пользователь запустил бота | user={user_id} | username={username}")
    await update.message.reply_text(
        "🤖 **Добро пожаловать в игру 2048!**\n\n"
        "🎮 Одиночная игра\n"
        "🤖 Соревнование с ИИ (3 уровня)\n\n"
        "Выберите режим:",
        parse_mode="Markdown",
        reply_markup=get_main_menu()
    )


# ========== ЗАПУСК ==========
def main():
    logger.info("📡 Создание приложения Telegram...")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_error_handler(lambda update, context: logger.error(f"❌ Ошибка: {context.error}"))
    logger.info("✅ Бот запущен! Напишите /start в Telegram")
    app.run_polling()


if __name__ == "__main__":
    main()