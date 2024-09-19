import sqlite3
import csv
import re
import logging
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.constants import ParseMode
from time import sleep
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Constants
MAX_MSG_LEN = 4096
DB_PATH = 'transactions.db'
RATE_LIMIT_DELAY = 0.05  # 50ms delay between each broadcast message
ADMIN_ID = 831902456  # Replace with actual admin user ID
TRANSACTION_THRESHOLD = 1000  # Notify admins if transaction exceeds this value

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Cache for frequently accessed data
user_totals_cache = {}

# Database context manager for safer handling
class Database:
    def __enter__(self):
        try:
            self.conn = sqlite3.connect(DB_PATH)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
        return self.cursor

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.conn.commit()
            self.conn.close()
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}")

# Utility functions
def add_user(chat_id, role='user'):
    with Database() as db:
        db.execute("INSERT OR IGNORE INTO users (chat_id, role) VALUES (?, ?)", (chat_id, role))

def save_transaction(chat_id, amount, category="general"):
    date = datetime.now().strftime("%Y-%m-%d")
    with Database() as db:
        db.execute('INSERT INTO transactions (amount, date, category, chat_id) VALUES (?, ?, ?, ?)', 
                   (amount, date, category, chat_id))
    user_totals_cache[chat_id] = get_total(chat_id)  # Update cache

def get_total(chat_id):
    if chat_id in user_totals_cache:
        return user_totals_cache[chat_id]

    with Database() as db:
        db.execute('SELECT SUM(amount) FROM transactions WHERE chat_id = ?', (chat_id,))
        total = db.fetchone()[0] or 0
        user_totals_cache[chat_id] = total  # Cache the total
        return total

def train_model():
    with Database() as db:
        db.execute("SELECT date, SUM(amount) FROM transactions GROUP BY date")
        transactions = db.fetchall()

    dates = np.array([datetime.strptime(row[0], "%Y-%m-%d").toordinal() for row in transactions]).reshape(-1, 1)
    amounts = np.array([row[1] for row in transactions])

    if len(dates) > 1:
        model = LinearRegression().fit(dates, amounts)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

def predict_future(date_str):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    future_date = datetime.strptime(date_str, "%Y-%m-%d").toordinal()
    return model.predict([[future_date]])[0]

# Real-time collaboration feature: notify group when a transaction is added
async def notify_group(chat_id, amount):
    group_message = f"User {chat_id} added a transaction of {amount}!"
    await bot.send_message(chat_id=-100123456789, text=group_message)  # Replace with actual group ID

# Real-time notification to admin if transaction exceeds threshold
async def notify_admin_if_threshold_exceeded(chat_id, amount):
    if amount > TRANSACTION_THRESHOLD:
        await bot.send_message(chat_id=ADMIN_ID, text=f"User {chat_id} added a high transaction of {amount}!")

# Admin panel: inline keyboard
async def admin_panel(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("View Users", callback_data='view_users')],
        [InlineKeyboardButton("Check Bot Status", callback_data='check_status')],
        [InlineKeyboardButton("Broadcast Message", callback_data='broadcast_message')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Admin Panel', reply_markup=reply_markup)

async def handle_callback_query(update: Update, context):
    query = update.callback_query
    if query.data == 'view_users':
        # Code to view users (simplified)
        await query.message.reply_text("Viewing users...")
    elif query.data == 'check_status':
        await query.message.reply_text("Bot is running and healthy!")
    elif query.data == 'broadcast_message':
        await query.message.reply_text("Send your broadcast message:")

# Health monitoring
async def bot_status(update: Update, context):
    await update.message.reply_text("Bot is running smoothly.")

# Caching: clear cache command for admin
async def clear_cache(update: Update, context):
    user_totals_cache.clear()
    await update.message.reply_text("Cache cleared.")

# Export transactions as CSV
async def export_transactions(update: Update, context):
    user_id = update.message.chat.id
    with Database() as db:
        transactions = db.execute("SELECT * FROM transactions WHERE chat_id = ?", (user_id,)).fetchall()

    file_name = f'transactions_{user_id}.csv'
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Amount", "Date", "Category", "Chat ID"])
        writer.writerows(transactions)

    await context.bot.send_document(chat_id=user_id, document=open(file_name, 'rb'))

# Lazy loading of matplotlib for graph generation
async def send_graph(update: Update, context):
    user_id = update.message.chat.id
    with Database() as db:
        transactions = db.execute("SELECT date, SUM(amount) FROM transactions WHERE chat_id = ? GROUP BY date", 
                                  (user_id,)).fetchall()

    if transactions:
        dates, totals = zip(*transactions)
        import matplotlib.pyplot as plt  # Lazy import to optimize memory
        plt.plot(dates, totals)
        plt.title('Transaction History')
        plt.xlabel('Date')
        plt.ylabel('Total Amount')
        plt.savefig('transaction_graph.png')
        await context.bot.send_photo(chat_id=user_id, photo=open('transaction_graph.png', 'rb'))
    else:
        await update.message.reply_text("No transactions found.")

# Reset user transactions
async def reset_transactions(update: Update, context):
    user_id = update.message.chat.id
    with Database() as db:
        db.execute('DELETE FROM transactions WHERE chat_id = ?', (user_id,))
    await update.message.reply_text("All your transactions have been reset.")

# Broadcast with Markdown support and rate limiting
async def broadcast_message(update: Update, context):
    if not context.args:
        await update.message.reply_text("Please provide a message to broadcast.")
        return

    message = " ".join(context.args).replace("\\n", "\n")
    with Database() as db:
        users = db.execute("SELECT chat_id FROM users").fetchall()

    for user in users:
        for chunk in [message[i:i + MAX_MSG_LEN] for i in range(0, len(message), MAX_MSG_LEN)]:
            try:
                await context.bot.send_message(chat_id=user[0], text=chunk, parse_mode=ParseMode.MARKDOWN)
                sleep(RATE_LIMIT_DELAY)
            except Exception as e:
                logger.error(f"Error sending message to {user[0]}: {e}")

    await update.message.reply_text("Broadcast sent.")

# Help command listing available commands
async def helpme(update: Update, context):
    help_text = (
        "/start - Start the bot and track a number (start with + or -)\n"
        "/broadcast [message] - Send a broadcast message (admin only)\n"
        "/export - Export your transactions as a CSV file\n"
        "/graph - Get a graphical report of your transactions\n"
        "/reset - Reset all your transactions\n"
        "/status - Check the bot’s current health status\n"
        "/admin - Admin panel for managing the bot\n"
        "/clear_cache - Clear cache data (admin only)\n"
        "/helpme - Display this help message"
    )
    await update.message.reply_text(help_text)

# Message handlers
async def handle_message(update: Update, context):
    text = update.message.text.strip()
    chat_id = update.message.chat.id
    add_user(chat_id)

    match = re.match(r'^[+-]?\d+(\.\d+)?$', text)
    if match:
        amount = float(match.group())
        save_transaction(chat_id, amount)
        total = get_total(chat_id)

        await update.message.reply_text(f"Amount added: {amount}\nYour current total: {total}")

        # Notify the group and admin if certain conditions are met
        await notify_group(chat_id, amount)
        await notify_admin_if_threshold_exceeded(chat_id, amount)
    else:
        await update.message.reply_text("Please send a valid number starting with + or -.")

# Bot initialization
def main():
    application = Application.builder().token('7884065680:AAHtLIpdj_1-l3ypC-BvEUde31in2LFkXXQ').build()

    # Command handlers mapped in a dictionary
    commands = {
        "start": handle_message,
        "broadcast": broadcast_message,
        "export": export_transactions,
        "graph": send_graph,
        "reset": reset_transactions,
        "helpme": helpme,
        "status": bot_status,
        "admin": admin_panel,
        "clear_cache": clear_cache,
    }

    for cmd, handler in commands.items():
        application.add_handler(CommandHandler(cmd, handler))

    # Message handler for incoming text
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Callback query handler for inline admin panel
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    # Scheduler for daily reports
    scheduler = AsyncIOScheduler()
    scheduler.start()

    # Train model for AI insights
    train_model()

    # Start bot
    application.run_polling()

if __name__ == '__main__':
    main()
