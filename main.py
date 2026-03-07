from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters
)
from config import TELEGRAM_BOT_TOKEN
from bot import start, help_command, handle_button, handle_message, handle_file


def main():
    print("🤖 Starting Study Assistant Bot...")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("menu", start))

    # Button click handler
    app.add_handler(CallbackQueryHandler(handle_button))

    # File handler (must be before text handler)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    # Text message handler (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot is running! Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

