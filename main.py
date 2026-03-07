import logging
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters
)
from config import TELEGRAM_BOT_TOKEN
from bot import start, help_command, handle_button, handle_message, handle_file

# Set up logging so you can see errors in the terminal
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def error_handler(update, context):
    """Handle unexpected errors so the bot doesn't crash."""
    logger.error(f"Error: {context.error}")

    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Something went wrong. Please try again.\n\n"
                "If the problem continues, send /start to reset."
            )
        except Exception:
            pass


def main():
    print("🤖 Starting Personal Assistant Bot...")

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

    # Global error handler
    app.add_error_handler(error_handler)

    print("✅ Bot is running! Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()