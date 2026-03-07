from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)
from rag_engine import (
    process_document, process_url, process_urls,
    ask_question, scan_website, clear_history
)
from user_manager import get_user_sources, get_user_stats, delete_source, clear_user_data
from document_loader import get_supported_extensions
from config import DATA_DIR, MAX_FILE_SIZE_MB
import os

def get_main_menu():
    """Create the main menu buttons."""
    keyboard = [
        [
            InlineKeyboardButton("📄 Add Document", callback_data="add_doc"),
            InlineKeyboardButton("🌐 Add Website", callback_data="add_web"),
        ],
        [
            InlineKeyboardButton("❓ Ask Question", callback_data="ask"),
            InlineKeyboardButton("📚 My Knowledge Base", callback_data="my_kb"),
        ],
        [
            InlineKeyboardButton("❔ Help", callback_data="help"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    user = update.effective_user

    context.user_data["state"] = "idle"
    clear_history(user.id)

    await update.message.reply_text(
        f"👋 Welcome {user.first_name}!\n\n"
        f"I'm your personal study assistant. "
        f"Upload documents or websites, then ask me anything about them!\n\n"
        f"Your data is private — only you can see it.\n\n"
        f"What would you like to do?",
        reply_markup=get_main_menu()
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /help command."""
    await update.message.reply_text(
        "❔ How to use this bot:\n\n"
        "1️⃣ Add your study materials:\n"
        "   • Send me PDF, TXT, DOCX, or CSV files\n"
        "   • Send me a website URL\n\n"
        "2️⃣ Ask questions:\n"
        "   • Click 'Ask Question' or just type your question\n"
        "   • I'll find the answer from YOUR materials\n\n"
        "3️⃣ Manage your data:\n"
        "   • View what's in your knowledge base\n"
        "   • Delete specific sources or clear everything\n\n"
        "Your data is completely private — only you can see it!",
        reply_markup=get_main_menu()
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all button clicks."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    action = query.data

    if action == "menu":
        context.user_data["state"] = "idle"
        await query.edit_message_text(
            "What would you like to do?",
            reply_markup=get_main_menu()
        )

    elif action == "add_doc":
        context.user_data["state"] = "waiting_for_doc"
        await query.edit_message_text(
            "📄 Send me a file! I support:\n\n"
            "• PDF documents (.pdf)\n"
            "• Text files (.txt)\n"
            "• Word documents (.docx)\n"
            "• CSV files (.csv)\n\n"
            f"Maximum file size: {MAX_FILE_SIZE_MB}MB",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")]
            ])
        )

    elif action == "add_web":
        context.user_data["state"] = "waiting_for_url"
        await query.edit_message_text(
            "🌐 Send me a URL and I'll scan it for content.\n\n"
            "Example: https://example.com/course-notes",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")]
            ])
        )

    elif action == "ask":
        context.user_data["state"] = "asking_question"
        await query.edit_message_text(
            "❓ Go ahead, ask me anything about your materials!\n\n"
            "Just type your question.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")]
            ])
        )

    elif action == "my_kb":
        await show_knowledge_base(query, user_id)

    elif action == "help":
        await query.edit_message_text(
            "❔ How to use this bot:\n\n"
            "1️⃣ Add your study materials (PDFs, websites)\n"
            "2️⃣ Click 'Ask Question'\n"
            "3️⃣ Type your question\n"
            "4️⃣ I'll find the answer from YOUR materials\n\n"
            "Your data is private — only you can see it!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")]
            ])
        )

    elif action == "clear_all":
        context.user_data["state"] = "confirm_clear"
        await query.edit_message_text(
            "⚠️ Are you sure you want to delete ALL your data?\n\n"
            "This cannot be undone!",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Yes, Delete Everything", callback_data="confirm_clear_yes"),
                    InlineKeyboardButton("❌ Cancel", callback_data="my_kb"),
                ]
            ])
        )

    elif action == "confirm_clear_yes":
        clear_user_data(user_id)
        context.user_data["state"] = "idle"
        await query.edit_message_text(
            "🗑️ All your data has been deleted.\n\n"
            "You can start fresh by adding new materials!",
            reply_markup=get_main_menu()
        )

    elif action.startswith("delete_"):
        source_name = action.replace("delete_", "")
        delete_source(user_id, source_name)
        await show_knowledge_base(query, user_id, message="✅ Deleted successfully!\n\n")


async def show_knowledge_base(query, user_id, message=""):
    """Show the user their knowledge base contents."""
    sources = get_user_sources(user_id)

    if not sources:
        await query.edit_message_text(
            f"{message}📚 Your knowledge base is empty!\n\n"
            "Add documents or websites to get started.",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📄 Add Document", callback_data="add_doc"),
                    InlineKeyboardButton("🌐 Add Website", callback_data="add_web"),
                ],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")]
            ])
        )
        return

    stats = get_user_stats(user_id)

    text = f"{message}📚 Your Knowledge Base:\n\n"
    keyboard = []

    for i, source in enumerate(sources):
        icon = "📄" if source["type"] == "document" else "🌐"
        name = source["name"]
        short_name = name[:30] + "..." if len(name) > 30 else name
        text += f"{i+1}. {icon} {short_name} ({source['chunks']} chunks)\n"

        keyboard.append([
            InlineKeyboardButton(
                f"🗑️ Delete: {short_name}",
                callback_data=f"delete_{name}"
            )
        ])

    text += f"\nTotal: {stats['total_chunks']} chunks from {stats['total_sources']} sources"

    keyboard.append([InlineKeyboardButton("🗑️ Clear All", callback_data="clear_all")])
    keyboard.append([InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")])

    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle when a user sends a file."""
    user_id = update.effective_user.id
    document = update.message.document

    if not document:
        return

    file_name = document.file_name or "unknown_file"
    file_ext = os.path.splitext(file_name)[1].lower()
    supported = get_supported_extensions()

    if file_ext not in supported:
        await update.message.reply_text(
            f"❌ Unsupported file type: {file_ext}\n\n"
            f"I support: {', '.join(supported)}",
            reply_markup=get_main_menu()
        )
        return

    file_size_mb = document.file_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        await update.message.reply_text(
            f"❌ File too large ({file_size_mb:.1f}MB)\n\n"
            f"Maximum size: {MAX_FILE_SIZE_MB}MB",
            reply_markup=get_main_menu()
        )
        return

    await update.message.reply_text("⏳ Processing your file...")

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, f"{user_id}_{file_name}")

        tg_file = await document.get_file()
        await tg_file.download_to_drive(file_path)

        result = process_document(user_id, file_path, file_name)

        os.remove(file_path)

        if result["success"]:
            await update.message.reply_text(
                f"✅ Added {result['chunks_added']} chunks from {file_name}!",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("📄 Add More", callback_data="add_doc"),
                        InlineKeyboardButton("❓ Ask Question", callback_data="ask"),
                    ],
                    [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
                ])
            )
        else:
            await update.message.reply_text(
                f"❌ Error processing {file_name}: {result['error']}",
                reply_markup=get_main_menu()
            )

    except Exception as e:
        await update.message.reply_text(
            f"❌ Something went wrong: {str(e)}",
            reply_markup=get_main_menu()
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route text messages based on current state."""
    user_id = update.effective_user.id
    text = update.message.text or ""
    state = context.user_data.get("state", "asking_question")

    if state == "waiting_for_url" and text.startswith("http"):
        await handle_url(update, context)

    elif state == "selecting_links":
        await handle_link_selection(update, context)

    else:
        await handle_question(update, context)


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle when a user sends a URL."""
    user_id = update.effective_user.id
    url = update.message.text.strip()

    await update.message.reply_text("🔍 Scanning page for links...")

    links = scan_website(url)

    if not links:
        await update.message.reply_text("No other links found. Adding this page only...")
        result = process_url(user_id, url)

        if result["success"]:
            await update.message.reply_text(
                f"✅ Added {result['chunks_added']} chunks from this page!",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("🌐 Add More", callback_data="add_web"),
                        InlineKeyboardButton("❓ Ask Question", callback_data="ask"),
                    ],
                    [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
                ])
            )
        else:
            await update.message.reply_text(
                f"❌ Error: {result['error']}",
                reply_markup=get_main_menu()
            )
        return

    context.user_data["pending_links"] = links
    context.user_data["state"] = "selecting_links"

    message = f"Found {len(links)} links:\n\n"
    for i, link in enumerate(links):
        icon = "🔵" if link["is_internal"] else "🌐"
        title = link["title"][:50] or link["url"][:50]
        message += f"{i+1}. {icon} {title}\n"

    message += "\nReply with numbers to include (e.g., 1, 3, 5)\n"
    message += "Or type 'all' for everything.\n"
    message += "Type 'cancel' to go back."

    await update.message.reply_text(message)

async def handle_link_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle when user selects which links to scrape."""
    user_id = update.effective_user.id
    text = update.message.text.strip().lower()
    links = context.user_data.get("pending_links", [])

    if not links:
        await update.message.reply_text(
            "No links to select from. Try sending a URL again.",
            reply_markup=get_main_menu()
        )
        return

    if text == "cancel":
        context.user_data["state"] = "idle"
        await update.message.reply_text(
            "Cancelled!",
            reply_markup=get_main_menu()
        )
        return

    if text == "all":
        selected = links
    else:
        try:
            numbers = [int(n.strip()) - 1 for n in text.split(",")]
            selected = [links[n] for n in numbers if 0 <= n < len(links)]
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Invalid input. Use numbers separated by commas (e.g., 1, 3, 5)\n"
                "Or type 'all' or 'cancel'."
            )
            return

    if not selected:
        await update.message.reply_text("No valid links selected. Try again.")
        return

    urls = [link["url"] for link in selected]
    await update.message.reply_text(f"📥 Scraping {len(urls)} pages... This may take a moment.")

    results = process_urls(user_id, urls)

    summary = ""
    total_chunks = 0
    for result in results:
        if result["success"]:
            summary += f"✅ {result['source'][:40]}: {result['chunks_added']} chunks\n"
            total_chunks += result["chunks_added"]
        else:
            summary += f"❌ {result['source'][:40]}: {result['error']}\n"

    summary += f"\nTotal: {total_chunks} chunks added"

    context.user_data["state"] = "idle"
    context.user_data["pending_links"] = []

    await update.message.reply_text(
        summary,
        reply_markup=InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🌐 Add More", callback_data="add_web"),
                InlineKeyboardButton("❓ Ask Question", callback_data="ask"),
            ],
            [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
        ])
    )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle when a user asks a question."""
    user_id = update.effective_user.id
    question = update.message.text

    if not question or len(question.strip()) < 3:
        await update.message.reply_text(
            "Please type a longer question!",
            reply_markup=get_main_menu()
        )
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    result = ask_question(user_id, question)

    answer = result["answer"]
    sources = result["sources"]

    response = f"🤖 {answer}"

    if sources:
        response += "\n\n📚 Sources:\n"
        for source in sources:
            short = source[:40] + "..." if len(source) > 40 else source
            response += f"  • {short}\n"

    await update.message.reply_text(
        response,
        reply_markup=InlineKeyboardMarkup([
            [
                InlineKeyboardButton("❓ Ask Another", callback_data="ask"),
                InlineKeyboardButton("🔙 Menu", callback_data="menu"),
            ]
        ])
    )


