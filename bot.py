import time
import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)
from rag_engine import (
    process_document, process_url, process_urls,
    ask_question, scan_website, clear_history,
    transcribe_audio, text_to_speech, process_image,
    detect_category, ask_question_stream
)
from user_manager import (
    get_user_sources, get_user_stats, 
    delete_source, clear_user_data,
    get_user_collection
)
from document_loader import get_supported_extensions
from config import DATA_DIR, MAX_FILE_SIZE_MB, CATEGORIES
import os

async def send_typing(update):
    """Show 'typing...' indicator in Telegram."""
    await update.effective_chat.send_action("typing")

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
        f"I'm your personal assistant. "
        f"Upload documents or websites, then ask me anything about them!\n\n"
        f"Your data is private — only you can see it.\n\n"
        f"What would you like to do?",
        reply_markup=get_main_menu()
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /help command."""
    await update.message.reply_text(
        "❔ How to use this bot:\n\n"
        "1️⃣ Add your materials:\n"
        "   • Send me PDF, TXT, DOCX, or CSV files\n"
        "   • Send me a website URL\n\n"
        "   • Send me a photo or screenshot 🖼️\n\n"
        "2️⃣ Ask questions:\n"
        "   • Type your question\n"
        "   • Or send a voice message 🎤\n"
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
            "• CSV files (.csv)\n"
            "• Photos and screenshots 🖼️\n\n"
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
            "Type your question or send a voice message 🎤",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")]
            ])
        )

    elif action == "my_kb":
        await show_knowledge_base(query, user_id)

    elif action == "help":
        await query.edit_message_text(
            "❔ How to use this bot:\n\n"
            "1️⃣ Add your materials (PDFs, websites)\n"
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
    elif action.startswith("retag_"):
        source_name = action.replace("retag_", "")
        context.user_data["retag_source"] = source_name

        keyboard = []
        row = []
        for key, info in CATEGORIES.items():
            row.append(InlineKeyboardButton(
                info["label"],
                callback_data=f"setcat_{key}_{source_name}"
            ))
            if len(row) == 2:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)

        keyboard.append([InlineKeyboardButton("🔙 Cancel", callback_data="menu")])

        await query.edit_message_text(
            "🏷️ Choose the correct category:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif action.startswith("setcat_"):
        parts = action.replace("setcat_", "").split("_", 1)
        new_category = parts[0]
        source_name = parts[1] if len(parts) > 1 else ""

        # Update metadata for all chunks from this source
        collection = get_user_collection(user_id)
        all_data = collection.get(include=["metadatas", "documents"])

        for i, metadata in enumerate(all_data["metadatas"]):
            if metadata.get("source") == source_name:
                metadata["category"] = new_category
                collection.update(
                    ids=[all_data["ids"][i]],
                    metadatas=[metadata]
                )

        cat_label = CATEGORIES.get(new_category, {}).get("label", "📝 General")

        await query.edit_message_text(
            f"✅ Updated category to {cat_label}!",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📄 Add More", callback_data="add_doc"),
                    InlineKeyboardButton("❓ Ask Question", callback_data="ask"),
                ],
                [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
            ])
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

    # Check 1: Supported file type?
    if file_ext not in supported:
        await update.message.reply_text(
            f"❌ Unsupported file type: {file_ext}\n\n"
            f"I support: {', '.join(supported)}",
            reply_markup=get_main_menu()
        )
        return

    # Check 2: File size OK?
    file_size_mb = document.file_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        await update.message.reply_text(
            f"❌ File too large ({file_size_mb:.1f}MB)\n\n"
            f"Maximum size: {MAX_FILE_SIZE_MB}MB",
            reply_markup=get_main_menu()
        )
        return

    # Show processing indicator
    processing_msg = await update.message.reply_text(
        f"⏳ Processing {file_name}...\n"
        f"This may take a moment."
    )

    file_path = None

    try:
        await send_typing(update)

        # Download file
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, f"{user_id}_{file_name}")

        tg_file = await document.get_file()
        await tg_file.download_to_drive(file_path)

        # Check 3: File not empty?
        if os.path.getsize(file_path) == 0:
            await processing_msg.edit_text(
                f"❌ {file_name} appears to be empty.",
                reply_markup=get_main_menu()
            )
            return

        await send_typing(update)

        # Process the document
        result = process_document(user_id, file_path, file_name)

        if result["success"]:
            cat_key = result.get("category", "general")
            cat_label = CATEGORIES.get(cat_key, {}).get("label", "📝 General")

            await processing_msg.edit_text(
                f"✅ Successfully processed {file_name}!\n\n"
                f"📊 {result['chunks_added']} chunks added\n"
                f"🏷️ Category: {cat_label}\n\n"
                f"Is the category correct?",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("✅ Yes", callback_data="ask"),
                        InlineKeyboardButton("🏷️ Change", callback_data=f"retag_{file_name}"),
                    ],
                    [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
                ])
            )
        else:
            await processing_msg.edit_text(
                f"❌ Could not process {file_name}\n\n"
                f"Reason: {result['error']}\n\n"
                f"Please check the file and try again.",
                reply_markup=get_main_menu()
            )

    except Exception as e:
        await processing_msg.edit_text(
            f"❌ Something went wrong while processing {file_name}\n\n"
            f"Please try again. If the problem continues, try a different file.",
            reply_markup=get_main_menu()
        )

    finally:
        # Always clean up the temp file, even if something crashed
        if file_path and os.path.exists(file_path):
            os.remove(file_path)



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

    # Basic URL validation
    if not url.startswith("http://") and not url.startswith("https://"):
        await update.message.reply_text(
            "❌ That doesn't look like a valid URL.\n\n"
            "URLs should start with http:// or https://",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")]
            ])
        )
        return

    processing_msg = await update.message.reply_text("🔍 Scanning page for links...")

    try:
        await send_typing(update)

        links = scan_website(url)

        if not links:
            await processing_msg.edit_text("No other links found. Adding this page only...")
            await send_typing(update)

            result = process_url(user_id, url)

            if result["success"]:
                cat_key = result.get("category", "general")
                cat_label = CATEGORIES.get(cat_key, {}).get("label", "📝 General")

                await processing_msg.edit_text(
                    f"✅ Added {result['chunks_added']} chunks from this page!\n"
                    f"🏷️ Category: {cat_label}\n\n"
                    f"Is the category correct?",
                    reply_markup=InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("✅ Yes", callback_data="ask"),
                            InlineKeyboardButton("🏷️ Change", callback_data=f"retag_{url[:50]}"),
                        ],
                        [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
                    ])
                )
            else:
                await processing_msg.edit_text(
                    f"❌ Could not scrape this page.\n\n"
                    f"Reason: {result['error']}\n\n"
                    f"The website might be blocking automated access.",
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

        await processing_msg.edit_text(message)

    except Exception as e:
        await processing_msg.edit_text(
            "❌ Could not access this URL.\n\n"
            "Possible reasons:\n"
            "• The URL might be incorrect\n"
            "• The website might be down\n"
            "• The website might be blocking bots\n\n"
            "Please check the URL and try again.",
            reply_markup=get_main_menu()
        )


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
        context.user_data["pending_links"] = []
        await update.message.reply_text(
            "Cancelled!",
            reply_markup=get_main_menu()
        )
        return

    # Parse selection
    if text == "all":
        selected = links
    else:
        try:
            numbers = [int(n.strip()) - 1 for n in text.split(",")]
            selected = [links[n] for n in numbers if 0 <= n < len(links)]
        except (ValueError, IndexError):
            await update.message.reply_text(
                "❌ Invalid input.\n\n"
                "Use numbers separated by commas: 1, 3, 5\n"
                "Or type 'all' for everything\n"
                "Or type 'cancel' to go back"
            )
            return

    if not selected:
        await update.message.reply_text(
            "❌ No valid links selected.\n\n"
            "The numbers should be from the list above."
        )
        return

    urls = [link["url"] for link in selected]
    processing_msg = await update.message.reply_text(
        f"📥 Scraping {len(urls)} pages...\n"
        f"This may take a moment."
    )

    try:
        await send_typing(update)
        results = process_urls(user_id, urls)

        summary = ""
        total_chunks = 0
        detected_categories = []

        for result in results:
            if result["success"]:
                short_source = result['source'][:40]
                cat_key = result.get("category", "general")
                cat_label = CATEGORIES.get(cat_key, {}).get("label", "📝 General")
                summary += f"✅ {short_source}: {result['chunks_added']} chunks ({cat_label})\n"
                total_chunks += result["chunks_added"]
                if cat_key not in detected_categories:
                    detected_categories.append(cat_key)
            else:
                short_source = result['source'][:40]
                summary += f"❌ {short_source}: failed\n"

        summary += f"\n📊 Total: {total_chunks} chunks added"

        if detected_categories:
            cat_labels = [CATEGORIES.get(c, {}).get("label", c) for c in detected_categories]
            summary += f"\n🏷️ Categories: {', '.join(cat_labels)}"

        context.user_data["state"] = "idle"
        context.user_data["pending_links"] = []

        await processing_msg.edit_text(
            summary,
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🌐 Add More", callback_data="add_web"),
                    InlineKeyboardButton("❓ Ask Question", callback_data="ask"),
                ],
                [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
            ])
        )

    except Exception as e:
        context.user_data["state"] = "idle"
        context.user_data["pending_links"] = []
        await processing_msg.edit_text(
            "❌ Something went wrong while scraping.\n\n"
            "Please try again.",
            reply_markup=get_main_menu()
        )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle when a user asks a question with streaming response."""
    user_id = update.effective_user.id
    question = update.message.text

    if not question or len(question.strip()) < 3:
        await update.message.reply_text(
            "Please type a longer question!",
            reply_markup=get_main_menu()
        )
        return

    stats = get_user_stats(user_id)
    if stats["total_chunks"] == 0:
        await update.message.reply_text(
            "📚 Your knowledge base is empty!\n\n"
            "Add some documents or websites first, then I can answer your questions.",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📄 Add Document", callback_data="add_doc"),
                    InlineKeyboardButton("🌐 Add Website", callback_data="add_web"),
                ],
                [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
            ])
        )
        return

    # Send initial message
    streaming_msg = await update.message.reply_text("🤖 Thinking...")

    try:
        full_text = ""
        metadata = None
        last_edit_time = time.time()
        edit_interval = 1.0  # edit message every 1 second

        for chunk in ask_question_stream(user_id, question):
            if isinstance(chunk, dict):
                # Final metadata
                metadata = chunk
            else:
                # Text token
                full_text += chunk

                # Edit message every 1 second (Telegram rate limit)
                now = time.time()
                if now - last_edit_time >= edit_interval:
                    display = f"🤖 {full_text}"
                    if len(display) > 4000:
                        display = display[:4000]
                    
                    try:
                        await streaming_msg.edit_text(display + " ▌")
                    except Exception:
                        pass  # ignore edit errors (message unchanged, etc.)
                    
                    last_edit_time = now
                    await asyncio.sleep(0.1)  # small delay to not block

        # Final edit with complete answer + sources + buttons
        response = f"🤖 {full_text}"

        if metadata:
            sources = metadata.get("sources", [])
            categories_searched = metadata.get("categories_searched", ["all"])

            if sources:
                response += "\n\n📚 Sources:\n"
                for source in sources:
                    short = source[:40] + "..." if len(source) > 40 else source
                    response += f"  • {short}\n"

            if categories_searched and categories_searched != ["all"]:
                cat_labels = [CATEGORIES.get(c, {}).get("label", c) for c in categories_searched]
                response += f"\n🔍 Searched: {', '.join(cat_labels)}"

        if len(response) > 4000:
            response = response[:4000] + "\n\n... (truncated)"

        await streaming_msg.edit_text(
            response,
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("❓ Ask Another", callback_data="ask"),
                    InlineKeyboardButton("🔙 Menu", callback_data="menu"),
                ]
            ])
        )

    except Exception as e:
        await streaming_msg.edit_text(
            "❌ Something went wrong. Please try again.",
            reply_markup=get_main_menu()
        )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle when a user sends a voice message."""
    user_id = update.effective_user.id
    voice = update.message.voice

    if not voice:
        return

    if voice.duration > 120:
        await update.message.reply_text(
            "❌ Voice message too long (max 2 minutes).\n\n"
            "Please send a shorter message.",
            reply_markup=get_main_menu()
        )
        return

    stats = get_user_stats(user_id)
    if stats["total_chunks"] == 0:
        await update.message.reply_text(
            "📚 Your knowledge base is empty!\n\n"
            "Add some documents or websites first, then I can answer your questions.",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📄 Add Document", callback_data="add_doc"),
                    InlineKeyboardButton("🌐 Add Website", callback_data="add_web"),
                ],
                [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
            ])
        )
        return

    processing_msg = await update.message.reply_text("🎤 Converting your voice message to text...")

    voice_input_path = None
    voice_output_path = None

    try:
        await send_typing(update)

        # Download the voice file
        os.makedirs(DATA_DIR, exist_ok=True)
        voice_input_path = os.path.join(DATA_DIR, f"{user_id}_voice_in.ogg")

        tg_file = await voice.get_file()
        await tg_file.download_to_drive(voice_input_path)

        # Transcribe speech to text
        transcription = transcribe_audio(voice_input_path)

        if not transcription["success"]:
            await processing_msg.edit_text(
                "❌ Could not understand the voice message.\n\n"
                "Please try again or type your question instead.",
                reply_markup=get_main_menu()
            )
            return

        question = transcription["text"]

        await processing_msg.edit_text(
            f"🎤 I heard: \"{question}\"\n\n"
            f"🔍 Searching your materials..."
        )

        await send_typing(update)

        # RAG pipeline
        result = ask_question(user_id, question)

        answer = result["answer"]
        sources = result["sources"]

        # Build text response
        response = f"🎤 You asked: \"{question}\"\n\n"
        response += f"🤖 {answer}"

        if sources:
            response += "\n\n📚 Sources:\n"
            for source in sources:
                short = source[:40] + "..." if len(source) > 40 else source
                response += f"  • {short}\n"

        if len(response) > 4000:
            response = response[:4000] + "\n\n... (response truncated)"

        # Send the text response
        await processing_msg.edit_text(response)

        # Now generate voice response
        # Detect if the answer is mostly English or Arabic (TTS supported languages)
        def is_tts_supported(text):
            """Check if text is mostly English or Arabic characters."""
            english_arabic = 0
            other = 0
            for char in text:
                if char.isascii() or '\u0600' <= char <= '\u06FF':
                    english_arabic += 1
                elif char.isalpha():
                    other += 1
            total = english_arabic + other
            if total == 0:
                return False
            return (english_arabic / total) > 0.5

        # Only generate voice if the answer is in a supported language
        voice_output_path = os.path.join(DATA_DIR, f"{user_id}_voice_out.wav")
        tts_result = {"success": False}

        if is_tts_supported(answer):
            await send_typing(update)
            tts_result = text_to_speech(answer, voice_output_path)
        if tts_result["success"]:
            # Send voice message
            with open(voice_output_path, "rb") as audio:
                await update.message.reply_voice(
                    voice=audio,
                    reply_markup=InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("❓ Ask Another", callback_data="ask"),
                            InlineKeyboardButton("🔙 Menu", callback_data="menu"),
                        ]
                    ])
                )
        else:
            await update.message.reply_text(
                "🔇 Voice reply is only available in English and Arabic. "
                "Your text answer is above!",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("❓ Ask Another", callback_data="ask"),
                        InlineKeyboardButton("🔙 Menu", callback_data="menu"),
                    ]
                ])
            )

    except Exception as e:
        await processing_msg.edit_text(
            "❌ Something went wrong processing your voice message.\n\n"
            "Please try again or type your question instead.",
            reply_markup=get_main_menu()
        )

    finally:
        # Clean up both audio files
        if voice_input_path and os.path.exists(voice_input_path):
            os.remove(voice_input_path)
        if voice_output_path and os.path.exists(voice_output_path):
            os.remove(voice_output_path)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle when a user sends a photo."""
    user_id = update.effective_user.id

    # Get the largest version of the photo
    photo = update.message.photo[-1]

    processing_msg = await update.message.reply_text(
        "🔍 Analyzing your image... This may take a moment."
    )

    file_path = None

    try:
        await send_typing(update)

        # Download the photo
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, f"{user_id}_photo.jpg")

        tg_file = await photo.get_file()
        await tg_file.download_to_drive(file_path)

        await send_typing(update)

        # Analyze the image
        result = process_image(
            user_id,
            file_path,
            source_name=f"image_{photo.file_unique_id}",
            add_to_kb=True
        )

        if result["success"]:
            analysis = result["analysis"]
            cat_key = result.get("category", "general")
            cat_label = CATEGORIES.get(cat_key, {}).get("label", "📝 General")

            if len(analysis) > 3600:
                analysis = analysis[:3600] + "\n\n... (truncated)"

            response = f"🖼️ Image Analysis:\n\n{analysis}"

            if result["chunks_added"] > 0:
                response += f"\n\n✅ Added {result['chunks_added']} chunks"
                response += f"\n🏷️ Category: {cat_label}"

            await processing_msg.edit_text(
                response,
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("✅ Category OK", callback_data="ask"),
                        InlineKeyboardButton("🏷️ Change", callback_data=f"retag_image_{photo.file_unique_id}"),
                    ],
                    [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
                ])
            )
        else:
            await processing_msg.edit_text(
                f"❌ Could not analyze the image.\n\n"
                f"Reason: {result['error']}\n\n"
                f"Please try a clearer image.",
                reply_markup=get_main_menu()
            )

    except Exception as e:
        await processing_msg.edit_text(
            "❌ Something went wrong analyzing the image.\n\n"
            "Please try again.",
            reply_markup=get_main_menu()
        )

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)