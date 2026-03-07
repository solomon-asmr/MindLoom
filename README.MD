# 🤖 Personal Assistant Bot

A Telegram chatbot that lets anyone upload their materials and ask questions about them using AI. Powered by RAG (Retrieval Augmented Generation), it finds relevant information from your documents and generates accurate answers.

Each user gets their own **private knowledge base** — your data is never shared with other users.

---

## ✨ Features

**📄 Multi-Format Document Support**
Upload PDFs, Word documents (.docx), text files (.txt), and CSV files. The bot extracts text, splits it into searchable chunks, and stores it in your personal knowledge base.

**🌐 Website Scraping**
Send a URL and the bot scans the page for all links. Choose which pages to include — the bot scrapes them and adds the content to your knowledge base.

**❓ Smart Q&A**
Ask questions in natural language. The bot searches your materials using semantic similarity (not just keyword matching), finds the most relevant passages, and generates an accurate answer with source citations.

**🎤 Voice Input & Output**
Ask questions by sending voice messages. The bot transcribes your speech using Whisper, finds the answer, and replies with both text and a spoken audio response using Orpheus TTS.

**🌍 Multilingual Support**
Ask questions and receive answers in any language — English, Hebrew, Amharic, Arabic, and more. Voice replies are available in English and Arabic.

**🔒 Per-User Privacy**
Every user gets their own isolated knowledge base. Your documents, websites, and conversations are completely private.

**💬 Conversation Memory**
The bot remembers your recent messages, so you can ask follow-up questions like "Tell me more about that" or "Can you give an example?"

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Messaging | Telegram Bot API | User interface |
| LLM | Groq (LLaMA 3.3 70B) | Answer generation |
| Speech-to-Text | Groq Whisper | Voice transcription |
| Text-to-Speech | Groq Orpheus | Voice replies |
| Embeddings | ChromaDB (all-MiniLM-L6-v2) | Semantic search |
| Vector Database | ChromaDB | Chunk storage & retrieval |
| PDF Processing | PyMuPDF | PDF text extraction |
| DOCX Processing | python-docx | Word document reading |
| Web Scraping | BeautifulSoup4 | Website content extraction |

---

## 📁 Project Structure

```
MindLoom/
├── config.py              # API keys and settings
├── document_loader.py     # Read PDF, TXT, DOCX, CSV files
├── web_scraper.py         # Scan URLs, find links, scrape pages
├── chunker.py             # Split text into chunks with overlap
├── rag_engine.py          # Core RAG pipeline (retrieve, augment, generate)
├── user_manager.py        # Per-user ChromaDB collections
├── bot.py                 # Telegram bot UI (buttons, menus, routing)
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
├── .env                   # Secret keys (not in repo)
├── .gitignore             # Files excluded from Git
├── data/                  # Temporary file storage
└── chroma_db/             # Persistent vector database
```

---

## 🔧 How It Works

The bot uses the **RAG (Retrieval Augmented Generation)** pattern:

```
1. PREPARE:  Documents → Split into chunks → Embed → Store in ChromaDB
2. RETRIEVE: User question → Embed → Find similar chunks (cosine similarity)
3. AUGMENT:  Combine question + relevant chunks into a prompt
4. GENERATE: Send prompt to LLM → Get accurate answer
```

This allows the LLM to answer questions based on **your specific documents** rather than its general training data. The semantic search finds relevant information even when the exact words don't match.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq](https://groq.com) API key (free)
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/MindLoom.git
   cd MindLoom
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   source venv/Scripts/activate    # Windows (Git Bash)
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create the `.env` file**
   ```
   GROQ_API_KEY=gsk_your_key_here
   TELEGRAM_BOT_TOKEN=your_token_here
   ```

5. **Run the bot**
   ```bash
   python main.py
   ```

6. **Open Telegram** and send `/start` to your bot

---

## 💬 Usage

### Adding Documents
Click **📄 Add Document** and send a file (PDF, TXT, DOCX, or CSV). The bot processes it and adds the content to your knowledge base.

### Adding Websites
Click **🌐 Add Website** and send a URL. The bot scans the page, shows all found links, and lets you choose which ones to include.

### Asking Questions
Click **❓ Ask Question** and type your question — or simply send a voice message. The bot searches your materials and responds with an answer and source citations.

### Managing Your Data
Click **📚 My Knowledge Base** to see all your sources, how many chunks each contains, and options to delete individual sources or clear everything.

---

## ⚙️ Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model for generating answers |
| `LLM_TEMPERATURE` | `0` | Response creativity (0 = focused, 2 = creative) |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K` | `3` | Number of chunks retrieved per question |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload file size |

---

## 📸 Screenshots

**Main Menu**
```
👋 Welcome! I'm your personal study assistant.
Upload documents or websites, then ask me anything!

[📄 Add Document]  [🌐 Add Website]
[❓ Ask Question]  [📚 My Knowledge Base]
[❔ Help]
```

**Asking a Question**
```
🤖 Machine learning is a subset of artificial intelligence
that allows computers to learn from data without being
explicitly programmed. It includes supervised learning,
unsupervised learning, and reinforcement learning.

📚 Sources:
  • ml_notes.pdf
  • course_lecture_3.txt
```

**Voice Interaction**
```
🎤 You asked: "What is deep learning?"

🤖 Deep learning is a type of machine learning that uses
neural networks with many layers to process complex data...

📚 Sources:
  • ai_textbook.pdf

🔊 [Voice message with the answer]
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---


## 🙏 Acknowledgments

- [Groq](https://groq.com) — Fast LLM inference, Whisper STT, and Orpheus TTS
- [ChromaDB](https://www.trychroma.com) — Open-source vector database
- [python-telegram-bot](https://python-telegram-bot.org) — Telegram Bot API wrapper
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF text extraction
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) — Web scraping
