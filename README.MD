# 🧠 MindLoom

A powerful AI-powered personal assistant on Telegram that lets you build your own private knowledge base from any source — documents, websites, photos, or voice — and get instant, accurate answers powered by RAG.

Each user gets their own **private, organized knowledge base** with smart categorization and hybrid search.

---

## ✨ Features

### 📄 Multi-Format Document Support
Upload PDFs, Word documents (.docx), text files (.txt), and CSV files. The bot extracts all text — and if your document contains images, diagrams, or charts, it analyzes those too using AI vision.

### 🖼️ Image Analysis
Send a photo of lecture notes, a whiteboard, a screenshot, or any image. The bot deeply scans it — extracting text, equations, diagrams, tables, and context — then adds everything to your knowledge base.

### 🌐 Website Scraping
Send a URL and the bot scans the page for all links. Choose which pages to include — the bot scrapes them and adds the content to your knowledge base. Internal and external links are clearly marked.

### ❓ Smart Q&A with Hybrid Search
Ask questions in natural language. The bot uses **hybrid search** — combining semantic vector search (understands meaning) with BM25 keyword search (finds exact terms) — for the most accurate results. Answers include source citations.

### ⚡ Streaming Responses
Answers appear in real-time as the AI generates them, token by token. No more staring at a blank screen — you start reading within half a second.

### 🎤 Voice Input & Output
Ask questions by sending voice messages. The bot transcribes your speech using Whisper, finds the answer, and replies with both text and a spoken audio response using Orpheus TTS.

### 🌍 Multilingual Support
Ask questions and receive answers in any language — English, Hebrew, Amharic, Arabic, Spanish, and more. The bot automatically responds in the same language you ask in. Voice replies are available in English and Arabic.

### 🏷️ Smart Categorization
Documents are automatically categorized (Work, Finance, Health, Education, Technical, Personal, Food, Travel, Creative, General). When you ask a question, the bot automatically routes it to the right category — searching only relevant documents for faster, more accurate answers.

### 🔒 Per-User Privacy
Every user gets their own isolated knowledge base. Your documents, websites, and conversations are completely private and never shared with other users.

### 💬 Conversation Memory
The bot remembers your recent messages, so you can ask follow-up questions like "Tell me more about that" or "Can you give an example?" without repeating context.

---

## 🛠️ Tech Stack

| Component       | Technology                  | Purpose                                 |
|-----------------|-----------------------------|-----------------------------------------|
| Messaging       | Telegram Bot API            | User interface with inline buttons      |
| LLM             | Groq (LLaMA 3.3 70B)        | Answer generation                       |
| Speech-to-Text  | Groq Whisper                | Voice message transcription             |
| Text-to-Speech  | Groq Orpheus                | Voice reply generation                  |
| Image Analysis  | Groq (LLaMA 4 Scout Vision) | Photo and image understanding           |
| Vector Search   | ChromaDB (all-MiniLM-L6-v2) | Semantic similarity search              |
| Keyword Search  | BM25 (rank-bm25)            | Exact term matching                     |
| Search Strategy | Reciprocal Rank Fusion      | Combines vector + keyword results       |
| PDF Processing  | PyMuPDF                     | Text and image extraction from PDFs     |
| DOCX Processing | python-docx                 | Word document text and image extraction |
| Web Scraping    | BeautifulSoup4              | Website content extraction              |

---

## 📁 Project Structure

```
mindloom/
├── config.py              # API keys, settings, and categories
├── document_loader.py     # Read PDF, TXT, DOCX, CSV + extract images
├── web_scraper.py         # Scan URLs, find links, scrape pages
├── chunker.py             # Split text into chunks with overlap
├── rag_engine.py          # Core RAG pipeline + streaming + vision + TTS
├── user_manager.py        # Per-user collections + hybrid search
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

MindLoom uses an advanced **RAG (Retrieval Augmented Generation)** pipeline:

```
UPLOAD FLOW:
  Document/URL/Photo
    → Extract text (+ analyze embedded images with AI vision)
    → Split into overlapping chunks
    → Auto-detect category with LLM
    → Store chunks + embeddings + metadata in ChromaDB

QUERY FLOW:
  User question (text or voice)
    → Transcribe voice (if applicable)
    → LLM detects relevant categories (query routing)
    → Hybrid search: vector similarity + BM25 keywords
    → Reciprocal Rank Fusion combines results
    → Stream answer token-by-token to Telegram
    → Generate voice reply (if voice input)
```

### Why Hybrid Search?

| Search Type | Finds                                              | Misses                                  |
|-------------|----------------------------------------------------|-----------------------------------------|
| Vector only | "affordable options" when you search "cheap deals" | Exact names, codes, IDs like "API-v2.3" |
| BM25 only   | Exact term "/api/v2/auth"                          | Synonyms and paraphrases                |
| **Hybrid**  | **Both meaning matches AND exact terms**           | **Very little**                         |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq](https://groq.com) API key (free)
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mindloom.git
   cd mindloom
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

5. **Accept Groq model terms**
   
   Visit these links and accept terms (required for voice features):
   - [Orpheus English TTS](https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english)

6. **Run the bot**
   ```bash
   python main.py
   ```

7. **Open Telegram** and send `/start` to your bot

---

## 💬 Usage

### Main Menu
```
[📄 Add Document]  [🌐 Add Website]
[❓ Ask Question]  [📚 My Knowledge Base]
[❔ Help]
```

### Adding Documents
Send any supported file. The bot extracts text, analyzes embedded images, auto-detects the category, and lets you confirm or change it.
```
✅ Successfully processed company_handbook.pdf!
📊 45 chunks added
🏷️ Category: 💼 Work & Business
[✅ Yes]  [🏷️ Change]
```

### Adding Websites
Send a URL. The bot scans for links, lets you choose which pages to include, then scrapes and categorizes them.
```
Found 6 links:
1. 🔵 Introduction to Python
2. 🔵 Data Structures
3. 🔵 Algorithms
4. 🌐 External Resource
Reply with numbers (e.g., 1, 2, 3) or 'all'
```

### Analyzing Images
Send any photo — lecture notes, whiteboard, screenshot, diagram. The bot extracts every detail and adds it to your knowledge base.
```
🖼️ Image Analysis:
The image shows handwritten notes about neural networks...
✅ Added 3 chunks
🏷️ Category: 📚 Education & Learning
```

### Asking Questions
Type a question or send a voice message. Answers stream in real-time with source citations.
```
🤖 Machine learning is a subset of artificial
   intelligence that allows computers to learn
   from data without being explicitly programmed...▌

📚 Sources:
  • ml_textbook.pdf
🔍 Searched: 📚 Education & Learning
```

### Voice Interaction
Send a voice message in any language. Get a text answer plus a spoken audio reply.
```
🎤 You asked: "What is deep learning?"
🤖 Deep learning is a type of machine learning...
📚 Sources: ai_notes.pdf
🔊 [Voice message with the answer]
```

---

## 🏷️ Categories

Documents are automatically sorted into these categories:

| Category           | Label                    | Examples                                  |
|--------------------|--------------------------|-------------------------------------------|
| work               | 💼 Work & Business       | Company policies, meeting notes, reports  |
| finance            | 💰 Finance & Legal       | Invoices, tax documents, contracts        |
| health             | 🏥 Health & Wellness     | Medical records, fitness plans, nutrition |
| education          | 📚 Education & Learning  | Textbooks, lectures, course notes         |
| technical          | 🛠️ Technical & IT        | Code docs, API references, tutorials      |
| personal           | 🏠 Personal & Lifestyle  | Shopping lists, home guides, family docs  |
| food               | 🍳 Food & Recipes        | Recipes, meal plans, restaurant info      |
| travel             | ✈️ Travel & Places       | Bookings, itineraries, travel guides      |
| creative           | 🎨 Creative & Hobbies    | Art references, music notes, craft guides |
| general            | 📝 General               | Everything else                           |

You can always change the auto-detected category after uploading.

---

## ⚙️ Configuration

All settings in `config.py`:

| Setting            | Default                   | Description                             |
|--------------------|---------------------------|-----------------------------------------|
| `LLM_MODEL`        | `llama-3.3-70b-versatile` | Groq model for generating answers       |
| `LLM_TEMPERATURE`  | `0`                       | Response creativity (0 = focused)       |
| `CHUNK_SIZE`       | `500`                     | Characters per text chunk               |
| `CHUNK_OVERLAP`    | `50`                      | Overlap between consecutive chunks      |
| `TOP_K`            | `3`                       | Number of chunks retrieved per question |
| `MAX_FILE_SIZE_MB` | `20`                      | Maximum upload file size                |

---


## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Groq](https://groq.com) — Fast LLM inference, Whisper STT, Orpheus TTS, and LLaMA 4 Vision
- [ChromaDB](https://www.trychroma.com) — Open-source vector database
- [python-telegram-bot](https://python-telegram-bot.org) — Telegram Bot API wrapper
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 keyword search
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF text and image extraction
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) — Web scraping