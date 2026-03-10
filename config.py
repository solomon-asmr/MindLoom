import os
os.environ['CHROMA_TELEMETRY_MOUNT'] = 'False'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

CHROMA_DB_PATH = "./chroma_db"

DATA_DIR = "./data"
MAX_FILE_SIZE_MB = 20

# Document categories for tagging
CATEGORIES = {
    "work": {"label": "💼 Work & Business", "keywords": ["company", "business", "meeting", "project", "client", "office", "employee", "salary", "contract", "policy", "hr", "management", "team", "report", "deadline", "presentation"]},
    "finance": {"label": "💰 Finance & Legal", "keywords": ["money", "bank", "tax", "invoice", "budget", "investment", "insurance", "legal", "law", "contract", "accounting", "payment", "loan", "mortgage", "stock"]},
    "health": {"label": "🏥 Health & Wellness", "keywords": ["health", "medical", "doctor", "medicine", "exercise", "diet", "mental", "therapy", "hospital", "symptom", "disease", "nutrition", "fitness", "sleep", "wellness"]},
    "education": {"label": "📚 Education & Learning", "keywords": ["course", "lecture", "study", "exam", "assignment", "university", "school", "lesson", "tutorial", "textbook", "degree", "research", "professor", "homework", "grade"]},
    "technical": {"label": "🛠️ Technical & IT", "keywords": ["code", "programming", "software", "api", "database", "server", "python", "javascript", "algorithm", "debug", "deploy", "github", "docker", "linux", "cloud"]},
    "personal": {"label": "🏠 Personal & Lifestyle", "keywords": ["family", "home", "shopping", "birthday", "wedding", "relationship", "kids", "pets", "garden", "cleaning", "moving", "furniture", "clothes", "gift"]},
    "food": {"label": "🍳 Food & Recipes", "keywords": ["recipe", "cooking", "ingredient", "restaurant", "meal", "breakfast", "dinner", "baking", "kitchen", "food", "dish", "calories", "vegetarian", "spice"]},
    "travel": {"label": "✈️ Travel & Places", "keywords": ["travel", "flight", "hotel", "booking", "visa", "passport", "airport", "vacation", "trip", "destination", "tourism", "luggage", "reservation", "itinerary"]},
    "creative": {"label": "🎨 Creative & Hobbies", "keywords": ["art", "music", "design", "photo", "video", "writing", "drawing", "painting", "craft", "guitar", "film", "podcast", "blog", "game", "hobby"]},
    "general": {"label": "📝 General", "keywords": []},
}