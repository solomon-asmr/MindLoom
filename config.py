import os
os.environ['CHROMA_TELEMETRY_MOUNT'] = 'False'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

LLM_MODEL = os.getenv("openai/gpt-oss-20b")
LLM_TEMPERATURE = 0

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

CHROMA_DB_PATH = "./chroma_db"

DATA_DIR = "./data"
MAX_FILE_SIZE = 20