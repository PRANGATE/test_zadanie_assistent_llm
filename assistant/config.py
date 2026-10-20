from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "assistant.sqlite3"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STT_MODEL = "large-v3-turbo" #STT_MODEL = "small", medium
SAMPLE_RATE = 16_000
MAX_RECORD_SECONDS = 90
