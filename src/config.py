import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_PATH = ROOT_DIR / "prompts" / "prompts.yaml"
CHROMA_DIR = str(ROOT_DIR / "chroma_db")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Chunking
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5


def load_prompts() -> dict:
    with open(PROMPTS_PATH, "r") as f:
        return yaml.safe_load(f)