import os
import time
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

_langfuse = None


def get_langfuse() -> Langfuse:
    """Single Langfuse instance reused across the app."""
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        )
    return _langfuse


def flush():
    """Flush all pending traces — call this at the end of each request."""
    get_langfuse().flush()