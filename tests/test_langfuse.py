import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(
    name="test-connection",
    metadata={"test": True}
)

trace.span(
    name="test-span",
    input={"message": "hello from rag system"},
    output={"status": "connected"}
)

langfuse.flush()
print("Langfuse connection successful!")
print(f"Trace ID: {trace.id}")
print(f"View at: http://localhost:3000")