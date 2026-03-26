from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

if key and key.startswith("sk-"):
    print(f"Key loaded successfully: {key[:8]}...{key[-4:]}")
else:
    print("ERROR: Key not found or looks wrong. Check your .env file.")