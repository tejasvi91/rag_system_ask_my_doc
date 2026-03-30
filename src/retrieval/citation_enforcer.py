import json
import re
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config import OPENAI_API_KEY, CHAT_MODEL
from src.config import load_prompts


def enforce_citations(answer: str, context: str) -> dict:
    """
    Audit the generated answer against the source chunks.
    Returns a dict with supported (bool) and unsupported_claims (list).
    """
    prompts = load_prompts()
    system_prompt = prompts["citation_check"]["system"]

    user_prompt = prompts["citation_check"]["user"]
    user_prompt = user_prompt.replace("{context}", context)
    user_prompt = user_prompt.replace("{answer}", answer)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Fix double braces that come from YAML escaping
    raw = raw.replace("{{", "{").replace("}}", "}")

    try:
        result = json.loads(raw)
        if "supported" not in result:
            raise ValueError("Missing 'supported' key")
        if "unsupported_claims" not in result:
            result["unsupported_claims"] = []
        return result
    except (json.JSONDecodeError, ValueError):
        print(f"  [Citation enforcer] Could not parse response: {raw[:200]}")
        return {
            "supported": False,
            "unsupported_claims": [f"Parse error. Raw response: {raw[:100]}"]
        }