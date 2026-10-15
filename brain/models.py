"""Aria's brain model: Gemini first, then other Gemini models, then Claude if all are busy."""
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / "my_agent" / ".env")

# Tried in this order. The next one is used only if the previous one fails (busy, quota, error).
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
CLAUDE_MODEL = "claude-opus-5-5"


def get_model(tools=None):
    """Return a chat model that falls back to the next model on errors.

    tools: optional list of Python functions the model may call.
    """
    models = [ChatGoogleGenerativeAI(model=name, max_retries=1, timeout=60) for name in GEMINI_MODELS]
    models.append(ChatAnthropic(model=CLAUDE_MODEL, max_retries=1, timeout=60))
    if tools:
        models = [model.bind_tools(tools) for model in models]
    first, *backups = models
    return first.with_fallbacks(backups)


if __name__ == "__main__":
    model = get_model()
    reply = model.invoke("Say hello in one short sentence, in English and Hindi.")
    print("Model used:", reply.response_metadata.get("model_name") or reply.response_metadata.get("model"))
    print("Reply:", reply.text)
