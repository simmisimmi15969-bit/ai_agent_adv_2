import sys
from pathlib import Path

import requests
# from pypdf import PdfReader
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models import FallbackModel
from google.adk.models.anthropic_llm import AnthropicLlm
from google.genai import types

RAG_DIR = Path(__file__).parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))

from step3_search import search

GEMINI_RETRY = types.HttpRetryOptions(initial_delay=1, attempts=2)

# MAX_CHARS_PER_FILE = 20000

def search_documents(query: str) -> dict:
    """Searches the user's documents (company policies, HR rules, resumes, notes)
    and returns the most relevant passages.

    Args:
        query: A short search phrase describing what to look for,
            e.g. "work from home policy" or "number of paid leaves".

    Returns:
        A dict with status and a list of results, each with 'source' (file name),
        'text' (the passage) and 'distance' (lower means more relevant).
    """
    try:
        matches = search(query)
    except Exception as e:
        return {"status": "error", "error_message": f"Document search is unavailable right now: {e}"}
    if not matches:
        return {"status": "error", "error_message": "No documents have been indexed yet."}

    results = []
    for m in matches:
        results.append({
            "source": m["source"],
            "text": m["text"],
            "distance": round(m["distance"], 3),
            "page": m["page"],

        })
    return {"status": "success", "results": results}

# def _extract_text(path: Path) -> str:
#     if path.suffix.lower() == ".pdf":
#         reader = PdfReader(path)
#         return "\n".join(page.extract_text() or "" for page in reader.pages)
#     return path.read_text(encoding="utf-8", errors="ignore")


# def read_documents() -> dict:
#     """Reads all of the user's documents (PDF, TXT, MD files) and returns their text.

#     Use this for any question that might be answered by the user's
#     own files, such as company policies, resumes, notes or reports.

#     Returns:
#         A dict with status and a list of documents, each with
#         'file' (file name) and 'content' (the text).
#     """
#     files = [p for p in DOCS_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}]
#     if not files:
#         return {"status": "error", "error_message": "No documents found in the docs folder."}

    # documents = []
    # for path in files:
    #     text = _extract_text(path)[:MAX_CHARS_PER_FILE]
    #     documents.append({"file": path.name, "content": text})
    # return {"status": "success", "documents": documents}


def get_weather(city: str) -> dict:
    """Gets the real current weather for a city.

    Args:
        city: Name of the city, e.g. "Mohali" or "London".

    Returns:
        A dict with status, and temperature, humidity and wind speed
        if successful, or an error_message if not.
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()

        if not geo.get("results"):
            return {"status": "error", "error_message": f"City '{city}' not found."}

        place = geo["results"][0]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            },
            timeout=10,
        ).json()

        current = weather["current"]
        return {
            "status": "success",
            "city": place["name"],
            "country": place.get("country"),
            "temperature_c": current["temperature_2m"],
            "humidity_percent": current["relative_humidity_2m"],
            "wind_kmh": current["wind_speed_10m"],
        }
    except requests.RequestException as e:
        return {"status": "error", "error_message": f"Weather service failed: {e}"}


root_agent = Agent(
    name="my_agent",
    model=FallbackModel(
        models=[
            Gemini(model="gemini-3.5-flash", retry_options=GEMINI_RETRY),
            Gemini(model="gemini-3.6-flash", retry_options=GEMINI_RETRY),
            Gemini(model="gemini-3.5-flash-lite", retry_options=GEMINI_RETRY),
            AnthropicLlm(model="claude-opus-5-5"),
        ],
    ),
    description="Answers questions from the user's documents, tools, or general knowledge.",
    instruction=(
        "You are a helpful assistant. Decide which source to use:\n"
        "- Live data (like current weather): call get_weather directly. "
        "Do NOT call search_documents for these.\n"
        "- Questions about the user's own information (company, policies, "
        "resume, notes): call search_documents with a short, focused query. "
        "Always write that query in English, because the documents are in English.\n"
        "- Reply in the same language as the user's question: Hindi (in Devanagari) "
        "if they ask in Hindi, English if they ask in English. Keep citations exactly "
        "in the form '(source: file, page N)' with the file name unchanged.\n"
        "- Read the returned passages carefully. Answer ONLY from passages that "
        "actually contain the answer, and cite the file and page if given, "
        "e.g. '(source: leave_policy.pdf, page 3)'.\n"
        "- If none of the passages answer the question, start with "
        "'Not found in your documents, answering from general knowledge:' "
        "and answer from your own knowledge.\n"
        "If a tool returns an error, tell the user politely. "
        "Never invent facts about the user's documents."
    ),
    tools=[search_documents, get_weather],
)
