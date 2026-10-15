"""Tools Aria's brain can call: search the company documents, and get live weather."""
import sys
from pathlib import Path

import requests
from langchain_core.tools import tool

RAG_DIR = Path(__file__).parent.parent / "rag"
LEADS_DIR = Path(__file__).parent.parent / "leads"
sys.path.insert(0, str(RAG_DIR))
sys.path.insert(0, str(LEADS_DIR))

from step3_search import search
from db import top as top_leads


@tool
def search_documents(query: str) -> dict:
    """Searches the company documents (policies, HR rules, notes) and returns the most relevant passages.

    Args:
        query: A short search phrase in English describing what to look for,
            e.g. "work from home policy" or "number of paid leaves".
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
            "page": m["page"],
            "text": m["text"],
            "distance": round(m["distance"], 3),
        })
    return {"status": "success", "results": results}


@tool
def get_weather(city: str) -> dict:
    """Gets the real current weather for a city.

    Args:
        city: Name of the city, e.g. "Mohali" or "London".
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


@tool
def get_leads(kind: str = "", service: str = "", min_score: int = 6, days: int = 3) -> dict:
    """Gets the best sales leads found by the daily Lead Finder.

    Args:
        kind: "project" (work to bid on), "company" (newly funded startups to approach),
            "learning" (new open-source projects to learn from), or "" for all kinds.
        service: "web development", "mobile apps", "ai agents & chatbots", "blockchain", or "" for all.
        min_score: only leads scored at least this (1-10). Use 6 unless the user asks otherwise.
        days: how many recent days to include. Use 3 unless the user asks otherwise.
    """
    try:
        leads = top_leads(kind=kind or None, service=service or None, min_score=min_score, days=days, limit=5)
    except Exception as e:
        return {"status": "error", "error_message": f"Leads are unavailable right now: {e}"}
    if not leads:
        return {"status": "success", "leads": [], "note": "No matching leads. The Lead Finder may not have run today."}
    return {"status": "success", "leads": [
        {"id": l["id"], "kind": l["kind"], "score": l["score"], "title": l["title"], "budget": l["budget"],
         "services": l["services"], "why": l["why"], "url": l["url"], "posted_at": l["posted_at"]}
        for l in leads
    ]}


TOOLS = [search_documents, get_weather, get_leads]


if __name__ == "__main__":
    # Run each tool directly (no AI involved) to check it works.
    print("Tool names:", [t.name for t in TOOLS])

    found = search_documents.invoke({"query": "work from home policy"})
    print("\nsearch_documents:", found["status"])
    for r in found.get("results", []):
        print(f"  {r['source']} page {r['page']} (distance {r['distance']}): {r['text'][:80]}...")

    print("\nget_weather:", get_weather.invoke({"city": "Mohali"}))
