"""Collectors: read new leads from public APIs and RSS feeds (read-only, one polite request each)."""
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

from config import FUNDING_FEEDS, FUNDING_WORDS, MAX_AGE_DAYS, REDDIT_SUBS, SERVICES, SOURCES, USD_RATES, USER_AGENT

HN_DAYS = 31      # the freelancer thread is monthly
GITHUB_DAYS = 14  # repos need a little time to collect stars


def _get(url: str, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    """One polite request: our app name, 15 s timeout, no retries."""
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT, **(headers or {})}, timeout=15)
    response.raise_for_status()
    return response


def _clean(text: str) -> str:
    """Remove HTML tags and extra spaces."""
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _is_recent(when: datetime, days: int = MAX_AGE_DAYS) -> bool:
    return when >= datetime.now(timezone.utc) - timedelta(days=days)


def services_in(text: str) -> list[str]:
    """Which of our services a text mentions (free keyword check, no AI)."""
    text = text.lower()
    return [name for name, words in SERVICES.items() if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)]


def _lead(source, kind, title, text, url, when, budget=None) -> dict:
    return {
        "source": source,
        "kind": kind,
        "title": _clean(title),
        "text": _clean(text)[:1500],
        "url": url,
        "posted_at": when.date().isoformat(),
        "budget": budget,
    }


def _rss_entries(url: str) -> list[dict]:
    """Read an RSS or Atom feed into simple dicts: title, link, text, when."""
    root = ET.fromstring(_get(url).content)
    entries = []
    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        fields = {child.tag.split("}")[-1]: child for child in item}
        link = fields.get("link")
        link = (link.get("href") or link.text) if link is not None else ""
        date_el = fields.get("pubDate") or fields.get("updated") or fields.get("published")
        date_text = date_el.text if date_el is not None else ""
        try:
            when = parsedate_to_datetime(date_text) if "," in date_text else datetime.fromisoformat(date_text)
        except (TypeError, ValueError):
            when = datetime.now(timezone.utc)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        body = fields.get("description") or fields.get("content") or fields.get("summary")
        entries.append({
            "title": fields["title"].text or "" if "title" in fields else "",
            "link": link,
            "text": body.text or "" if body is not None else "",
            "when": when,
        })
    return entries


# ---------------- projects to bid on ----------------

def freelancer() -> list[dict]:
    leads = []
    for service in SERVICES:
        data = _get(
            "https://www.freelancer.com/api/projects/0.1/projects/active/",
            params={"query": service, "limit": 20, "full_description": "true", "sort_field": "time_updated"},
        ).json()
        for p in data["result"]["projects"]:
            when = datetime.fromtimestamp(p.get("time_submitted") or p.get("time_updated") or 0, timezone.utc)
            if not _is_recent(when):
                continue
            budget = p.get("budget") or {}
            low, high = budget.get("minimum"), budget.get("maximum")
            currency = (p.get("currency") or {}).get("code", "")
            amount = f"{low:g}-{high:g}" if low and high else f"{low:g}+" if low else "?"
            rate = USD_RATES.get(currency)
            if rate and low and currency != "USD":
                usd = f"~${low * rate:,.0f}-{high * rate:,.0f}" if high else f"~${low * rate:,.0f}+"
                amount += f" {currency} ({usd})"
                currency = ""
            leads.append(_lead(
                "freelancer", "project", p["title"], p.get("description") or p.get("preview_description", ""),
                f"https://www.freelancer.com/projects/{p['seo_url']}", when,
                budget=f"{amount} {currency}".strip(),
            ))
    return leads


def reddit() -> list[dict]:
    leads = []
    for e in _rss_entries(f"https://www.reddit.com/r/{'+'.join(REDDIT_SUBS)}/new/.rss"):
        if re.match(r"\s*\[hiring\]", e["title"], re.I) and _is_recent(e["when"]):
            budget = re.search(r"\$\s?\d[\d,.]*\s*(?:-\s*\$?\s?\d[\d,.]*)?\s*(?:/\s?(?:hr|hour|h))?", e["title"])
            leads.append(_lead("reddit", "project", e["title"], e["text"], e["link"], e["when"],
                               budget=budget.group().strip() if budget else None))
    return leads


def hackernews() -> list[dict]:
    threads = _get(
        "https://hn.algolia.com/api/v1/search_by_date",
        params={"query": "Freelancer? Seeking freelancer?", "tags": "story", "hitsPerPage": 10},
    ).json()["hits"]
    threads = [t for t in threads if t["title"].startswith("Ask HN: Freelancer? Seeking freelancer?")]
    if not threads:
        return []
    comments = _get(
        "https://hn.algolia.com/api/v1/search_by_date",
        params={"tags": f"comment,story_{threads[0]['objectID']}", "hitsPerPage": 200},
    ).json()["hits"]
    leads = []
    for c in comments:
        text = _clean(c.get("comment_text") or "")
        when = datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))
        if text.upper().startswith("SEEKING FREELANCER") and _is_recent(when, HN_DAYS):
            leads.append(_lead("hackernews", "project", text[:100], text,
                               f"https://news.ycombinator.com/item?id={c['objectID']}", when))
    return leads


# ---------------- companies to approach ----------------

def funding_news() -> list[dict]:
    leads = []
    for feed in FUNDING_FEEDS:
        for e in _rss_entries(feed):
            title = e["title"].lower()
            if any(word in title for word in FUNDING_WORDS) and _is_recent(e["when"]):
                leads.append(_lead("funding_news", "company", e["title"], e["text"], e["link"], e["when"]))
    return leads


# ---------------- projects to learn from ----------------

def github() -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=GITHUB_DAYS)).date().isoformat()
    data = _get(
        "https://api.github.com/search/repositories",
        params={"q": f"topic:ai-agents created:>{since}", "sort": "stars", "order": "desc", "per_page": 10},
        headers={"Accept": "application/vnd.github+json"},
    ).json()
    leads = []
    for r in data["items"]:
        when = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        text = f"{r.get('description') or ''} | stars: {r['stargazers_count']} | language: {r.get('language')}"
        leads.append(_lead("github", "learning", r["full_name"], text, r["html_url"], when))
    return leads


COLLECTORS = {"freelancer": freelancer, "reddit": reddit, "hackernews": hackernews,
              "funding_news": funding_news, "github": github}


def collect_all() -> list[dict]:
    """Run every enabled source. A failing source is skipped, never retried."""
    leads = []
    for name, collect in COLLECTORS.items():
        if not SOURCES.get(name):
            continue
        try:
            found = collect()
        except Exception as e:
            print(f"  ⚠️  {name}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        kept = 0
        for lead in found:
            lead["services"] = services_in(f"{lead['title']} {lead['text']}")
            # projects must match one of our services; companies and learning are judged by the AI later
            if lead["kind"] != "project" or lead["services"]:
                leads.append(lead)
                kept += 1
        print(f"  {name}: {len(found)} found, {kept} kept")
    return leads


if __name__ == "__main__":
    print("Collecting (read-only, no AI, nothing saved)...")
    leads = collect_all()
    print(f"\nTotal kept: {len(leads)}")
    for kind in ("project", "company", "learning"):
        sample = [l for l in leads if l["kind"] == kind][:3]
        print(f"\n--- {kind} ---")
        for l in sample:
            print(f"[{l['source']}] {l['title'][:90]}")
            print(f"    {l['posted_at']} | budget: {l['budget']} | services: {l['services']} | {l['url']}")
