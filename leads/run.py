"""Run the Lead Finder once: collect -> save new -> AI score -> print today's best leads."""
from config import MIN_SCORE
from db import save_new, top
from score import score_unscored
from sources import collect_all

GROUPS = [
    ("project", "💼 Projects to bid on"),
    ("company", "🏢 Companies to approach"),
    ("learning", "📚 Projects to learn from"),
]


def run() -> None:
    print("1. Collecting from sources (read-only)...")
    found = collect_all()

    new = save_new(found)
    print(f"\n2. Saved {len(new)} new leads ({len(found) - len(new)} already known)")

    print("\n3. AI scoring...")
    scored = score_unscored()
    print(f"   scored {scored} leads")

    print(f"\n4. Best leads (score {MIN_SCORE}+, last 3 days)")
    for kind, heading in GROUPS:
        leads = top(kind=kind, min_score=MIN_SCORE, days=3, limit=5)
        print(f"\n{heading}")
        if not leads:
            print("   (none today)")
        for l in leads:
            services = ", ".join(l["services"]) or "-"
            print(f"   [{l['score']}/10] #{l['id']} {l['title'][:80]}")
            print(f"          budget: {l['budget'] or '-'} | services: {services} | {l['source']}")
            print(f"          why: {l['why']}")
            print(f"          {l['url']}")


if __name__ == "__main__":
    run()
