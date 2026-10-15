"""Lead Finder settings: what we look for, and where."""

# Your company's services: leads must match at least one of these.
SERVICES = {
    "web development": ["website", "web app", "web development", "frontend", "backend", "full stack", "react", "next.js", "wordpress", "shopify", "landing page"],
    "mobile apps": ["mobile app", "android", "ios", "flutter", "react native", "app development"],
    "ai agents & chatbots": ["ai agent", "chatbot", "llm", "gpt", "rag", "langchain", "automation", "ai assistant", "voice agent", "machine learning"],
    "blockchain": ["blockchain", "web3", "smart contract", "crypto", "nft", "token", "defi", "solidity"],
}

MIN_SCORE = 6          # only show leads the AI scores 6/10 or higher
MAX_AGE_DAYS = 3       # ignore posts older than this
USER_AGENT = "AriaLeadFinder/0.1 (read-only lead research)"

# Turn sources on/off.
SOURCES = {
    "freelancer": True,     # projects to bid on
    "reddit": True,         # [HIRING] posts on r/forhire, r/hiring
    "hackernews": True,     # monthly "Seeking freelancer" thread
    "funding_news": True,   # newly funded startups (YourStory, TechCrunch)
    "github": True,         # new AI-agent / web projects to learn from
}

# Approximate value of 1 unit in US dollars, used only to compare budgets fairly.
# Update now and then; exact rates are not needed for scoring.
USD_RATES = {
    "USD": 1, "INR": 0.012, "EUR": 1.08, "GBP": 1.27, "AUD": 0.66, "CAD": 0.73,
    "SGD": 0.74, "AED": 0.27, "NZD": 0.60, "HKD": 0.13, "PHP": 0.017, "PKR": 0.0036,
}

REDDIT_SUBS = ["forhire", "hiring"]
FUNDING_FEEDS = ["https://yourstory.com/feed", "https://techcrunch.com/category/startups/feed/"]
FUNDING_WORDS = ["raises", "raised", "funding", "seed round", "series a", "series b", "investment", "valuation"]
