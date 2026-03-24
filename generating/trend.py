import os
import time
from urllib.parse import quote

import feedparser
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

from content import contents

keyword = "crime darkest"
keyword = quote(keyword)

USE_PYTRENDS = os.getenv("TRENDS_USE_PYTRENDS", "1") == "1"
PYTRENDS_RETRIES = int(os.getenv("TRENDS_PYTRENDS_RETRIES", "3"))
PYTRENDS_BACKOFF = float(os.getenv("TRENDS_PYTRENDS_BACKOFF", "2"))

# Worldwide trends
pytrends = TrendReq(hl="en", tz=0)

trends_list = []
if USE_PYTRENDS:
    for attempt in range(1, PYTRENDS_RETRIES + 1):
        try:
            pytrends.build_payload([keyword], geo="")
            trending = pytrends.related_queries()[keyword]["top"]
            if trending is not None:
                for _, row in trending.head(5).iterrows():
                    trends_list.append(
                        {
                            "source": "Google Trends",
                            "title": row["query"],
                            "score": int(row["value"]),
                        }
                    )
            break
        except TooManyRequestsError as exc:
            if attempt == PYTRENDS_RETRIES:
                print(f"Pytrends rate-limited (429). Skipping Google Trends. {exc}")
                break
            sleep_for = PYTRENDS_BACKOFF * attempt
            print(f"Pytrends rate-limited. Retrying in {sleep_for:.1f}s...")
            time.sleep(sleep_for)
        except Exception as exc:
            if attempt == PYTRENDS_RETRIES:
                print(f"Pytrends failed, skipping Google Trends. {exc}")
                break
            sleep_for = PYTRENDS_BACKOFF * attempt
            time.sleep(sleep_for)

# Worldwide Google News (no region lock)
rss_url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '+')}"

feed = feedparser.parse(rss_url)

news_list = []

for entry in feed.entries[:100]:
    news_list.append(
        {
            "source": entry.source.title if "source" in entry else "Google News",
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
        }
    )

top_items = trends_list + news_list

infamous_keywords = [
    "infamous",
    "notorious",
    "serial",
    "killer",
    "murder",
    "abduction",
    "missing",
    "cold case",
    "unsolved",
    "mystery",
    "homicide",
]


def _is_infamous(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in infamous_keywords)


infamous_items = [item for item in top_items if _is_infamous(item.get("title", ""))]
if len(infamous_items) >= 2:
    # pick the two best (score if available, otherwise keep order)
    infamous_items = sorted(infamous_items, key=lambda x: x.get("score", 0), reverse=True)[:2]
    selected_items = infamous_items
else:
    selected_items = top_items

if not selected_items:
    # fallback list if both sources are empty
    selected_items = [
        {"source": "Fallback", "title": "The Lindbergh baby kidnapping case"},
        {"source": "Fallback", "title": "The Tylenol murders"},
        {"source": "Fallback", "title": "The disappearance of Maura Murray"},
    ]

content = ""
for i, item in enumerate(selected_items, 1):
    content += f"{i}. {item['title']} ({item['source']})\n"
content = content.strip()
content = content.split("\n")
contents(content)
