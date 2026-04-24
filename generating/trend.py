import os
import sys
import time
from urllib.parse import quote

import feedparser
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import random
from content import contents
search_keywords = [
"dark crime","unsolved crime","mysterious murder","serial killer case","cold case mystery","true crime story","infamous crime case","unsolved murder mystery","notorious criminal case","dark investigation case",
"missing person case","unsolved disappearance","serial killer mystery","true crime unsolved","homicide investigation","crime mystery story","dark criminal case","unsolved homicide","mysterious case file","infamous murder story",
"cold case investigation","unsolved kidnapping","dark crime mystery","serial killer investigation","missing case mystery","unsolved true crime","notorious murder case","dark unsolved case","mystery homicide case","true crime mystery",
"unsolved serial killer","missing person mystery","dark crime investigation","cold case murder","infamous killer case","unsolved mystery case","true crime homicide","mysterious disappearance","dark criminal mystery","serial killer story",

"unsolved case india","dark crime india","murder mystery india","true crime india","unsolved murder india","missing case india","serial killer india","crime investigation india","dark case india","cold case india",
"unsolved case usa","dark crime usa","murder mystery usa","true crime usa","unsolved murder usa","missing case usa","serial killer usa","crime investigation usa","dark case usa","cold case usa",

"gruesome murder case","horrific crime story","disturbing crime case","darkest crime ever","shocking murder mystery","unsolved brutal case","serial killer brutal crimes","crime documentary case","true crime files","real crime mystery",
"unsolved violent crime","darkest unsolved mystery","serial killer investigation case","crime files mystery","dark crime documentary","real murder mystery","unsolved serial crimes","dark criminal files","mysterious killings","unsolved crime files",

"famous serial killer","notorious crime mystery","darkest murder case","unsolved mystery files","true crime investigation","crime case analysis","unsolved crime analysis","dark mystery case","serial killer files","real crime files",
"unsolved horror crime","dark investigation files","true crime deep dive","murder case breakdown","unsolved criminal files","crime mystery breakdown","dark case analysis","serial killer deep dive","cold case files","true crime breakdown",

"missing child case","kidnapping mystery case","unsolved abduction","dark kidnapping case","missing person investigation","unsolved missing case","mysterious kidnapping","true crime missing","cold case disappearance","dark missing mystery",
"unsolved disappearance case","missing person files","kidnapping investigation","dark crime disappearance","unsolved abduction case","true crime kidnapping","mystery missing person","cold case missing","dark mystery disappearance","unsolved vanish case",

"psychopath killer case","serial murderer files","dark psychology crime","criminal mind case","true crime psychology","killer profile case","dark criminal psychology","serial killer profile","unsolved psychopath case","crime behavior analysis",
"dark criminal behavior","killer investigation files","true crime profiling","serial killer psychology","dark crime profiling","unsolved killer profile","criminal investigation files","dark mind crime","killer case study","true crime study",

"real crime horror","dark real case","unsolved horror mystery","true crime horror","dark mystery horror","crime horror case","serial killer horror","unsolved horror crime","dark crime horror story","real horror crime",
"unsolved horror case","true crime horror files","dark case horror","murder horror mystery","crime horror files","serial horror crimes","dark mystery horror case","unsolved scary case","real crime scary","dark scary crime",

"unsolved crime 2024","unsolved crime 2025","recent crime mystery","latest murder case","recent unsolved case","latest crime investigation","new true crime case","recent dark crime","latest serial killer case","recent cold case",
"modern crime mystery","recent murder mystery","new unsolved mystery","latest crime files","recent investigation case","modern unsolved crime","latest criminal case","recent true crime","modern crime files","latest mystery case",

"ancient crime mystery","historical murder case","old unsolved crime","historic crime case","ancient murder mystery","old serial killer case","historic cold case","ancient crime files","old mystery case","historic investigation",
"ancient crime investigation","old crime mystery","historical unsolved case","ancient criminal files","historic mystery case","old murder investigation","ancient case files","historical crime files","old cold case","ancient mystery files",

"famous crime mystery","top crime cases","biggest unsolved crime","most famous murders","popular true crime","top serial killer cases","famous unsolved mysteries","top crime investigation","popular murder mystery","famous case files",
"top dark crimes","famous criminal cases","most shocking crimes","popular crime stories","top murder cases","famous investigation files","biggest crime mystery","popular unsolved crime","top mystery cases","famous cold cases",
]
keyword = search_keywords(random.randint(0, len(search_keywords)-1))
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
start=int(random.random()*100)
for entry in feed.entries[0:start]:
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
    print("No trends found from Google Trends or Google News. Aborting.")
    sys.exit(1)

content = ""
for i, item in enumerate(selected_items, 1):
    content += f"{i}. {item['title']} ({item['source']})\n"
content = content.strip()
content = content.split("\n")
contents(content)
