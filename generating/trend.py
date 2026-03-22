from urllib.parse import quote

import feedparser
from pytrends.request import TrendReq

from content import contents

keyword = "crime darkest"
keyword = quote(keyword)
# Worldwide trends
pytrends = TrendReq(hl="en", tz=0)

pytrends.build_payload([keyword], geo="")
trending = pytrends.related_queries()[keyword]['top']

trends_list = []

if trending is not None:
    for i, row in trending.head(5).iterrows():
        trends_list.append({
            "source": "Google Trends",
            "title": row['query'],
            "score": int(row['value'])
        })

# Worldwide Google News (no region lock)
rss_url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '+')}"

feed = feedparser.parse(rss_url)

news_list = []

for entry in feed.entries[:100]:
    news_list.append({
        "source": entry.source.title if "source" in entry else "Google News",
        "title": entry.title,
        "link": entry.link,
        "published": entry.published
    })

top_items = trends_list + news_list
content=''
for i, item in enumerate(top_items, 1):
    content += f"{i}. {item['title']} ({item['source']})\n"
content = content.strip()
content = content.split("\n")
contents(content)
