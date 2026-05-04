import json
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
from pytrends.exceptions import TooManyRequestsError
from pytrends.request import TrendReq

from content import contents

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if load_dotenv is not None:
    load_dotenv()


# Generic seed keywords used to discover trends.
# These are broad search seeds only, not hardcoded case names.
SEARCH_KEYWORDS = ['dark crime',
 'unsolved crime',
 'mysterious murder',
 'serial killer case',
 'cold case mystery',
 'true crime story',
 'infamous crime case',
 'unsolved murder mystery',
 'notorious criminal case',
 'dark investigation case',
 'missing person case',
 'unsolved disappearance',
 'serial killer mystery',
 'true crime unsolved',
 'homicide investigation',
 'crime mystery story',
 'dark criminal case',
 'unsolved homicide',
 'mysterious case file',
 'infamous murder story',
 'cold case investigation',
 'unsolved kidnapping',
 'dark crime mystery',
 'serial killer investigation',
 'missing case mystery',
 'unsolved true crime',
 'notorious murder case',
 'dark unsolved case',
 'mystery homicide case',
 'true crime mystery',
 'unsolved serial killer',
 'missing person mystery',
 'dark crime investigation',
 'cold case murder',
 'infamous killer case',
 'unsolved mystery case',
 'true crime homicide',
 'mysterious disappearance',
 'dark criminal mystery',
 'serial killer story',
 'unsolved case india',
 'dark crime india',
 'murder mystery india',
 'true crime india',
 'unsolved murder india',
 'missing case india',
 'serial killer india',
 'crime investigation india',
 'dark case india',
 'cold case india',
 'unsolved case usa',
 'dark crime usa',
 'murder mystery usa',
 'true crime usa',
 'unsolved murder usa',
 'missing case usa',
 'serial killer usa',
 'crime investigation usa',
 'dark case usa',
 'cold case usa',
 'gruesome murder case',
 'horrific crime story',
 'disturbing crime case',
 'darkest crime ever',
 'shocking murder mystery',
 'unsolved brutal case',
 'serial killer brutal crimes',
 'crime documentary case',
 'true crime files',
 'real crime mystery',
 'unsolved violent crime',
 'darkest unsolved mystery',
 'serial killer investigation case',
 'crime files mystery',
 'dark crime documentary',
 'real murder mystery',
 'unsolved serial crimes',
 'dark criminal files',
 'mysterious killings',
 'unsolved crime files',
 'famous serial killer',
 'notorious crime mystery',
 'darkest murder case',
 'unsolved mystery files',
 'true crime investigation',
 'crime case analysis',
 'unsolved crime analysis',
 'dark mystery case',
 'serial killer files',
 'real crime files',
 'unsolved horror crime',
 'dark investigation files',
 'true crime deep dive',
 'murder case breakdown',
 'unsolved criminal files',
 'crime mystery breakdown',
 'dark case analysis',
 'serial killer deep dive',
 'cold case files',
 'true crime breakdown',
 'missing child case',
 'kidnapping mystery case',
 'unsolved abduction',
 'dark kidnapping case',
 'missing person investigation',
 'unsolved missing case',
 'mysterious kidnapping',
 'true crime missing',
 'cold case disappearance',
 'dark missing mystery',
 'unsolved disappearance case',
 'missing person files',
 'kidnapping investigation',
 'dark crime disappearance',
 'unsolved abduction case',
 'true crime kidnapping',
 'mystery missing person',
 'cold case missing',
 'dark mystery disappearance',
 'unsolved vanish case',
 'psychopath killer case',
 'serial murderer files',
 'dark psychology crime',
 'criminal mind case',
 'true crime psychology',
 'killer profile case',
 'dark criminal psychology',
 'serial killer profile',
 'unsolved psychopath case',
 'crime behavior analysis',
 'dark criminal behavior',
 'killer investigation files',
 'true crime profiling',
 'serial killer psychology',
 'dark crime profiling',
 'unsolved killer profile',
 'criminal investigation files',
 'dark mind crime',
 'killer case study',
 'true crime study',
 'real crime horror',
 'dark real case',
 'unsolved horror mystery',
 'true crime horror',
 'dark mystery horror',
 'crime horror case',
 'serial killer horror',
 'unsolved horror crime',
 'dark crime horror story',
 'real horror crime',
 'unsolved horror case',
 'true crime horror files',
 'dark case horror',
 'murder horror mystery',
 'crime horror files',
 'serial horror crimes',
 'dark mystery horror case',
 'unsolved scary case',
 'real crime scary',
 'dark scary crime',
 'unsolved crime 2024',
 'unsolved crime 2025',
 'recent crime mystery',
 'latest murder case',
 'recent unsolved case',
 'latest crime investigation',
 'new true crime case',
 'recent dark crime',
 'latest serial killer case',
 'recent cold case',
 'modern crime mystery',
 'recent murder mystery',
 'new unsolved mystery',
 'latest crime files',
 'recent investigation case',
 'modern unsolved crime',
 'latest criminal case',
 'recent true crime',
 'modern crime files',
 'latest mystery case',
 'ancient crime mystery',
 'historical murder case',
 'old unsolved crime',
 'historic crime case',
 'ancient murder mystery',
 'old serial killer case',
 'historic cold case',
 'ancient crime files',
 'old mystery case',
 'historic investigation',
 'ancient crime investigation',
 'old crime mystery',
 'historical unsolved case',
 'ancient criminal files',
 'historic mystery case',
 'old murder investigation',
 'ancient case files',
 'historical crime files',
 'old cold case',
 'ancient mystery files',
 'famous crime mystery',
 'top crime cases',
 'biggest unsolved crime',
 'most famous murders',
 'popular true crime',
 'top serial killer cases',
 'famous unsolved mysteries',
 'top crime investigation',
 'popular murder mystery',
 'famous case files',
 'top dark crimes',
 'famous criminal cases',
 'most shocking crimes',
 'popular crime stories',
 'top murder cases',
 'famous investigation files',
 'biggest crime mystery',
 'popular unsolved crime',
 'top mystery cases',
 'famous cold cases']


# -------------------------
# Config helpers
# -------------------------

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _split_config_values(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[\n,;|]+", raw)
    return [p.strip() for p in parts if p.strip()]


def _load_keywords() -> list[str]:
    """
    Loads trend seed keywords from the in-code SEARCH_KEYWORDS list.

    This keeps the old behavior: keywords come from the Python file itself,
    not from a separate text file. TRENDS_KEYWORDS can still add extra generic
    seed keywords from .env, but no external keyword file is used.
    """
    keywords: list[str] = list(SEARCH_KEYWORDS)
    keywords.extend(_split_config_values(os.getenv("TRENDS_KEYWORDS", "")))

    cleaned: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        norm = _normalize(keyword)
        if not norm or norm in seen:
            continue
        cleaned.append(keyword.strip())
        seen.add(norm)

    if not cleaned:
        raise RuntimeError("No trend keywords available in SEARCH_KEYWORDS or TRENDS_KEYWORDS.")

    return cleaned


def _pick_keywords(keywords: list[str]) -> list[str]:
    count = max(1, _env_int("TRENDS_KEYWORD_SAMPLE", "8"))
    mode = os.getenv("TRENDS_KEYWORD_MODE", "random").strip().lower()

    if count >= len(keywords):
        selected = list(keywords)
    elif mode == "first":
        selected = keywords[:count]
    else:
        selected = random.sample(keywords, count)

    return selected


USE_PYTRENDS = _env_bool("TRENDS_USE_PYTRENDS", "1")
USE_GOOGLE_NEWS = _env_bool("TRENDS_USE_GOOGLE_NEWS", "1")
USE_YOUTUBE_SUGGEST = _env_bool("TRENDS_USE_YOUTUBE_SUGGEST", "1")
USE_AI_JUDGE = _env_bool("TRENDS_USE_AI_JUDGE", "1")

PYTRENDS_RETRIES = _env_int("TRENDS_PYTRENDS_RETRIES", "3")
PYTRENDS_BACKOFF = _env_float("TRENDS_PYTRENDS_BACKOFF", "2")
PYTRENDS_GEO = os.getenv("TRENDS_PYTRENDS_GEO", "")
PYTRENDS_HL = os.getenv("TRENDS_PYTRENDS_HL", "en")
PYTRENDS_TZ = _env_int("TRENDS_PYTRENDS_TZ", "0")

NEWS_HL = os.getenv("TRENDS_NEWS_HL", "en-US")
NEWS_GL = os.getenv("TRENDS_NEWS_GL", "US")
NEWS_CEID = os.getenv("TRENDS_NEWS_CEID", "US:en")
NEWS_LIMIT_PER_KEYWORD = _env_int("TRENDS_NEWS_LIMIT_PER_KEYWORD", "8")
NEWS_QUERY_SUFFIX = os.getenv("TRENDS_NEWS_QUERY_SUFFIX", "").strip()

YOUTUBE_SUGGEST_LIMIT_PER_KEYWORD = _env_int("TRENDS_YT_SUGGEST_LIMIT_PER_KEYWORD", "10")
REQUEST_TIMEOUT = _env_float("TRENDS_REQUEST_TIMEOUT", "10")

CANDIDATE_LIMIT_FOR_AI = _env_int("TRENDS_AI_CANDIDATE_LIMIT", "60")
SELECTED_TREND_COUNT = _env_int("TRENDS_SELECTED_COUNT", "5")
AI_MODEL = os.getenv("TRENDS_AI_MODEL", os.getenv("CONTENT_MODEL", "openai/gpt-oss-120b"))
AI_BASE_URL = os.getenv("TRENDS_AI_BASE_URL", os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
AI_API_KEY = os.getenv("TRENDS_AI_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
AI_MAX_ATTEMPTS = _env_int("TRENDS_AI_MAX_ATTEMPTS", "3")
AI_MAX_TOKENS = _env_int("TRENDS_AI_MAX_TOKENS", "4096")
AI_TEMPERATURE = _env_float("TRENDS_AI_TEMPERATURE", "0.25")
AI_TOP_P = _env_float("TRENDS_AI_TOP_P", "0.9")
AI_ENABLE_THINKING = _env_bool("TRENDS_AI_ENABLE_THINKING", "1")
AI_REASONING_BUDGET = _env_int("TRENDS_AI_REASONING_BUDGET", "4096")

OUTPUT_JSON = Path(os.getenv("OUTPUT_JSON_PATH", str(PROJECT_ROOT / "output.json")))
COLLISION_SIM_THRESHOLD = _env_float("TRENDS_COLLISION_SIM", "0.72")
USED_PROMPT_LIMIT = _env_int("TRENDS_USED_PROMPT_LIMIT", "120")
AI_REQUIRE_SOURCE_MATCH = _env_bool("TRENDS_AI_REQUIRE_SOURCE_MATCH", "1")



# -------------------------
# Text helpers
# -------------------------

def _normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_noise(title: str) -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    title = re.sub(r"\s+-\s+[^-]{2,80}$", "", title).strip()
    return title


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _parse_json_dict(text: str) -> dict:
    if not text or not text.strip():
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        candidate = _extract_json_object(text)
        if not candidate:
            return {}
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}




def _strip_trend_text(text: str) -> str:
    """Keep trend strings compatible with content.py's trend cleaner."""
    value = str(text or "").strip()
    value = re.sub(r"^\s*\d+\s*[\.)\-]\s*", "", value)
    value = re.sub(r"^\s*\d+\s+", "", value)
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    value = re.sub(r"\s+-\s+[^-]{2,80}$", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _content_trend_text(text: str) -> str:
    value = _strip_trend_text(text)
    if not value or value.isdigit():
        return ""
    return value


def _token_set(text: str) -> set[str]:
    return set(_normalize(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _topic_collides(candidate: str, used_topics: list[str]) -> bool:
    candidate_clean = _content_trend_text(candidate)
    candidate_norm = _normalize(candidate_clean)
    if not candidate_norm:
        return True

    candidate_tokens = _token_set(candidate_clean)
    for used in used_topics:
        used_clean = _content_trend_text(used)
        used_norm = _normalize(used_clean)
        if not used_norm:
            continue
        if candidate_norm == used_norm:
            return True
        if len(candidate_norm.split()) >= 2 and candidate_norm in used_norm:
            return True
        if len(used_norm.split()) >= 2 and used_norm in candidate_norm:
            return True
        if _jaccard(candidate_tokens, _token_set(used_clean)) >= COLLISION_SIM_THRESHOLD:
            return True
    return False


def _load_used_topics() -> list[str]:
    """Read output.json so trend.py avoids sending already-used ideas into content.py."""
    if not OUTPUT_JSON.exists():
        return []

    try:
        data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Trend collision check skipped: {OUTPUT_JSON} is not valid JSON.")
        return []
    except OSError as exc:
        print(f"Trend collision check skipped: unable to read {OUTPUT_JSON}: {exc}")
        return []

    if not isinstance(data, list):
        return []

    used: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        for key in ("trend", "title"):
            value = _content_trend_text(str(item.get(key) or ""))
            norm = _normalize(value)
            if value and norm and norm not in seen:
                used.append(value)
                seen.add(norm)
    return used


def _filter_used_candidates(candidates: list[dict], used_topics: list[str]) -> list[dict]:
    if not used_topics:
        return candidates

    filtered: list[dict] = []
    removed = 0
    for item in candidates:
        title = str(item.get("title") or "")
        keyword = str(item.get("keyword") or "")
        if _topic_collides(title, used_topics) or _topic_collides(keyword, used_topics):
            removed += 1
            continue
        filtered.append(item)

    if removed:
        print(f"Filtered {removed} candidates already used in output.json/title history.")
    return filtered


def _candidate_title_norms(candidates: list[dict]) -> set[str]:
    return {_normalize(str(item.get("title") or "")) for item in candidates if item.get("title")}

# -------------------------
# Candidate collection
# -------------------------

def _add_candidate(
    candidates: list[dict],
    seen: set[str],
    *,
    title: str,
    source: str,
    keyword: str,
    score: int = 0,
    link: str = "",
    published: str = "",
) -> None:
    clean_title = _strip_noise(title)
    norm = _normalize(clean_title)
    if not norm or norm in seen:
        return
    seen.add(norm)
    candidates.append(
        {
            "title": clean_title,
            "source": source,
            "keyword": keyword,
            "score": int(score or 0),
            "link": link,
            "published": published,
        }
    )


def _collect_google_trends(keywords: list[str], candidates: list[dict], seen: set[str]) -> None:
    if not USE_PYTRENDS:
        return

    pytrends = TrendReq(hl=PYTRENDS_HL, tz=PYTRENDS_TZ)

    for keyword in keywords:
        for attempt in range(1, PYTRENDS_RETRIES + 1):
            try:
                pytrends.build_payload([keyword], geo=PYTRENDS_GEO)
                related = pytrends.related_queries().get(keyword) or {}

                for label in ("top", "rising"):
                    frame = related.get(label)
                    if frame is None:
                        continue
                    for _, row in frame.head(10).iterrows():
                        query = str(row.get("query") or "").strip()
                        value = row.get("value", 0)
                        try:
                            score = int(value)
                        except Exception:
                            score = 0
                        _add_candidate(
                            candidates,
                            seen,
                            title=query,
                            source=f"Google Trends {label}",
                            keyword=keyword,
                            score=score,
                        )
                break
            except TooManyRequestsError as exc:
                if attempt == PYTRENDS_RETRIES:
                    print(f"Pytrends rate-limited for '{keyword}'. Skipping. {exc}")
                    break
                sleep_for = PYTRENDS_BACKOFF * attempt
                print(f"Pytrends rate-limited for '{keyword}'. Retrying in {sleep_for:.1f}s...")
                time.sleep(sleep_for)
            except Exception as exc:
                if attempt == PYTRENDS_RETRIES:
                    print(f"Pytrends failed for '{keyword}'. Skipping. {exc}")
                    break
                time.sleep(PYTRENDS_BACKOFF * attempt)


def _collect_google_news(keywords: list[str], candidates: list[dict], seen: set[str]) -> None:
    if not USE_GOOGLE_NEWS:
        return

    for keyword in keywords:
        query = f"{keyword} {NEWS_QUERY_SUFFIX}".strip()
        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(query)}&hl={quote_plus(NEWS_HL)}&gl={quote_plus(NEWS_GL)}&ceid={quote_plus(NEWS_CEID)}"
        )
        feed = feedparser.parse(rss_url)
        entries = list(feed.entries or [])
        random.shuffle(entries)

        for entry in entries[:NEWS_LIMIT_PER_KEYWORD]:
            source = entry.source.title if "source" in entry else "Google News"
            _add_candidate(
                candidates,
                seen,
                title=getattr(entry, "title", ""),
                source=source,
                keyword=keyword,
                score=0,
                link=getattr(entry, "link", ""),
                published=getattr(entry, "published", ""),
            )


def _collect_youtube_suggestions(keywords: list[str], candidates: list[dict], seen: set[str]) -> None:
    if not USE_YOUTUBE_SUGGEST:
        return
    if requests is None:
        print("YouTube suggestions skipped: requests is not installed.")
        return

    for keyword in keywords:
        url = (
            "https://suggestqueries.google.com/complete/search?"
            f"client=firefox&ds=yt&q={quote_plus(keyword)}"
        )
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            print(f"YouTube suggestions failed for '{keyword}'. Skipping. {exc}")
            continue

        suggestions = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        for suggestion in suggestions[:YOUTUBE_SUGGEST_LIMIT_PER_KEYWORD]:
            _add_candidate(
                candidates,
                seen,
                title=str(suggestion),
                source="YouTube Suggest",
                keyword=keyword,
                score=0,
            )


def collect_candidates(keywords: list[str]) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()

    _collect_google_trends(keywords, candidates, seen)
    _collect_google_news(keywords, candidates, seen)
    _collect_youtube_suggestions(keywords, candidates, seen)

    candidates.sort(key=lambda item: int(item.get("score") or 0), reverse=True)
    return candidates


# -------------------------
# AI trend judge
# -------------------------

def _build_trend_judge_prompt(candidates: list[dict], used_topics: list[str]) -> str:
    trimmed = candidates[:CANDIDATE_LIMIT_FOR_AI]
    compact_candidates = [
        {
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "keyword": item.get("keyword", ""),
            "score": item.get("score", 0),
            "published": item.get("published", ""),
        }
        for item in trimmed
    ]
    compact_used = used_topics[:USED_PROMPT_LIMIT]

    return (
        "You are a YouTube Shorts trend selector for a true-crime and mystery channel. "
        "Pick the best topics that can become engaging, respectful, non-graphic Shorts. "
        "Use ONLY the candidate list. Do not invent cases. Do not use hardcoded examples. "
        "Reject movies, TV shows, fictional stories, urban legends, vague category keywords, "
        "political arguments, celebrity gossip without a clear case, and overly graphic topics. "
        "Prefer real searchable case names, clear mystery angles, missing-person/cold-case style curiosity, "
        "recent relevance, strong hook potential, and safe YouTube wording. "
        "Avoid every used topic/title in the previously_used list. Do not choose the same case with a new headline. "
        "The output goes directly into content.py, so case_name must be a clean trend string only: "
        "no numbering, no source name, no URL, no date, no extra description, and no parenthetical source. "
        f"Select exactly {SELECTED_TREND_COUNT} topics unless fewer usable topics exist. "
        "Return ONE valid JSON object ONLY with this schema: "
        "{\"selected\":[{\"case_name\":\"...\",\"angle\":\"...\",\"source_title\":\"...\","
        "\"source\":\"...\",\"shorts_score\":0,\"safety\":\"safe\"}]}. "
        "case_name must be concise, searchable, content.py-compatible, and without numbering/source names. "
        "source_title must exactly match the candidate title you used. "
        "shorts_score must be 0-100. "
        "No markdown, no commentary, no reasoning. "
        f"Previously used topics/titles JSON: {json.dumps(compact_used, ensure_ascii=False)} "
        f"Candidates JSON: {json.dumps(compact_candidates, ensure_ascii=False)}"
    )


def _run_ai_judge(candidates: list[dict], used_topics: list[str]) -> list[dict]:
    if not USE_AI_JUDGE:
        return []
    if OpenAI is None:
        print("AI trend judge skipped: openai package is not installed.")
        return []
    if not AI_API_KEY:
        print("AI trend judge skipped: NVIDIA_API_KEY/TRENDS_AI_API_KEY is missing.")
        return []
    if not candidates:
        return []

    client = OpenAI(base_url=AI_BASE_URL, api_key=AI_API_KEY)
    prompt = _build_trend_judge_prompt(candidates, used_topics)

    for attempt in range(1, AI_MAX_ATTEMPTS + 1):
        try:
            completion = client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You must output ONLY valid JSON. No markdown, no code fences, "
                            "no commentary, no reasoning."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=AI_TEMPERATURE,
                top_p=AI_TOP_P,
                max_tokens=AI_MAX_TOKENS,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": AI_ENABLE_THINKING},
                    "reasoning_budget": AI_REASONING_BUDGET if AI_ENABLE_THINKING else 0,
                },
                stop=["\n[Reasoning]", "\nReasoning:", "\n[Analysis]", "\nAnalysis:"],
            )
            text = (completion.choices[0].message.content or "").strip()
            parsed = _parse_json_dict(text)
            selected = parsed.get("selected") if isinstance(parsed, dict) else None
            if isinstance(selected, list):
                cleaned = _validate_selected(selected, candidates, used_topics)
                if cleaned:
                    return cleaned
        except Exception as exc:
            print(f"AI trend judge failed ({attempt}/{AI_MAX_ATTEMPTS}): {exc}")
            time.sleep(min(2 * attempt, 8))

    return []


def _validate_selected(items: list[dict], candidates: list[dict], used_topics: list[str]) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    title_norms = _candidate_title_norms(candidates)

    for item in items:
        if not isinstance(item, dict):
            continue
        case_name = _content_trend_text(str(item.get("case_name") or ""))
        source_title = str(item.get("source_title") or "").strip()
        if not case_name:
            continue
        if AI_REQUIRE_SOURCE_MATCH and _normalize(source_title) not in title_norms:
            continue
        if _topic_collides(case_name, used_topics) or _topic_collides(source_title, used_topics):
            continue
        norm = _normalize(case_name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        selected.append(
            {
                "case_name": case_name,
                "angle": str(item.get("angle") or "").strip(),
                "source_title": source_title,
                "source": str(item.get("source") or "").strip(),
                "shorts_score": int(item.get("shorts_score") or 0),
                "safety": str(item.get("safety") or "safe").strip() or "safe",
            }
        )
        if len(selected) >= SELECTED_TREND_COUNT:
            break

    return selected


def _fallback_selected(candidates: list[dict], used_topics: list[str]) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()

    for item in candidates:
        title = _content_trend_text(str(item.get("title") or ""))
        if not title or _topic_collides(title, used_topics):
            continue
        norm = _normalize(title)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        selected.append(
            {
                "case_name": title,
                "angle": "",
                "source_title": title,
                "source": str(item.get("source") or "").strip(),
                "shorts_score": int(item.get("score") or 0),
                "safety": "unchecked",
            }
        )
        if len(selected) >= SELECTED_TREND_COUNT:
            break

    return selected


# -------------------------
# Main flow
# -------------------------

def main() -> None:
    try:
        all_keywords = _load_keywords()
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

    used_topics = _load_used_topics()
    if used_topics:
        print(f"Loaded {len(used_topics)} used trends/titles from output.json for collision checks.")

    keywords = _pick_keywords(all_keywords)
    print(f"Using {len(keywords)} trend seed keywords.")

    candidates = collect_candidates(keywords)
    candidates = _filter_used_candidates(candidates, used_topics)
    if not candidates:
        print("No unused trends found from configured sources. Aborting.")
        sys.exit(1)

    print(f"Collected {len(candidates)} unique unused trend candidates.")

    selected = _run_ai_judge(candidates, used_topics)
    if selected:
        print(f"AI trend judge selected {len(selected)} topics using {AI_MODEL}.")
    else:
        selected = _fallback_selected(candidates, used_topics)
        print(f"Using fallback selection with {len(selected)} topics.")

    if not selected:
        print("No usable unused trends selected. Aborting.")
        sys.exit(1)

    trend_titles: list[str] = []
    seen_titles: set[str] = set()
    for item in selected:
        trend = _content_trend_text(item.get("case_name", ""))
        norm = _normalize(trend)
        if not trend or not norm or norm in seen_titles or _topic_collides(trend, used_topics):
            continue
        trend_titles.append(trend)
        seen_titles.add(norm)

    if not trend_titles:
        print("Selected trends collided with previous output. Aborting.")
        sys.exit(1)

    print("Selected trends passed to content.py:")
    for i, item in enumerate(selected, 1):
        trend = _content_trend_text(item.get("case_name", ""))
        if trend not in trend_titles:
            continue
        score = item.get("shorts_score", 0)
        source = item.get("source") or "unknown source"
        angle = item.get("angle") or ""
        print(f"{i}. {trend} | score={score} | source={source}")
        if angle:
            print(f"   angle: {angle}")

    contents(trend_titles)


if __name__ == "__main__":
    main()
