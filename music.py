import json
from pathlib import Path

# ✅ Correct path (same folder as music.py)
PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_JSON = PROJECT_ROOT / "output.json"

print("Looking for:", OUTPUT_JSON)

if not OUTPUT_JSON.exists():
    print("❌ File not found!")
    exit()

print("Loading output.json...")

with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print("✅ Loaded data:")
print(data)
