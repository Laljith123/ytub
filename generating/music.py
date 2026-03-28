import os
import csv
from yt_dlp import YoutubeDL
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents

OUTPUT_JSON = PROJECT_ROOT / "output.json"

print("Loading output.json...")
with open(OUTPUT_JSON, "r") as f:
    data = json.load(f)
    print("Loaded data from output.json:")
    print(data)
