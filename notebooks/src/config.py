"""Paths and constants used by every notebook, so nothing is hard-coded twice."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]     # notebooks/src/config.py -> repo root

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_LABELS = ROOT / "data" / "labels"
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"

RAW_FILE = DATA_RAW / "youtube_comments_gyIxD5K4lUo.csv"
CLEAN_FILE = DATA_PROCESSED / "comments_clean.csv"          # all comments after cleaning
LABELED_FILE = DATA_LABELS / "labeled_comments.csv"         # hand-labelled sample
TRAIN_FILE = DATA_PROCESSED / "train.csv"                   # fixed train split
TEST_FILE = DATA_PROCESSED / "test_300.csv"                 # fixed 300-row test subset
HARD_FILE = DATA_LABELS / "hard_cases.csv"                  # 30 hand-written hard sentences

PREDICTIONS_FILE = RESULTS / "predictions_test.csv"         # one column per model
HARD_PREDICTIONS_FILE = RESULTS / "predictions_hard.csv"
METRICS_FILE = RESULTS / "metrics.csv"
LLM_CACHE = RESULTS / "llm_cache.jsonl"

SEED = 42
TEST_SIZE = 300
LABELS = ["negative", "neutral", "positive"]

for p in (DATA_PROCESSED, DATA_LABELS, RESULTS, CHARTS):
    p.mkdir(parents=True, exist_ok=True)
