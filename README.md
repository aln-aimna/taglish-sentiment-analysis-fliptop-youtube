# Lupet or Luto? Sentiment Analysis of Taglish Comments on the Most Viewed FlipTop Battle

Milestone 4 - *Sentiment Analysis from Classic NLP to LLMs.*

YouTube comments on FlipTop's most watched battle, **Loonie & Abra vs Shehyee & Smugglaz** (Dos Por Dos 2012,
[video `gyIxD5K4lUo`](https://www.youtube.com/watch?v=gyIxD5K4lUo)), are classified as **positive / neutral /
negative** with three approaches, then compared on accuracy, speed and cost:

| Tier | Approach | Tool | Notebook |
|---|---|---|---|
| 1. Rule-based | Word-score lexicon, no training (stock, and with a Taglish word list added) | VADER | `02_vader.ipynb` |
| 2. Classic ML | Bag-of-words features + Naive Bayes, trained on 700 labelled comments | NLTK + Naive Bayes | `03_classic_ml.ipynb` |
| 3. LLM | Prompted with instructions, no training (zero-shot) | Claude API (Haiku 5.5) | `04_llm.ipynb` |

**The question:** when is a simple model good enough, and when does an LLM earn its cost?

The comments are mostly **Tagalog**, with English and **Taglish** (both in one sentence), full of slang,
misspellings, laughter, emoji and profanity, so a big part of the project is **cleaning Taglish text**.

## Results

Every model is scored on the same fixed, stratified **300-comment test set** and the same **30 hard cases**.
Main metric: **macro-F1**.

| Model | Accuracy | Macro-F1 | Hard-cases acc. | Sec / 1k preds | Cost / 1k preds |
|---|---|---|---|---|---|
| VADER (stock) | *run 02* | | | | $0 |
| VADER + Taglish lexicon | *run 02* | | | | $0 |
| NLTK + Naive Bayes | *run 03* | | | | ~$0 |
| Claude Haiku 5.5 zero-shot | *run 04* | | | | ≈ $0.04 |

After running notebooks 02 → 05, the full table is in `results/metrics.csv` and `results/results_table.md`
(ready to paste here), and the charts are in `results/charts/`.

## Repo layout

```
taglish-sentiment-fliptop-comments/
  README.md
  requirements.txt
  prompts.md              the Tier 3 prompt
  .env.example            copy to .env and add your API keys (.env is never committed)
  data/
    youtube_comments_scraper.py   downloads the comments with the YouTube Data API -> data/raw/
    raw/                  youtube_comments_gyIxD5K4lUo.csv (78,566 comments and replies)
    labels/               labeled_comments.csv (1,000), hard_cases.csv (30), LABELING_GUIDE.md
    processed/            train.csv (700), test_300.csv (300), comments_clean.csv (rebuilt by 01)
  notebooks/
    01_data.ipynb         Day 1 - environment check, cleaning, EDA, fixed train / test split
    02_vader.ipynb        Day 2 - Tier 1: VADER baseline
    03_classic_ml.ipynb   Day 3 - Tier 2: NLTK + Naive Bayes
    04_llm.ipynb          Day 4 - Tier 3: LLM zero-shot, cost and latency log
    05_compare.ipynb      Day 5 - results table, charts, error analysis
    src/                  helper code the notebooks import (cleaning, VADER word list, LLM client, metrics)
  results/
    metrics.csv           one row per model
    charts/               macro-F1 and hard-case bar charts, confusion matrices, EDA charts, word cloud
    llm_cache.jsonl       every LLM answer with tokens and latency (created by 04)
    nltk_naive_bayes.pickle   the trained Tier 2 model (created by 03)
    predictions_*.csv     each model's prediction per comment (used by 05)
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate                 # Mac / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys (never commit `.env`):

```
ANTHROPIC_API_KEY=sk-ant-...
YOUTUBE_API_KEY=...          # only needed to re-scrape
```

Then open `notebooks/` in Jupyter or VS Code and run **01 → 05 in order**.

| Notebook | Time | Notes |
|---|---|---|
| `01_data` | ~15 s | rebuilds `comments_clean.csv` and the fixed split (same rows every time, seed 42) |
| `02_vader` | seconds | |
| `03_classic_ml` | ~10 s | saves the trained model to `results/` |
| `04_llm` | a few minutes the first time | 330 API calls (300 test + 30 hard), about US$0.01 with Haiku 5.5; re-runs read `results/llm_cache.jsonl` and cost nothing |
| `05_compare` | seconds | writes `results/results_table.md` and the charts |

To re-scrape the comments: `python data/youtube_comments_scraper.py` (from the repo root). It writes to `data/raw/`.

## Data and labels

- **Scraped:** 78,566 comments and replies (2012 - 2026) with the official YouTube Data API v3.
- **Cleaned:** 68,097 comments after removing empty, symbol-only and duplicate comments; 67,000 are short enough
  (≤ 400 characters) to model.
- **Labelled:** 1,000 random comments - 435 negative, 325 positive, 240 neutral - following
  `data/labels/LABELING_GUIDE.md`. First-pass labels were made with Claude (Opus) as annotator and should be
  reviewed by a person (column `reviewed`). Because Tier 3 is also Claude, read the Tier 3 score with that in mind.
- **Split:** stratified, seed 42 → 700 train / 300 test, fixed once and never changed.
- **Hard cases:** 30 hand-written Taglish sentences: negation, curse words used as praise, sarcasm, *pero / kaso*
  contrast, questions, quoted bars, like-bait.

## Taglish cleaning in one table

| Step | Example |
|---|---|
| remove URLs and @mentions, fix fancy unicode | `@user 𝗴𝗮𝗹𝗶𝗻𝗴` → `galing` |
| laughter → one token | `HAHAHAHA`, `hehe`, `=))`, 😂 → `haha` |
| stretched letters | `lupeeeet` → `lupet` |
| spelling / slang map (~140 entries) | `hnd nmn tlga` → `hindi naman talaga`, `lufet` → `lupet`, `sheyee` → `shehyee` |
| phrases joined | `walang kwenta` → `walang_kwenta`, `team s.s` → `team_ss` |
| emoji → words | 🔥 → `fire`, 👏 → `clap`, 🙄 → `eyeroll` |
| timestamps / years | `32:10` → `timestamp`, `2024` → `year` |
| negation marking (Tier 2) | `hindi naman magaling` → `hindi NEG_magaling` |
| *pero / kaso* → *but* (Tier 1) | lets VADER's "but" rule work |

## Notes

- **API keys** are read from environment variables / `.env`; nothing secret is stored in the code.
- **Curse words are masked** (`g**o`, `t*****a`) in everything shown: notebook outputs, charts, the word cloud
  and the docs. The data and the models still use the original words, because they carry much of the sentiment.
- `data/raw/` keeps the commenters' display names (`author` column). They are never used or shown; drop the
  column before making the repo public if you prefer.
