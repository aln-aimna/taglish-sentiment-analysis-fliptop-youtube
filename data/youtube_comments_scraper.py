"""
Get ALL comments (and replies) from one YouTube video
Uses the official YouTube Data API v3 — Python 3.14

SETUP (one time):
  1. Go to https://console.cloud.google.com/
  2. Create a project -> "APIs & Services" -> "Library"
     -> search "YouTube Data API v3" -> Enable
  3. "APIs & Services" -> "Credentials" -> "Create credentials" -> "API key"
  4. pip install requests pandas

Run:
  python data/youtube_comments_scraper.py   (from the repo root)

The API key is read from YOUTUBE_API_KEY (environment variable or the repo's .env file) if set,
otherwise you'll be prompted for it (hidden input, never saved in the code).
"""

import os
import time
from getpass import getpass
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests

# ---------------- Settings ----------------
VIDEO_URL       = "https://www.youtube.com/watch?v=gyIxD5K4lUo&t=590s"
INCLUDE_REPLIES = True      # also fetch every reply under each comment
ORDER           = "time"    # "time" (newest first) or "relevance" (top comments)
MAX_ROWS        = None      # e.g. 1000 for a quick test; None = fetch everything
# ------------------------------------------

BASE_URL = "https://www.googleapis.com/youtube/v3"
try:                                    # read YOUTUBE_API_KEY from the repo's .env file, if there is one
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass
API_KEY = os.environ.get("YOUTUBE_API_KEY") or getpass("Paste your YouTube API key: ")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")   # data/raw/

def get_video_id(url):
    """Pull the video ID out of any common YouTube URL format."""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.path.startswith(("/shorts/", "/embed/")):
        return parsed.path.split("/")[2]
    return parse_qs(parsed.query)["v"][0]


def call_api(endpoint, params):
    """One GET request to the API, with a clear error message if it fails."""
    params = {**params, "key": API_KEY}
    resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    if resp.status_code != 200:
        error = resp.json().get("error", {})
        raise RuntimeError(f"API error {resp.status_code}: {error.get('message')}")
    return resp.json()


def to_row(comment, video_id, parent_id=None):
    """Turn one API comment object into a flat dict (one table row)."""
    s = comment["snippet"]
    return {
        "comment_id":   comment["id"],
        "parent_id":    parent_id,                 # None = top-level comment
        "is_reply":     parent_id is not None,
        "author":       s.get("authorDisplayName"),
        "text":         s.get("textOriginal"),
        "like_count":   s.get("likeCount", 0),
        "published_at": s.get("publishedAt"),
        "updated_at":   s.get("updatedAt"),
        "video_id":     video_id,
    }


def get_all_replies(parent_id, video_id):
    """Fetch every reply to one top-level comment (handles paging)."""
    rows, page_token = [], None
    while True:
        data = call_api("comments", {
            "part": "snippet",
            "parentId": parent_id,
            "maxResults": 100,
            "textFormat": "plainText",
            **({"pageToken": page_token} if page_token else {}),
        })
        rows += [to_row(c, video_id, parent_id) for c in data.get("items", [])]
        page_token = data.get("nextPageToken")
        if not page_token:
            return rows


def get_all_comments(video_id, include_replies=True, order="time", max_rows=None):
    """Fetch top-level comments page by page; stop early if max_rows is reached."""
    rows, page_token, page = [], None, 0
    while True:
        page += 1
        data = call_api("commentThreads", {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": min(100, max_rows) if max_rows else 100,
            "order": order,
            "textFormat": "plainText",
            **({"pageToken": page_token} if page_token else {}),
        })

        for thread in data.get("items", []):
            top = thread["snippet"]["topLevelComment"]
            rows.append(to_row(top, video_id))

            if include_replies and thread["snippet"].get("totalReplyCount", 0) > 0:
                rows += get_all_replies(top["id"], video_id)

            if max_rows and len(rows) >= max_rows:      # stop early for testing
                print(f"Reached MAX_ROWS = {max_rows}. Stopping.")
                return rows[:max_rows]

        print(f"Page {page}: total rows so far = {len(rows)}")
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.2)

    return rows


def main():
    video_id = get_video_id(VIDEO_URL)
    print(f"Video ID: {video_id}")

    rows = get_all_comments(video_id, INCLUDE_REPLIES, ORDER, MAX_ROWS)

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"])

    os.makedirs(RAW_DIR, exist_ok=True)
    out_file = os.path.join(RAW_DIR, f"youtube_comments_{video_id}.csv")
    df.to_csv(out_file, index=False, encoding="utf-8-sig")   # utf-8-sig keeps emojis readable in Excel

    print(f"\nSaved {len(df)} rows -> {out_file}")
    print(f"Top-level comments: {(~df['is_reply']).sum()} | Replies: {df['is_reply'].sum()}")
    print(df.head())


if __name__ == "__main__":
    main()