"""Tier 3: zero-shot sentiment with the Claude API, with a disk cache and cost / latency logging.

Every answer is appended to results/llm_cache.jsonl, keyed by (model, prompt version, text).
Re-running the notebook therefore costs nothing and gives the same predictions; delete the
file (or change PROMPT_VERSION) to call the API again.

The API key is read from the ANTHROPIC_API_KEY environment variable (put it in a .env file,
which .gitignore keeps out of git). It is never written to the cache or to any output.
"""
import hashlib
import json
import os
import time

from . import config

MODEL = "claude-haiku-5-5"
# USD per million tokens for prompts under 100K tokens (platform.claude.com/docs pricing page,
# checked Nov 2026). Update these if you switch model or prices change.
PRICE_PER_MTOK = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}

PROMPTS = {
    # v2: still zero-shot (no labelled examples) but explains the domain and the label rules
    "v2_taglish": (
        "You classify the sentiment of YouTube comments on a FlipTop rap battle video "
        "(Team LA = Loonie & Abra vs Team SS = Shehyee & Smugglaz). Comments are in Tagalog, "
        "English or Taglish (mixed), with slang, misspellings and profanity.\n"
        "Labels:\n"
        "- positive: praise, hype, enjoyment, support for a battler or team, nostalgic "
        "appreciation (\"classic\", \"solid\", \"lupet\", \"galing\", \"idol\"). A curse word "
        "used as hype together with praise (\"... ang lupet\") is positive.\n"
        "- negative: insults or curses aimed at battlers, fans or judges; criticism (corny, "
        "pikon, mayabang, luto/rigged, walang kwenta); mocking a battler; angry disagreement "
        "with the result; taunting the other side's fans.\n"
        "- neutral: questions, dates or \"who's still watching\", timestamps, like-bait, quoted "
        "bars with no opinion, factual remarks, requests, laughter only, or mixed with no clear "
        "side.\n"
        "With pero / kaso / but, weigh the part after it. Sarcasm counts as its real meaning.\n"
        "Reply with exactly one word: positive, neutral, or negative."
    ),
}
PROMPT_VERSION = "v2_taglish"


def _key(model, prompt_version, text):
    return hashlib.sha1(f"{model}|{prompt_version}|{text}".encode("utf-8")).hexdigest()


def load_cache(path=config.LLM_CACHE) -> dict:
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
    return cache


def parse_label(answer: str) -> str:
    a = answer.strip().lower().strip(".!\"' ")
    for lab in config.LABELS:
        if a.startswith(lab):
            return lab
    return "neutral"           # fallback for anything unexpected (counted in 'unparsed')


class ClaudeClassifier:
    def __init__(self, model=MODEL, prompt_version=PROMPT_VERSION, cache_path=config.LLM_CACHE):
        self.model = model
        self.prompt_version = prompt_version
        self.system = PROMPTS[prompt_version]
        self.cache_path = cache_path
        self.cache = load_cache(cache_path)
        self._client = None
        self._thinking_off = True     # try thinking={"type": "disabled"} first
        self.api_calls = 0

    @property
    def client(self):
        if self._client is None:
            import anthropic
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set ANTHROPIC_API_KEY (e.g. in a .env file) before calling the API.")
            self._client = anthropic.Anthropic()
        return self._client

    def _call(self, text):
        import anthropic
        kwargs = dict(model=self.model, max_tokens=10, system=self.system,
                      messages=[{"role": "user", "content": f"Comment: {text}"}])
        if self._thinking_off:
            kwargs["thinking"] = {"type": "disabled"}
        for attempt in range(6):
            try:
                t0 = time.perf_counter()
                resp = self.client.messages.create(**kwargs)
                return resp, time.perf_counter() - t0
            except anthropic.BadRequestError as e:
                if "thinking" in kwargs and "thinking" in str(e).lower():
                    kwargs.pop("thinking")          # this model can't turn thinking off
                    kwargs["max_tokens"] = 400
                    self._thinking_off = False
                    continue
                raise
            except (anthropic.RateLimitError, anthropic.APIConnectionError,
                    anthropic.InternalServerError):
                time.sleep(2 ** attempt)
        raise RuntimeError("API kept failing after retries")

    def classify(self, text: str) -> dict:
        k = _key(self.model, self.prompt_version, text)
        if k in self.cache:
            return self.cache[k]
        resp, latency = self._call(text)
        self.api_calls += 1
        answer = next((b.text for b in resp.content if b.type == "text"), "")
        rec = {"key": k, "model": self.model, "prompt_version": self.prompt_version,
               "text": text, "raw_answer": answer.strip(), "label": parse_label(answer),
               "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
               "latency_s": round(latency, 4)}
        self.cache[k] = rec
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def predict(self, texts, verbose_every=50):
        out = []
        for i, t in enumerate(texts, 1):
            out.append(self.classify(t)["label"])
            if verbose_every and i % verbose_every == 0:
                print(f"{i}/{len(texts)} done ({self.api_calls} API calls so far)")
        return out

    def stats(self, texts) -> dict:
        """Average cost and latency per 1,000 predictions, from the cached token counts."""
        recs = [self.cache[_key(self.model, self.prompt_version, t)] for t in texts]
        price = PRICE_PER_MTOK.get(self.model, {"input": float("nan"), "output": float("nan")})
        n = len(recs)
        tin = sum(r["input_tokens"] for r in recs)
        tout = sum(r["output_tokens"] for r in recs)
        cost = tin / 1e6 * price["input"] + tout / 1e6 * price["output"]
        return {
            "n": n,
            "avg_input_tokens": tin / n,
            "avg_output_tokens": tout / n,
            "cost_per_1k_usd": cost / n * 1000,
            "sec_per_1k": sum(r["latency_s"] for r in recs) / n * 1000,   # sequential calls
            "unparsed": sum(r["raw_answer"].strip().lower().strip(".!\"' ") not in config.LABELS
                            for r in recs),
        }
