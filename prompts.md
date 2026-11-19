# Prompts (Tier 3 - LLM zero-shot)

Model: `claude-haiku-5-5` (Claude Haiku 5.5), `max_tokens=10`, thinking turned off (`thinking={"type": "disabled"}`;
if a model refuses that, `notebooks/src/llm.py` retries without it). Temperature is left at the default.

Each call sends the prompt below as the **system** message and the comment as the user message:

```
Comment: <text_clean>
```

The reply is lower-cased and matched to `positive` / `neutral` / `negative`; anything else counts as
`neutral` and is reported as `unparsed` by `ClaudeClassifier.stats()`.

The prompt is **zero-shot**: it contains no labelled examples from the dataset.

## v2_taglish

Explains the domain (FlipTop, the two teams, Tagalog / Taglish) and the label rules from
`data/labels/LABELING_GUIDE.md`.

```
You classify the sentiment of YouTube comments on a FlipTop rap battle video (Team LA = Loonie & Abra vs Team SS = Shehyee & Smugglaz). Comments are in Tagalog, English or Taglish (mixed), with slang, misspellings and profanity.
Labels:
- positive: praise, hype, enjoyment, support for a battler or team, nostalgic appreciation ("classic", "solid", "lupet", "galing", "idol"). A curse word used as hype together with praise ("... ang lupet") is positive.
- negative: insults or curses aimed at battlers, fans or judges; criticism (corny, pikon, mayabang, luto/rigged, walang kwenta); mocking a battler; angry disagreement with the result; taunting the other side's fans.
- neutral: questions, dates or "who's still watching", timestamps, like-bait, quoted bars with no opinion, factual remarks, requests, laughter only, or mixed with no clear side.
With pero / kaso / but, weigh the part after it. Sarcasm counts as its real meaning.
Reply with exactly one word: positive, neutral, or negative.
```

## Caching

Every answer is stored in `results/llm_cache.jsonl` with the raw reply, token counts and latency, keyed by
(model, prompt version, text). Re-running uses the cache; change the prompt version name if you edit a
prompt so old answers are not reused.
