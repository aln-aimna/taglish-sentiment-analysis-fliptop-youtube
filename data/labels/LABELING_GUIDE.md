# Labeling guide

Files in this folder:

| file | what it is |
|---|---|
| `label_pool.csv` | 1,500 comments sampled at random (seed 42) from the cleaned model pool. Rows `label_id` 0-999 are labelled; 1000-1499 are spare rows if you want to label more. |
| `labeled_comments.csv` | the 1,000 labelled comments (+ the cleaned text columns). `annotator` says who labelled it, `reviewed` is for a human check. |
| `hard_cases.csv` | 30 hand-written hard sentences with a `phenomenon` tag. |

## Who labelled the data

First-pass labels were produced by **Claude (Opus) acting as annotator**, one comment at a time, following
the rules below. They have **not yet been checked by a person**. Please review at least the 300 test rows
(`data/processed/test_300.csv` lists their `label_id`s): fix the `label`, set `reviewed` to `True`, save
`labeled_comments.csv`, then re-run notebooks 01 → 05. The split is by `label_id`, so it will not change.

Why this matters: Tier 3 is also a Claude model. Unreviewed labels make the LLM look better than it is.

## The question for every comment

> What is the writer's attitude in this comment - towards the battle, a battler, a team, the judges or other
> commenters?

## positive

Praise, hype, enjoyment, support, nostalgic appreciation.

- "galing ni smugg", "lupet", "solid", "classic", "GOAT", "idol", "bangis", "the best dos por dos"
- support without insults: "Team LA pa rin!", "idol smugglaz at shehyee"
- profanity used as hype: "t\*\*g i\*a ang lupet", "p\*\*\*\*\*\*\*a ang bangis"
- nostalgia with appreciation: "2023 still watching, solid pa rin", "nakakamiss yung ganitong battle"

## negative

Insults, criticism, mocking, anger, taunting.

- insults or curses aimed at a battler, judge, fan or commenter: "b\*\*o", "g\*\*o ka", "t\*\*g i\*a mo"
- criticism: corny, pikon, mayabang, luto / rigged, madaya, walang kwenta, pabuhat
- mocking a battler: "buhat na buhat ni smug si sheyee", "iyak mga LA fans"
- angry disagreement with the result: "luto 'to, dapat LA nanalo"
- fan-war comments that are mostly insults, **even if they also cheer for one side**

## neutral

No clear attitude.

- questions: "title ng intro song?", "kelan upload ng finals?"
- dates / presence: "2024 still watching?", "anyone from 2025?", "up July 2022"
- timestamps and quoted bars with no opinion: "7:35 'di nyo ko kahilerang tatlo'"
- like-bait and polls: "like kung team SS"
- factual remarks, requests, greetings, laughter only ("HAHAHA")
- truly mixed comments with no side winning

## Tie-breakers

- **pero / kaso / but** → weigh the part after it. "Ganda ng laban pero luto" = negative.
- **Sarcasm** counts as its real meaning. "Ang galing mag-choke" = negative.
- Praise for one team + insults at the other: label by whichever dominates; if insults dominate → negative.
- Not sure between neutral and a side → neutral.
