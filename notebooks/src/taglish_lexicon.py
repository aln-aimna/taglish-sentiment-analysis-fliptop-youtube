"""Taglish additions for VADER (Tier 1).

VADER scores a sentence by looking up each word in a hand-made lexicon (valence -4 .. +4),
then applies rules: negation flips/dampens, boosters ("very") amplify, "but" shifts weight to
the second clause, ALL CAPS and "!!!" add emphasis. Stock VADER only knows English, so almost
every Tagalog comment scores 0 = neutral. This module adds:

  * TAGLISH_LEXICON  - Tagalog / FlipTop slang words with a valence
  * TAGALOG_NEGATORS - words that should trigger VADER's negation rule
  * TAGALOG_BOOSTERS - intensifiers ("sobrang", "napaka") and dampeners ("medyo")

Valences follow VADER's scale (e.g. English "good" = 1.9, "great" = 3.1, "bad" = -2.5,
"stupid" = -2.4). Words are in the spelling produced by `cleaning.for_vader()` (the SPELLING
map there turns "lufet" / "lupit" into "lupet", etc.).

Curse words are a judgement call: in this corpus the most common Tagalog curse words appear both as
insults (aimed at a battler) and as hype (followed by "ang lupet"). They get a small negative valence
(-1.0) so a praise word in the same comment can still win; direct insults get -2.4 to -2.8. These
entries live in notebooks/src/profanity.py (PROFANE_LEXICON) so this file stays free of curse words.
Slurs are deliberately left out of the lexicon.
"""
from .profanity import PROFANE_LEXICON

TAGLISH_LEXICON = {
    # ---- praise / hype
    "galing": 2.5, "magaling": 2.5, "pinakamagaling": 3.0, "husay": 2.5, "mahusay": 2.5,
    "lupet": 2.8, "malupet": 2.8, "pinakamalupet": 3.2, "lulupet": 2.8,
    "bangis": 2.6, "mabangis": 2.6, "astig": 2.5, "angas": 0.5,  # angas = swagger, mostly neutral
    "solid": 2.2, "classic": 1.8, "legend": 2.5, "legendary": 2.8, "goat": 2.5, "idol": 2.0,
    "lodi": 2.0, "petmalu": 2.5, "ganda": 2.5, "maganda": 2.5, "pinakamaganda": 3.0,
    "gwapo": 1.8, "pogi": 1.8, "panalo": 2.0, "nanalo": 1.0, "malakas": 1.8, "lakas": 1.8,
    "pinakamalakas": 2.5, "bigat": 1.5, "mabigat": 1.5, "tindi": 1.8, "matindi": 1.8,
    "lalim": 1.8, "malalim": 1.8, "sarap": 2.0, "nakakamiss": 1.5, "kamiss": 1.5,
    "salamat": 1.8, "congrats": 2.5, "tagos": 1.5, "halimaw": 2.0, "matalino": 2.0,
    "nakakatawa": 1.2, "laughtrip": 1.8, "deserve": 1.2, "deserving": 1.2, "sakalam": 2.2,
    "nakakasawa": -1.0,  # "di nakakasawa" -> negation flips it to positive
    "kupas": -1.0,       # "wala paring kupas" -> negation flips it to positive
    # ---- criticism / insults
    "pangit": -2.5, "ampangit": -2.5, "corny": -1.8, "yabang": -2.0, "mayabang": -2.0,
    "hambog": -2.0, "pikon": -1.8, "pikunin": -1.8, "iyakin": -1.8, "epal": -2.0,
    "luto": -2.2, "daya": -2.5, "madaya": -2.5, "dinaya": -2.5, "bayad": -1.0, "benta": -1.5,
    "talo": -1.5, "natalo": -1.0, "talunan": -2.2, "choke": -1.8, "nagchoke": -1.8,
    "pabuhat": -2.0, "buhat": -0.8, "walang_kwenta": -2.8, "kwenta": 1.0,
    "engot": -2.2, "abnoy": -2.2, "abno": -2.2, "bugok": -2.4, "unggoy": -2.0, "hayop": -2.2,
    "peste": -2.2, "bwisit": -2.2, "nakakabwisit": -2.5, "badtrip": -2.0, "nakakainis": -2.2,
    "kainis": -2.0, "nakakairita": -2.2, "kadiri": -2.5, "nakakadiri": -2.5, "nakakahiya": -2.0,
    "bastos": -2.2, "plastik": -2.0, "insecure": -1.5, "inggit": -1.5, "dismayado": -2.0,
    "kadisappoint": -2.2, "sayang": -1.2, "boring": -1.8, "baho": -2.0, "mabaho": -2.0,
    "tae": -1.5, "duwag": -2.2, "panis": -1.2, "lampaso": -1.2,
    "lintek": -0.8,
    # insults and curse words are added from notebooks/src/profanity.py (PROFANE_LEXICON), see docstring
}
TAGLISH_LEXICON.update(PROFANE_LEXICON)

# VADER checks the word up to 3 positions before a sentiment word for a negator.
TAGALOG_NEGATORS = ["hindi", "wala", "walang", "huwag", "ayaw", "di", "nde", "ndi"]

# Positive numbers amplify, negative numbers dampen (VADER uses +-0.293 for English).
TAGALOG_BOOSTERS = {
    "sobrang": 0.293, "sobra": 0.293, "napaka": 0.293, "napakaganda": 0.293, "grabe": 0.293,
    "talaga": 0.15, "talagang": 0.15, "super": 0.293, "pinaka": 0.293,
    "medyo": -0.293, "konti": -0.293, "parang": -0.15,
}


def make_vader(taglish: bool = True):
    """Return a VADER SentimentIntensityAnalyzer, optionally with the Taglish additions.

    Works with either package:
      * `vaderSentiment` (pip install vaderSentiment) - preferred, it also reads emoji
      * NLTK (pip install nltk, then nltk.download("vader_lexicon")) - what classic_nlp used
    Negators / boosters are stored at module or class level in both packages, so adding
    them affects every analyzer in this Python process. Score stock VADER first, then build
    the Taglish one (notebook 02 does it in that order).
    """
    try:
        import vaderSentiment.vaderSentiment as vs
        sia = vs.SentimentIntensityAnalyzer()
        negate, boosters = vs.NEGATE, vs.BOOSTER_DICT
    except ImportError:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        negate, boosters = sia.constants.NEGATE, sia.constants.BOOSTER_DICT

    if taglish:
        sia.lexicon.update(TAGLISH_LEXICON)
        for w in TAGALOG_NEGATORS:
            if w in negate:
                continue
            negate.add(w) if isinstance(negate, set) else negate.append(w)
        boosters.update(TAGALOG_BOOSTERS)
    return sia


def vader_label(compound: float, pos_th: float = 0.05, neg_th: float = -0.05) -> str:
    """Standard VADER thresholds: >= 0.05 positive, <= -0.05 negative, else neutral."""
    if compound >= pos_th:
        return "positive"
    if compound <= neg_th:
        return "negative"
    return "neutral"
