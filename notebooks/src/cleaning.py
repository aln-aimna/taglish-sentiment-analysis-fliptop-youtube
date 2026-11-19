"""
Text cleaning for Taglish (Tagalog + English) YouTube comments.

Three levels of cleaning are produced, because each model tier wants something different:

    text_clean  light clean, keeps case, punctuation and emoji      -> LLM (it reads context well)
    text_vader  text_clean + Taglish spelling fixes, "pero" -> "but" -> VADER + Taglish lexicon
    text_norm   lowercase, emoji -> words, laughter / slang unified -> NLTK + Naive Bayes

Only the standard library is used, so this runs anywhere.
"""
import re
import unicodedata

import pandas as pd

from .profanity import PROFANE_PHRASES, PROFANE_SPELLING

# ---------------------------------------------------------------- regex patterns
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
MENTION_RE = re.compile(r"@[\w.\-]+")                    # YouTube reply handles, e.g. @kaitakashi6587
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # 32:25, 1:02:10
LAUGH_RE = re.compile(r"\b(?:b?w?a?h+a+h+[ahjsk]*|a?h+a+h+a*[ahjsk]*|h+e+h+e+[he]*|ja+ja+[ja]*|"
                      r"lo+l+s?|lmf?ao+|rofl|xd+|wahaha+)\b", re.I)
EMOTICON_LAUGH_RE = re.compile(r"(?:=\)\)+|:-?d+\b|x-?d+\b|:-?\)+|\(\s*:)", re.I)
EMOTICON_SAD_RE = re.compile(r"(?::-?\(+|</3|:'\()")
EMOTICON_THUMB_RE = re.compile(r"\(y\)", re.I)
ELONGATION_RE = re.compile(r"([a-zñ])\1{2,}")          # grabeeee -> grabe, lupeeet -> lupet
WS_RE = re.compile(r"\s+")

# Multi-word expressions merged into one token BEFORE tokenising (order matters: longer first).
PHRASES = [
    (r"\bteam\s*s\.?\s*s\.?\b", "team_ss"),
    (r"\bteam\s*l\.?\s*a\.?\b", "team_la"),
    (r"\bs\.s\.?(?=\s|$)", "team_ss"),
    (r"\bl\.a\.?(?=\s|$)", "team_la"),
    (r"\bdos\s*por\s*dos\b", "dos_por_dos"),
    (r"\bstill\s+watching\b", "still_watching"),
    (r"\bthumbs\s+up\b", "thumbs_up"),
    (r"\bwalang\s+kwenta\b", "walang_kwenta"),
    (r"\bwala\s*ng\s+kwenta\b", "walang_kwenta"),
]
PHRASES = PROFANE_PHRASES + PHRASES          # curse-word phrases come first (notebooks/src/profanity.py)
PHRASES = [(re.compile(p, re.I), r) for p, r in PHRASES]

# Shortcuts / misspellings / slang -> one standard spelling. Keys are lowercase tokens.
SPELLING = {
    # function words
    "nmn": "naman", "nman": "naman", "namn": "naman", "nmen": "naman",
    "lng": "lang", "lamg": "lang", "nlng": "nalang", "nlang": "nalang",
    "tlga": "talaga", "tlg": "talaga", "talga": "talaga", "tlaga": "talaga", "tlagang": "talagang",
    "ksi": "kasi", "kc": "kasi", "kz": "kasi", "kse": "kasi", "kase": "kasi", "kasii": "kasi",
    "pra": "para", "wla": "wala", "wlang": "walang", "ung": "yung", "yng": "yung", "un": "yun",
    "yon": "yun", "cla": "sila", "sya": "siya", "cya": "siya", "xa": "siya", "c": "si",
    "kau": "kayo", "kyo": "kayo", "ngaun": "ngayon", "ngyon": "ngayon", "ngayun": "ngayon",
    "d2": "dito", "dto": "dito", "jan": "diyan", "dyan": "diyan", "dian": "diyan",
    "nyo": "niyo", "nio": "niyo", "nyu": "niyo", "nla": "nila", "anu": "ano", "dpat": "dapat",
    "sna": "sana", "hnd": "hindi", "hinde": "hindi", "indi": "hindi", "d": "hindi", "di": "hindi",
    "bkit": "bakit", "bat": "bakit", "kya": "kaya", "tpos": "tapos", "sbra": "sobra",
    "muka": "mukha", "pnalo": "panalo", "tlo": "talo", "pro": "pero", "prin": "parin",
    "padin": "parin", "pdin": "parin", "u": "you", "ur": "your", "pls": "please", "plz": "please",
    "wag": "huwag", "wgas": "huwag",
    # Filipino slang / evaluation words
    "steeg": "astig", "astg": "astig", "lufet": "lupet", "lupit": "lupet", "lpet": "lupet",
    "angass": "angas", "grbe": "grabe", "grabeh": "grabe", "grabi": "grabe",
    "korni": "corny", "korny": "corny", "kornie": "corny", "cornyy": "corny",
    "gnda": "ganda", "glng": "galing", "galeng": "galing", "panget": "pangit", "pngit": "pangit",
    "nice1": "nice", "matinde": "matindi", "truk": "true", "tru": "true", "gud": "good", "gd": "good", "sana2": "sana",
    # battler names (lots of spellings in the data)
    "sheyee": "shehyee", "shehyhee": "shehyee", "sheyhee": "shehyee", "shehye": "shehyee",
    "sheyhyee": "shehyee", "shehee": "shehyee", "sheyi": "shehyee", "shey": "shehyee",
    "smugg": "smugglaz", "smug": "smugglaz", "smuglaz": "smugglaz", "smuggs": "smugglaz",
    "smugglas": "smugglaz", "smuglas": "smugglaz",
    "lonie": "loonie", "looney": "loonie", "lony": "loonie", "lonnie": "loonie", "loney": "loonie",
}

SPELLING.update(PROFANE_SPELLING)            # curse-word spellings, kept in notebooks/src/profanity.py

# VADER only knows the English word "but" for contrast; map the Tagalog equivalents to it.
CONTRAST_TO_BUT = {"pero": "but", "kaso": "but", "kaya lang": "but"}

# Emoji -> word. Words were chosen so a lexicon or Naive Bayes can use them.
EMOJI_WORDS = {
    "😂": "haha", "🤣": "haha", "😆": "haha", "😹": "haha", "💀": "haha",
    "🔥": "fire", "💯": "solid", "❤": "love", "♥": "love", "💖": "love", "💕": "love",
    "😍": "love", "🥰": "love", "😘": "love", "💗": "love", "💓": "love", "💘": "love",
    "👏": "clap", "🙌": "clap", "👍": "good", "👌": "good", "💪": "strong", "🎉": "party",
    "😎": "cool", "😁": "smile", "😊": "smile", "😄": "smile", "😀": "smile", "😃": "smile",
    "☺": "smile", "🙂": "smile", "😇": "smile", "😅": "awkward", "🤘": "rock", "🤟": "rock",
    "🤙": "rock", "✌": "peace", "👊": "respect", "🐐": "goat", "🏆": "champion", "👑": "king",
    "👎": "bad", "😡": "angry", "😠": "angry", "🤬": "angry", "😤": "angry", "🖕": "middlefinger",
    "😢": "sad", "😭": "cry", "😞": "sad", "😔": "sad", "🤦": "facepalm", "🙄": "eyeroll",
    "😒": "eyeroll", "😴": "boring", "🤮": "disgust", "🤢": "disgust", "💩": "poop",
    "🤔": "hmm", "😮": "shock", "😱": "shock", "😏": "smirk", "😈": "devil",
}

# Tagalog / English function words used for the quick language tag and as stopwords.
TL_WORDS = {
    "ang", "ng", "sa", "mga", "si", "ni", "na", "at", "ko", "mo", "ka", "ako", "ikaw", "siya",
    "sila", "kayo", "nila", "niya", "naman", "lang", "nalang", "talaga", "pero", "kasi", "yung",
    "yun", "hindi", "wala", "walang", "ito", "yan", "dito", "diyan", "kung", "para", "pa", "din",
    "rin", "ba", "nga", "eh", "po", "kaya", "sino", "ano", "bakit", "mas", "may", "nag", "mag",
    "pag", "niyo", "nito", "kay", "dahil", "parin", "lahat", "dapat", "sana", "tapos", "ngayon",
    "kahit", "parang", "huwag", "ulit", "pala", "daw", "nung", "ay", "tong", "yang", "mong",
    "kang", "nang", "galing", "magaling", "laban", "panalo", "talo", "natalo", "nanalo", "ganda",
}
EN_WORDS = {
    "the", "is", "are", "was", "were", "this", "that", "and", "of", "to", "in", "it", "you",
    "i", "my", "me", "for", "on", "with", "still_watching", "watching", "best", "anyone", "who",
    "what", "but", "not", "so", "all", "from", "they", "he", "she", "his", "be", "have", "just",
    "good", "great", "never", "ever", "time", "one", "here", "will", "can", "dont", "if", "your",
    "we", "our", "their", "them", "there", "been", "only", "really", "very", "much", "more",
    "how", "why", "when", "where", "because", "than", "about", "after", "again", "love",
    "watch", "know", "think", "would", "should", "better", "worst", "greatest", "still",
}
# negators (Tagalog + English). Words after them get a NEG_ prefix for the Naive Bayes features.
NEGATORS = {"hindi", "huwag", "wala", "walang", "ayaw", "not", "no", "never", "dont", "cant",
            "wont", "isnt", "arent", "wasnt", "aint", "didnt", "doesnt", "nobody", "nothing"}
STOPWORDS = (TL_WORDS | EN_WORDS) - NEGATORS - {
    # keep words that carry sentiment even though they are common
    "galing", "magaling", "panalo", "talo", "natalo", "nanalo", "ganda", "best", "good", "great",
    "worst", "greatest", "love", "better", "sana", "dapat", "but", "pero", "still_watching",
}

TOKEN_RE = re.compile(r"[a-z0-9ñ_]+")
YEAR_RE = re.compile(r"^(?:19|20)\d\d$")


# ---------------------------------------------------------------- cleaning steps
def basic_clean(text: str) -> str:
    """Light clean: unicode normalise, drop URLs / @mentions, collapse whitespace.
    Case, punctuation and emoji are kept (VADER and the LLM both use them)."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)      # fancy unicode fonts -> plain letters
    t = t.replace("‍", "").replace("️", "")   # emoji joiners / variation selectors
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


def _merge_phrases(t: str) -> str:
    for pattern, repl in PHRASES:
        t = pattern.sub(repl, t)
    return t


def for_vader(text: str) -> str:
    """text_clean + Taglish spelling fixes, keeping capitals (VADER reads ALL CAPS as emphasis).
    Tagalog contrast words become "but" so VADER's but-rule fires."""
    t = basic_clean(text)
    t = _merge_phrases(t)
    t = LAUGH_RE.sub("haha", t)

    def fix(m):
        word = m.group(0)
        if len(word) == 1:                       # leave "D" in ":D", "c", "u" alone
            return word
        low = ELONGATION_RE.sub(r"\1", word.lower())     # grabeeee -> grabe
        new = CONTRAST_TO_BUT.get(low, SPELLING.get(low, low))
        new = CONTRAST_TO_BUT.get(new, new)
        if new == low and low == word.lower():
            return word                          # unchanged: keep original casing
        return new.upper() if word.isupper() else new

    # whole words only: not glued to digits ("d2") or emoticon eyes (":D")
    return re.sub(r"(?<![\w:;=])[A-Za-zÑñ_]+(?![\w])", fix, t)


def normalize(text: str) -> str:
    """Heavy normalisation for bag-of-words models."""
    t = basic_clean(text).lower()
    for emo, word in EMOJI_WORDS.items():
        if emo in t:
            t = t.replace(emo, f" {word} ")
    t = EMOTICON_LAUGH_RE.sub(" haha ", t)
    t = EMOTICON_SAD_RE.sub(" sad ", t)
    t = EMOTICON_THUMB_RE.sub(" good ", t)
    t = TIMESTAMP_RE.sub(" timestamp ", t)
    t = _merge_phrases(t)
    t = LAUGH_RE.sub(" haha ", t)
    t = ELONGATION_RE.sub(r"\1", t)
    t = t.replace("’", "").replace("'", "")       # don't -> dont
    tokens = [SPELLING.get(tok, tok) for tok in TOKEN_RE.findall(t)]
    tokens = ["year" if YEAR_RE.match(tok) else tok for tok in tokens]   # "anyone from 2024?"
    tokens = [tok for tok in tokens if not tok.isdigit()]
    tokens = [LAUGH_RE.sub("haha", tok) for tok in tokens]
    return " ".join(tokens)


def tokenize(norm_text: str, drop_stopwords: bool = True, mark_negation: bool = True) -> list:
    """Split normalised text, add NEG_ to the 2 words after a negator, drop stopwords."""
    out, neg_left = [], 0
    for tok in norm_text.split():
        if tok in NEGATORS:
            out.append(tok)
            neg_left = 2
            continue
        if tok in STOPWORDS:
            if not drop_stopwords:
                out.append(tok)                       # kept, but never NEG_-marked ...
            continue                                  # ... and it doesn't use up the window:
                                                      # "hindi naman talaga magaling" -> NEG_magaling
        out.append(f"NEG_{tok}" if (mark_negation and neg_left > 0) else tok)
        neg_left = max(0, neg_left - 1)
    return out


def detect_language(norm_text: str) -> str:
    """Rough tag: tagalog / english / taglish / unknown, from function-word counts."""
    toks = norm_text.split()
    tl = sum(t in TL_WORDS for t in toks)
    en = sum(t in EN_WORDS for t in toks)
    if tl + en == 0:
        return "unknown"
    share_tl = tl / (tl + en)
    if share_tl >= 0.8:
        return "tagalog"
    if share_tl <= 0.2:
        return "english"
    return "taglish"


def has_content(text_clean: str) -> bool:
    """True if the comment has at least 2 letters, or one emoji we know how to read."""
    if len(re.findall(r"[A-Za-zÑñ]", text_clean)) >= 2:
        return True
    return any(e in text_clean for e in EMOJI_WORDS)


# ---------------------------------------------------------------- whole dataframe
def clean_dataframe(df: pd.DataFrame, max_chars: int = 400, min_tokens: int = 1):
    """Clean the raw scraper output. Returns (clean_df, log) where log counts every step."""
    log = {"raw_rows": len(df)}
    d = df.copy()
    d = d[d["text"].notna() & (d["text"].astype(str).str.strip() != "")]
    log["after_drop_empty"] = len(d)

    d["text_clean"] = d["text"].map(basic_clean)
    d = d[d["text_clean"].map(has_content)]
    log["after_drop_no_content"] = len(d)

    d["text_norm"] = d["text"].map(normalize)
    d["text_vader"] = d["text"].map(for_vader)
    d["n_tokens"] = d["text_norm"].str.split().str.len()
    d["n_chars"] = d["text_clean"].str.len()

    # duplicates: same normalised text (spam like "first", "haha", copy-pasted bars). Keep most-liked.
    d = d.sort_values("like_count", ascending=False)
    d = d.drop_duplicates(subset="text_norm", keep="first")
    log["after_drop_duplicates"] = len(d)

    d["lang"] = d["text_norm"].map(detect_language)
    d["published_at"] = pd.to_datetime(d["published_at"], utc=True)
    d["year"] = d["published_at"].dt.year
    # rows suitable for labelling / modelling: not empty after normalising, not a wall of lyrics
    d["in_model_pool"] = (d["n_tokens"] >= min_tokens) & (d["n_chars"] <= max_chars)
    log["model_pool"] = int(d["in_model_pool"].sum())

    keep = ["comment_id", "parent_id", "is_reply", "text", "text_clean", "text_vader", "text_norm",
            "n_tokens", "n_chars", "lang", "like_count", "published_at", "year", "in_model_pool"]
    d = d[keep].sort_values("published_at").reset_index(drop=True)
    return d, log
