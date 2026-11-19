"""Everything to do with curse words, kept in one file.

1. Masking whenever text is SHOWN (tables, printouts, charts, word clouds).
2. The curse-word spellings, phrases and VADER scores the models need (bottom of the file).


The data files and the models still use the original words, because profanity carries a lot of the
sentiment in this comment section. Only what a person reads is masked:  g**o, b**o, t*****a.

The word lists are stored spelled backwards so the source file itself stays clean;
`_unmask` turns them the right way round at import time.
"""
import re

# Each entry is stored spelled backwards and reversed at import time, so the curse words are
# not spelled out in the code. (Read-only lists; add to them if you find more.)
_EXACT_ENC = [
    "ogag", "ugag", "agag", "gnogag", "obob", "ubob", "obub", "ubub", "gnobob", "agnat", "gnagnat", "lolu", "lulu", "lolo", "lapuk", "atup", "atop", "ahcup", "ahctup", "atupma", "atopma", "fma", "tihs", "kcuf", "uykap", "itit", "etit", "setit", "kepkep", "ikup", "ekup", "domat", "apuhc", "apuhcuhc", "niapuhc", "totnak", "anegnat", "angnat", "anignt", "anigntp", "omangnat", "odatnarat", "etepatup", "sigartup",
]
_PREFIX_ENC = [
    "anignat", "anignatup", "anignatop", "gnatup", "gnatop", "kcuf", "tihs", "datnarat", "tutnak", "totnak", "anignik", "anegnat", "mangnat",
]


def _unmask(enc: str) -> str:
    return enc[::-1]          # entries are stored reversed


EXACT = {_unmask(w) for w in _EXACT_ENC}
PREFIXES = tuple(_unmask(w) for w in _PREFIX_ENC)

_WORD_RE = re.compile(r"[A-Za-zÑñ_]+")
# two-word forms where each word alone is harmless ("ina" = mother): "<prefix> ina"
_PHRASE_RE = re.compile(r"\b(" + "|".join(_unmask(w) for w in ["gnat", "gnatup", "gnatop", "gnik"]) +
                        r")(\s+)(ina(?:ng)?|na)\b", re.I)
_ELONG_RE = re.compile(r"([a-zñ])\1{2,}")


def is_profane(word: str) -> bool:
    w = _ELONG_RE.sub(r"\1", word.lower())
    return w in EXACT or w.startswith(PREFIXES)


def mask_word(word: str) -> str:
    """g***o style: keep the first and last letter, star the middle."""
    if len(word) <= 2:
        return word[0] + "*" * (len(word) - 1)
    return word[0] + "*" * (len(word) - 2) + word[-1]


def mask(text) -> str:
    """Mask every profane word in a string (anything that is not a string is returned unchanged)."""
    if not isinstance(text, str):
        return text
    text = _PHRASE_RE.sub(lambda m: mask_word(m.group(1)) + m.group(2) + mask_word(m.group(3)), text)
    return _WORD_RE.sub(lambda m: mask_word(m.group(0)) if is_profane(m.group(0)) else m.group(0), text)


def mask_frame(df, cols=None):
    """Copy of a DataFrame with text columns masked (all object columns by default) - for display."""
    d = df.copy()
    cols = cols or [c for c in d.columns if d[c].map(lambda v: isinstance(v, str)).any()]
    for c in cols:
        d[c] = d[c].map(mask)
    return d


# ------------------------------------------------------------------ data used by the models
# Spelling variants -> one standard spelling (merged into cleaning.SPELLING). Stored reversed.
_SPELLING_ENC = [
    ("anignt", "anignat"),
    ("angnat", "anignat"),
    ("anignatup", "anignat"),
    ("anigntp", "anignat"),
    ("anignatop", "anignat"),
    ("omanignatup", "anignat"),
    ("omanignat", "anignat"),
    ("oynangnat", "anignat"),
    ("oynanignat", "anignat"),
    ("gnanignat", "anignat"),
    ("anegnat", "anignat"),
    ("atop", "atup"),
    ("ahcup", "atup"),
    ("ahctup", "atup"),
    ("ketup", "atup"),
    ("sigartup", "atup"),
    ("ugag", "ogag"),
    ("agag", "ogag"),
    ("ubub", "obob"),
    ("ubob", "obob"),
    ("obub", "obob"),
    ("lolo", "lolu"),
    ("lulu", "lolu"),
    ("pma", "atupma"),
    ("atopma", "atupma"),
]
PROFANE_SPELLING = {_unmask(k): _unmask(v) for k, v in _SPELLING_ENC}

# Multi-word forms -> one token (merged into cleaning.PHRASES, before the other phrases).
_P, _PG, _T, _TA = _unmask("atup"), _unmask("gnatup"), _unmask("gnat"), _unmask("anignat")
PROFANE_PHRASES = [
    (r"\b" + _P + r"\s*ng\s*ina(?:\s*(?:mo|nyo|niyo))?\b", _TA),
    (r"\b" + _PG + r"\s*ina(?:\s*(?:mo|nyo|niyo))?\b", _TA),
    (r"\b" + _T + r"\s*ina(?:\s*(?:mo|nyo|niyo))?\b", _TA),
    (r"\b" + _T + r"\s*inang\b", _TA),
]

# VADER valences for insults / curse words (merged into taglish_lexicon.TAGLISH_LEXICON).
PROFANE_LEXICON = {_unmask(k): v for k, v in {"obob": -2.8, "agnat": -2.8, "lolu": -2.6, "ogag": -2.4, "lapuk": -2.6, "odatnarat": -2.6, "anignat": -1.0, "atup": -1.0, "atupma": -1.0, "etepatup": -0.8}.items()}


_STARS_IN_WORD = re.compile(r"(?<=[A-Za-zÑñ])(\*+)(?=[A-Za-zÑñ])")


def mask_md(text: str) -> str:
    """mask() for Markdown: stars are escaped (g\\*\\*o) so they show as stars instead of bold/italic."""
    out = _STARS_IN_WORD.sub(lambda m: "\\*" * len(m.group(1)), mask(text))
    # inside `code spans` a backslash would be shown literally, so undo the escaping there
    return re.sub(r"`[^`\n]*`", lambda m: m.group(0).replace("\\*", "*"), out)
