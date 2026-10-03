"""Small, dependency-free text helpers for the curiosity organism.

Everything here is deterministic so that the parts of the organism that must
not be left to the language model (retrieval, deduplication, quote
verification) stay inspectable and testable.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each either even ever every few for from
    further had has have having he her here hers herself him himself his how i if in into is it its itself
    just like many may me might more most much must my myself neither no nor not now of off on once one only
    or other our ours ourselves out over own same shall she should so some such than that the their theirs
    them themselves then there these they this those thus through to too under until up upon us very was we
    were what when where whether which while who whom whose why will with would yet you your yours yourself
    yourselves thee thou thy hath doth unto shall said say says saying thing things
    """.split()
)

# Older translations spell several key words differently from modern questions
# ("enquire" in Jowett's Plato, "sceptic" in Hume). Normalise before stemming.
_SPELLING = (
    ("enquir", "inquir"),
    ("sceptic", "skeptic"),
    ("judgement", "judgment"),
    ("honour", "honor"),
    ("colour", "color"),
    ("behaviour", "behavior"),
)

_SUFFIXES = (
    # (suffix, replacement, minimum stem length that must remain)
    ("ities", "", 4),
    ("ity", "", 4),
    ("ousness", "os", 3),
    ("ous", "os", 3),
    ("ness", "", 4),
    ("ments", "", 4),
    ("ment", "", 4),
    ("ings", "", 3),
    ("ing", "", 3),
    ("ical", "", 4),
    ("ally", "", 4),
    ("edly", "", 4),
    ("ies", "y", 3),
    ("ed", "", 4),
    ("ly", "", 4),
    ("s", "", 3),
)

_WORD_RE = re.compile(r"[a-z]+")


def fold(text: str) -> str:
    """Lowercase and strip accents ("René" -> "rene")."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def stem(word: str) -> str:
    for old, new in _SPELLING:
        if word.startswith(old):
            word = new + word[len(old):]
            break
    for suffix, replacement, min_len in _SUFFIXES:
        if suffix == "s" and word.endswith(("ss", "us", "is")):
            continue  # "process", "thus", "basis" are not plurals
        if word.endswith(suffix) and len(word) - len(suffix) >= min_len:
            word = word[: -len(suffix)] + replacement
            break
    if len(word) >= 5 and word[-1] in "ey":
        word = word[:-1]
    return word


def tokens(text: str) -> list[str]:
    """Content-word stems, in order, with stopwords removed."""
    return [stem(w) for w in _WORD_RE.findall(fold(text)) if len(w) > 2 and w not in STOPWORDS]


@lru_cache(maxsize=8192)
def token_set(text: str) -> frozenset[str]:
    """Cached: drive readings compare every open question with every belief."""
    return frozenset(tokens(text))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap(a: str, b: str) -> float:
    """Share of the smaller text's content words that also occur in the other text."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def coverage(query: str, text: str) -> float:
    """Share of the query's content words that occur in ``text``."""
    sq = token_set(query)
    if not sq:
        return 0.0
    return len(sq & token_set(text)) / len(sq)


def _normalize_for_quote(text: str) -> str:
    text = fold(text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def verify_quote(quote: str, source_text: str, *, min_words: int = 4, fuzzy: float = 0.85, max_words: int = 60, keep_words: int = 40) -> str | None:
    """Return the quote if it really occurs in ``source_text``, else ``None``.

    Small models often paraphrase while claiming to quote. A claim of
    confirmation or contradiction only counts as evidence if its quote can be
    found in the passage that was actually read. Matching ignores case,
    punctuation and whitespace, accepts quotes split with an ellipsis, and
    tolerates small copying slips (``fuzzy`` = share of the quote that must
    match contiguously). Long quotes are shortened to ``keep_words``.
    """
    if not quote or not source_text:
        return None
    parts = [p for p in re.split(r"\.\.\.|…", quote) if p.strip()]
    if not parts:
        return None
    haystack = _normalize_for_quote(source_text)
    verified_parts: list[str] = []
    for part in parts:
        needle = _normalize_for_quote(part)
        words = needle.split()
        if len(words) < min_words:
            if len(parts) == 1:
                return None
            continue
        if len(words) > max_words:
            needle = " ".join(words[:max_words])
        if needle in haystack:
            verified_parts.append(part.strip())
            continue
        matcher = SequenceMatcher(None, haystack, needle, autojunk=False)
        match = matcher.find_longest_match(0, len(haystack), 0, len(needle))
        if match.size >= fuzzy * len(needle):
            verified_parts.append(part.strip())
            continue
        return None
    if not verified_parts:
        return None
    joined = " ... ".join(verified_parts)
    words = joined.split()
    return joined if len(words) <= keep_words else " ".join(words[:keep_words]) + " ..."


# A prediction hedged with "may" or "might" can never be wrong: whatever the texts
# say, it survives. Doubt belongs in the probability, not in the wording.
_HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|perhaps|potentially|presumably|probably|likely|"
    r"it is possible|some (?:texts|authors|thinkers|sources))\b",
    re.IGNORECASE,
)

# Words that let an answer avoid committing to anything ("a complex, multifaceted
# interplay of various factors"). One is a style; several together are a non-answer.
_VAGUE_RE = re.compile(
    r"\b(complex|complexity|multifaceted|multi-faceted|dynamic|interplay|nuanced|holistic|"
    r"intricate|various factors|multiple factors|many factors|range of factors|variety of factors|"
    r"it depends|depends on (?:the )?(?:context|individual)|individual differences|"
    r"cultural (?:differences|variations|contexts))\b",
    re.IGNORECASE,
)


def is_hedged(text: str) -> bool:
    return bool(_HEDGE_RE.search(text or ""))


def vagueness(text: str) -> int:
    """How many distinct non-committal phrases the text leans on."""
    return len({m.group(0).lower() for m in _VAGUE_RE.finditer(text or "")})


def topic_relevance(text: str, topic_words: frozenset[str]) -> float:
    """How much of a question is about the topic, from 0 to 1 (lexical, as a fallback).

    The share of the question's content words that belong to the topic's
    vocabulary (its seed questions and key terms), doubled so that a question
    half made of topic words counts as fully on topic.
    """
    words = token_set(text)
    if not words or not topic_words:
        return 1.0
    return min(1.0, 2.0 * len(words & topic_words) / len(words))


def keywords_of(text: str, limit: int = 6) -> str:
    """The first distinct content words of a phrase: a search query, not an invented title."""
    seen: list[str] = []
    for word in _WORD_RE.findall(fold(text)):
        if len(word) > 2 and word not in STOPWORDS and word not in seen:
            seen.append(word)
        if len(seen) >= limit:
            break
    return " ".join(seen)


def clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    stop = max(cut.rfind(". "), cut.rfind("\n"))
    if stop > max_chars * 0.6:
        cut = cut[: stop + 1]
    elif " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + " [...]"


def one_line(text: str, max_chars: int = 160) -> str:
    return clip(re.sub(r"\s+", " ", text or ""), max_chars)
