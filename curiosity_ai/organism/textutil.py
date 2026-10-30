"""Small, dependency-free text helpers for the curiosity organism.

Everything here is deterministic so that the parts of the organism that must
not be left to the language model (retrieval, deduplication, quote
verification) stay inspectable and testable.
"""
from __future__ import annotations

import math
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
        if _in_order(needle.split(), haystack.split()):
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
    r"it is possible|some (?:texts|authors|thinkers|sources)|"
    # "Carpenter will discuss X" says what a text is about, not what it claims: it can never be contradicted.
    r"will (?:discuss|explore|provide|mention|address|examine|describe|talk about|cover|present|review|"
    r"investigate|highlight|emphasi[sz]e|consider|touch on|deal with|focus on|elaborate on|analy[sz]e|"
    r"explain the (?:difference|differences|distinction|role|nature|mechanisms?|characteristics))"
    # "Insights are provided on the interaction between ..." (the one test)
    r"|insights? (?:is|are) (?:provided|offered|given|gained)"
    # "The paper discusses the role of whistlers" names a topic in the present tense: still no claim.
    r"|(?:discuss|explor|examin|address|review|investigat|consider|highlight|stud(?:y|ies|ied))\w*\s+(?:the\s+)?"
    r"(?:role|impact|influence|relationship|link|connection|effect|importance|significance|mechanisms?|nature)\s+(?:of|between)"
    r"|provid\w*\s+(?:new\s+|valuable\s+|key\s+)?insights?\s+into)\b",
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


# "The plasmapause significantly influences the propagation of whistlers" names no direction, size or
# condition, so whatever a paper reports agrees with it: no text can contradict it. Half of a 7B model's
# predictions in an hour were of this kind, and the texts contradicted none of its 192 predictions.
_INFLUENCE_RE = re.compile(
    r"\b(?:plays?|playing)\s+(?:a|an)\s+(?:\w+\s+){0,2}role\b"
    r"|\b(?:influence[sd]?|influencing|affect(?:s|ed|ing)?|impact(?:s|ed|ing)?|shap(?:e|es|ed|ing)|contribut(?:e|es|ed|ing)\s+to"
    r"|interact(?:s|ed|ing)?\s+with|(?:is|are)\s+(?:closely\s+)?(?:related|linked|connected|associated)\s+(?:to|with)"
    r"|depend(?:s|ent)?\s+on|var(?:y|ies)\s+with|(?:is|are)\s+(?:important|crucial|essential|key|significant)\s+(?:for|to|in))\b"
    r"|\bha(?:s|ve)\s+(?:a|an)\s+(?:\w+\s+){0,2}(?:effect|impact|influence)\s+on\b",
    re.IGNORECASE,
)
# What makes such a claim risky: a direction, a size, a comparison, a condition, or a mechanism.
_RISK_RE = re.compile(
    r"\d|%|\b(?:increas\w*|decreas\w*|higher|lower|more|less|fewer|greater|smaller|larger|faster|slower|earlier|later|above"
    r"|below|exceed\w*|before|after|only|never|always|not|no|than|instead|whereas|unless|except|stronger|weaker|reduc\w*"
    r"|enhanc\w*|suppress\w*|amplif\w*|attenuat\w*|trap\w*|block\w*|prevent\w*|confin\w*|guid\w*|duct\w*|reflect\w*"
    r"|refract\w*|absorb\w*|dominat\w*|mostly|mainly|primarily|rarely|majority|most|outward|inward|toward|away|along|across"
    r"|inside|outside|beyond)\b",
    re.IGNORECASE,
)


def is_influence_only(claim: str) -> bool:
    """A claim that only says one thing influences another, with no direction, size, condition or mechanism."""
    return bool(_INFLUENCE_RE.search(claim or "")) and not _RISK_RE.search(claim or "")


# "Would be wrong if X played no role at all" or "...if Y alone explained everything": such a falsifier
# shows that the answer only claims that something plays some part. That survives every finding (an
# existential claim cannot be falsified, as Popper noted), so it is not a real test. Seen in 31 of 53
# answers of a 7B model.
_STRAWMAN_RE = re.compile(
    r"\b(?:no|any)\s+(?:significant\s+|measurable\s+|real\s+|meaningful\s+)?(?:role|influence|effect|impact|contribution|involvement)\b"
    r"|\bnot\s+(?:play|have)\s+(?:a|any)\s+(?:significant\s+)?(?:role|influence|effect|impact)\b"
    r"|\b(?:do|does|did)\s+not\s+(?:interact|affect|influence|matter|contribute)\b"
    r"|\b(?:only|sole|single|exclusive)\s+(?:\w+\s+)?(?:factor|cause|mechanism|driver|determinant|agent)s?\b|\bsolely\b|\balone\b"
    r"|\bregardless of\b|\bunder (?:any|all)\b|\bin all cases\b|\bindependent(?:ly)?\s+of\b",
    re.IGNORECASE,
)


def is_strawman_falsifier(text: str) -> bool:
    """A falsifier that only rules out "no role at all" or "one factor alone", which no finding can show."""
    return bool(_STRAWMAN_RE.search(text or ""))


# "Further research is needed to uncover X" or "X is not yet clearly defined" says that the question is
# open, not what the answer is. A 7B model held such answers at 0.6 to 0.7.
_NON_ANSWER_RE = re.compile(
    r"\b(?:further|more)\s+(?:research|investigation|exploration|study|studies|work)\s+(?:is|are)\s+"
    r"(?:needed|required|necessary|crucial|essential|warranted)"
    r"|\bneeds?\s+(?:further|more)\s+(?:research|investigation|exploration|study)"
    r"|\b(?:is|are|remains?)\s+(?:not\s+yet|yet\s+to\s+be)\s+(?:fully\s+|clearly\s+|definitively\s+|well\s+)?"
    r"(?:known|understood|defined|determined|identified|established|clear)"
    r"|\b(?:is|are)\s+not\s+(?:fully\s+|clearly\s+|definitively\s+|well\s+)(?:known|understood|defined|determined|established)"
    r"|\bremains?\s+(?:unclear|unknown|uncertain|an\s+open\s+question|to\s+be\s+(?:seen|determined|understood))"
    r"|\bneeds?\s+to\s+be\s+(?:further\s+)?(?:understood|explored|investigated|determined|clarified)",
    re.IGNORECASE,
)


# "How do undiscovered plasma instabilities interact with ...?" or "What other yet-to-be-identified structures
# ...?" asks about things nobody has found, which no paper can answer. A 7B model kept asking such questions
# about "lesser-known structures E1, E2, E3" (its own prediction labels) and "unidentified structures".
_UNKNOWN_RE = re.compile(
    r"\b(?:undiscovered|unidentified|undetected|unrecognized|unrecognised|lesser[- ]known|overlooked"
    r"|yet[- ]to[- ]be[- ](?:identified|discovered|detected|determined|found|recognized)"
    r"|not\s+yet\s+(?:identified|discovered|detected|known)"
    r"|unknown\s+(?:\w+\s+){0,2}(?:structures?|mechanisms?|factors?|instabilit(?:y|ies)|process(?:es)?|phenomena|waves?|modes?|propert(?:y|ies))"
    r"|(?:hidden|unseen)\s+(?:\w+\s+)?(?:factors?|mechanisms?|structures?))\b",
    re.IGNORECASE,
)


def asks_about_the_unknown(question: str) -> bool:
    """A question about things nobody has identified yet, which no text can answer."""
    return bool(_UNKNOWN_RE.search(question or ""))


def is_non_answer(text: str) -> bool:
    """An answer that only says the question is still open."""
    return bool(_NON_ANSWER_RE.search(text or ""))


def lexically_related(claim: str, quote: str, topic_words: frozenset[str] = frozenset(), *, floor: float = 0.2) -> bool:
    """Could this quote bear on this claim at all? A cheap check before the judge reads the pair.

    In a real run a 7B judge accepted "The role of lightning polarization is context-dependent" on the
    quote "The large-scale plasma environment is expected to play a central role in selecting these
    propagation pathways", which says nothing about polarization. A pair passes only if the two share
    at least ``floor`` of the shorter one's content words, and at least one word that is not one of the
    topic's own (every sentence about the topic shares "lightning" and "whistler").
    """
    a, b = token_set(claim), token_set(quote)
    if not a or not b:
        return False
    shared = a & b
    return len(shared) / min(len(a), len(b)) >= floor and bool(shared - topic_words)


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


def _in_order(quote: list[str], text: list[str], *, share: float = 0.9, slack: float = 1.5) -> bool:
    """Nearly all the quote's words, in order, within a short stretch of the text.

    Text taken from PDFs carries layout noise inside sentences (citation marks
    such as "[24, 25]", a word from the next column, a page header). A model
    that copies such a sentence cleanly is quoting, not inventing. Quotes of
    fewer than 6 words must match exactly.
    """
    n = len(quote)
    if n < 6:
        return False
    need, window = math.ceil(share * n), int(slack * n) + 2
    starts = {w for w in quote[:3]}
    for i, word in enumerate(text):
        if word not in starts:
            continue
        found, k, end = 0, i, min(len(text), i + window)
        for w in quote:
            for j in range(k, end):
                if text[j] == w:
                    found, k = found + 1, j + 1
                    break
        if found >= need:
            return True
    return False


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
