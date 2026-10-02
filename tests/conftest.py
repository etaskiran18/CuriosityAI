from __future__ import annotations

from pathlib import Path

import pytest

from curiosity_ai.config import AppConfig

THEAETETUS = """---
title: Theaetetus
author: Plato
---

The Project Gutenberg eBook of Theaetetus. This header talks about the license and redistribution terms.

*** START OF THE PROJECT GUTENBERG EBOOK THEAETETUS ***

THEAETETUS: Yes, Socrates, and I am amazed when I think of them; by the Gods I am!
and I want to know what on earth they mean; and there are times when my head
quite swims with the contemplation of them.

SOCRATES: I see, my dear Theaetetus, that Theodorus had a true insight into
your nature when he said that you were a philosopher, for wonder is the feeling
of a philosopher, and philosophy begins in wonder. He was not a bad genealogist
who said that Iris (the messenger of heaven) is the child of Thaumas (wonder).

*** END OF THE PROJECT GUTENBERG EBOOK THEAETETUS ***
"""

DEWEY = """---
title: How We Think
author: John Dewey
---

Curiosity rises above the organic and the social planes and becomes intellectual
in the degree in which it is transformed into interest in problems provoked by the
observation of things and the accumulation of material. When the question is not
discharged by being asked of another, when the child continues to entertain it in
his own mind and to be alert for whatever will help answer it, curiosity has become
a positive intellectual force.
"""

KANT = """---
title: The Critique of Pure Reason
author: Immanuel Kant
---

Human reason, in one sphere of its cognition, is called upon to consider questions,
which it cannot decline, as they are presented by its own nature, but which it cannot
answer, as they transcend every faculty of the mind. It falls into this difficulty
without any fault of its own.
"""

COOKING = """---
title: Simple Soups
author: A Cook
---

Chop the onions and carrots, warm the butter in a heavy pot, and let the vegetables
sweat gently before adding the stock, the bay leaf, and a little salt and pepper.
"""

SEED = "Plato says philosophy begins in wonder: what exactly is wonder, and is it the same thing as curiosity?"


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    for name, text in {"theaetetus.md": THEAETETUS, "dewey.md": DEWEY, "kant.md": KANT, "soups.md": COOKING}.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


@pytest.fixture
def config(tmp_path: Path, corpus_dir: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.corpus.path = str(corpus_dir)
    cfg.organism.home = str(tmp_path / "organism")
    cfg.organism.random_seed = 7
    cfg.organism.reflect_every = 0
    cfg.organism.seed_questions = [SEED]
    cfg.organism.body.enabled = False  # no real rests in tests; see test_organism_body.py
    return cfg
