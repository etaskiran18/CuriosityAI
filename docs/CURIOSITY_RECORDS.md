# Curiosity records (v10): curiosity as a state with an object

Until v9 the organism modelled curiosity as a **drive**: every open question has a pull (gap, learning
progress, surprise, novelty, importance), and the question that pulls hardest is investigated at once.
Being curious and acting were the same thing.

v10 adds a second architecture, **records**, built on two sources:

* **İlhan İnan, *The Philosophy of Curiosity* (Routledge, 2012)** and "Inostensible reference and conceptual
  curiosity" (*Croatian Journal of Philosophy* 10(28), 2010). Intellectual curiosity has an object: something
  one can name but cannot yet identify, an *inostensible concept* ("the person who murdered Jones", "the cause
  of the dark stain on the wall"). Curiosity is satisfied when the curious one comes to believe that a new
  experience lets them know that object (2012, p. 135). Being aware of an unknown is not yet curiosity: "a
  certain kind of interest" is needed (p. 126). And curiosity does not always lead to action.
* **Nadia Ady and colleagues, "Five properties of specific curiosity you didn't know curious machines should
  have"** (arXiv:2212.00187, 2022; also chapters 4-5 of Ady's thesis *Specific Machine Curiosity*, University of
  Alberta, 2023, https://doi.org/10.7939/r3-km86-1f44): directedness towards inostensible referents,
  cessation when satisfied, voluntary exposure, transience, and coherent long-term learning. The situation that
  makes one curious differs from the one that satisfies it.

A language model works with concepts, so it can do what Ady's grid-world agent had to be given by hand: name
what it does not know and say what would answer it. The program keeps the state; the model forms the concept.

```bash
python live.py --architecture records --minutes 60            # the new architecture
python live.py --architecture records --ablate decision       # ... with one part switched off
python live.py --minutes 60                                    # the v9 drive (still the default)
```

## The curiosity record

Every question gets a record (`Question` in `mind.json`):

| field | what it holds |
|---|---|
| `unknown` | the thing it cannot yet identify, written as "the ... that ..." |
| `known` | what it already knows about that thing |
| `answer_form` | what would answer it: what it could then say, and what a text would have to state |
| `interest` | why it matters now |
| `reachable` | whether any text or observation could ever answer it, as the model judges |
| `wish`, `wish_reason` | whether it wanted to pursue it at once, and why |
| `intensity` | how strongly it is curious now (0 to 1) |
| `satisfied_by`, `identified_as`, `satisfied_at`, `satisfied_incidentally` | how the curiosity ended |

The step that writes it is **CONCEIVE** (`prompts.py`). A question it proposes (in SETTLE, or after a person
shared something) is born only if the model can name the unknown and say what would answer it; otherwise the
diary says "This caught my attention, but I could not say what exactly I do not know". Questions it began with,
was asked, or chose in a reflection are made precise the first time it pursues them.

## Satisfaction and cessation

After each heartbeat (`_satisfaction` in `organism.py`):

1. **The curious one believes it** (step **SATISFY**): given the record and the passages it has just read, the
   model says whether a passage lets it identify the unknown, copies the exact words, and says what the
   unknown is.
2. **The quote is verified**: the words must be in that passage (the person's own draft does not count).
3. **The blind judge checks it** (`Judge.satisfies`): could a reader of the quote alone say what the unknown
   is, in the way the answer form describes? A schema hint copied back ("yes or no: ...") counts as no verdict.

Only then does the curiosity end (status `satisfied`), and it is not pursued again: a question proposed later
that repeats it is not asked ("Not asked again: Q5 already ended"). A person can still ask it again.

The same check runs for up to `satisfaction_checks` other curiosities whose object the passages speak of, so a
curiosity it is **not** pursuing can end too, when the answer comes along while it reads for something else
(`satisfied_incidentally`).

With satisfaction on, the v9 rule that settles a question at confidence 0.8 is off: a curiosity ends because a
text answered it, not because the model grew sure of itself.

## What a session reports

`report.md` has a **Curiosities** line and a section **Curiosities that ended**; `metrics.json` adds
`records_made`, `records_not_made`, `curiosities_satisfied`, `satisfied_while_waiting`, `satisfaction_refused`,
`looks_to_satisfaction`, `already_satisfied_not_asked`, `questions_deferred`, `questions_faded` and
`questions_dropped`. `research_map.md` lists what it is curious about (the unknown and what would answer it)
and the curiosities that ended, each with its quote. Satisfaction verdicts appear in `judge_check.csv` for a
hand check (kind `satisfaction`).

## Switching parts off (ablation)

`--ablate` takes a comma-separated list of `decision`, `satisfaction`, `attention`, `transience`, `exposure`.

| part | what it does | without it |
|---|---|---|
| `satisfaction` | a verified quote, accepted by the judge, ends a curiosity | questions settle by confidence, as in v9 |
