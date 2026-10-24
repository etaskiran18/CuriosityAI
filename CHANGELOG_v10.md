# v10: curiosity records, after İnan and Ady

v9 modelled curiosity as a drive: the strongest pull was pursued at once, so being curious and acting were the
same thing. Reading İlhan İnan's *The Philosophy of Curiosity* (2012) and Nadia Ady and colleagues' "Five
properties of specific curiosity" (2022, and Ady's thesis, 2023) led to a second architecture, chosen with
`--architecture records`. The v9 drive stays the default. Full description: [docs/CURIOSITY_RECORDS.md](docs/CURIOSITY_RECORDS.md).

| Idea | What v10 does |
|---|---|
| Curiosity has an object: something one can name but not yet identify (İnan's inostensible concept). | Every question gets a **record**: what exactly is unknown, what is known about it, what would answer it, why it matters, whether anyone could ever find it out, and whether to pursue it now (step CONCEIVE). |
| Awareness of novelty is not yet intellectual curiosity (İnan); animals seek novelty without concepts. | **Two layers**: surprises, unexpected passages and new texts only draw attention; a question is born only when its unknown is named. |
| One can be curious and not act on it (İnan). | **A decision of its own**: pursue now or wait, by rules (a person's question, its own wish, whether any text speaks of it, room for at most 6), with the reason in the diary. A waiting curiosity is taken up when a text speaks of it. |
| Curiosity ends when satisfied, and is not pursued again (Ady et al.). | **Satisfaction**: the model says a passage lets it identify the unknown (step SATISFY), the quote is verified, the blind judge agrees (`Judge.satisfies`); then the curiosity ends. A curiosity it is not pursuing can end too. |
| Curiosity fades when attention is elsewhere; some unknowns cannot be made known (Ady et al.). | **Transience**: intensity grows when touched and fades when not; a faded curiosity is gone. Its doubt that anyone could answer makes a curiosity wait; a look that finds nothing lets it go. Six looks without an answer and it waits. |
| A curious learner returns to what makes it curious, not to what answered it (Ady et al.). | **Voluntary exposure**: with nothing to pursue, or after a rest, it goes back to the text that awakened most curiosities and reads on (step RETURN). **Interests** grow from pursued and answered curiosities. |

Each part can be switched off for experiments: `--ablate decision,satisfaction,attention,transience,exposure`.

## Measuring it

* Reports and `metrics.json` gain lines and numbers for records, decisions, attention, transience and
  exposure; every session writes `curiosity_records.jsonl`; `compare_sessions.py` shows the new numbers.
* `scripts/trap_tests.py`: four situations whose right outcome is known (scrambled texts, an unreachable
  question, a hidden answer, an old answer), in fresh lives or in copies of an existing one.
* `scripts/fork_life.py`: copy a life at a deferred curiosity and live it both ways: was waiting right?
* `scripts/rating_sheet.py`: blind rating sheets (questions from several runs, source hidden) and their scores,
  with the agreement of two raters; `scripts/plain_curiosity.py`: the same model without the loop, as a baseline.

## From the first real runs (qwen2.5:3b, mistral 7B)

* The answer-form instruction offered a list of answer kinds, and the small model copied it; the prompt now asks
  it to complete "a text or an observation saying ..." and refuses copied hints and labels.
* Asked whether a text could ever answer a question, qwen2.5:3b said no to all four it was given; asked whether
  nobody could ever find it out, it said no to all four. Its doubt alone never lets a curiosity go.
* An invented quote ("the tyranny, 332") offered as an answer was refused before the judge saw it.

## Also

* Question statuses `deferred`, `satisfied`, `faded` and `dropped`; a curiosity's history of status changes.
* Old minds load unchanged: every new field has a default.
* Tests: 384 (from 311), with new files for records, decisions, transience, exposure, traps and evaluation.
