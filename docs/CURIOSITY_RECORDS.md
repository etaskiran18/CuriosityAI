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
  certain kind of interest" is needed (p. 126). Animals and children before language seek novelty and
  sensation but have no curiosity of this kind (p. 125). And being curious does not always lead to action.
* **Nadia Ady and colleagues, "Five properties of specific curiosity you didn't know curious machines should
  have"** (arXiv:2212.00187, 2022; chapters 4-5 of Ady's thesis *Specific Machine Curiosity*, University of
  Alberta, 2023, https://doi.org/10.7939/r3-km86-1f44): directedness towards inostensible referents,
  cessation when satisfied, voluntary exposure, transience, and coherent long-term learning. The situation that
  makes one curious differs from the one that satisfies it. Their grid-world agent is given its curiosity by
  hand ("an oracle-like mechanism", thesis p. 163); naming the unknown is left open.

A language model works with concepts, so it can do what that agent had to be given: name what it does not know
and say what would answer it. In this design **the model forms the concept and the program keeps the state**:
the model has no wish that lasts from one call to the next, so whether a curiosity lasts, fades or ends lives
in a record the program keeps.

```bash
python live.py --architecture records --minutes 60             # the new architecture
python live.py --architecture records --ablate decision        # ... with one part switched off
python live.py --minutes 60                                     # the v9 drive (still the default)
```

The parts can be switched off one by one (`--ablate decision,satisfaction,attention,transience,exposure`) to see
what each does, as Ady and colleagues did with theirs.

## The curiosity record

Every question has a record (`Question` in `mind.json`):

| field | what it holds |
|---|---|
| `unknown` | the thing it cannot yet identify, written as "the ... that ..." |
| `known` | what it already knows about that thing |
| `answer_form` | what would answer it: "a text or an observation saying ..." |
| `interest` | why it matters now |
| `reachable` | whether anyone could ever find it out, as the model judges |
| `wish`, `wish_reason` | whether it wanted to pursue it at once, and why |
| `intensity` | how strongly it is curious now (0 to 1) |
| `interest_fit` | how close it was, when born, to what it had come to be interested in |
| `decision`, `decision_reason`, `history` | whether it pursues it, why, and every change of status |
| `satisfied_by`, `identified_as`, `satisfied_at`, `satisfied_incidentally` | how the curiosity ended |
| `origin_sources` | the texts it was reading when the curiosity was born |

The step that writes it is **CONCEIVE** (`prompts.py`, `organism._conceive`). Questions it began with, was
asked, or chose in a reflection are made precise the first time it pursues them.

## Attention, then a concept (`attention`)

Two layers. The first is pre-conceptual: in each heartbeat it notes what drew its attention, with a strength:
a confident prediction (p >= 0.5) a text contradicted, an unexpected passage the judge accepted, a text it had
never read (`organism._attention_events`). These signals give birth to nothing. A question it proposes is born
only if the model can name its unknown and say what would answer it; otherwise the diary says "This caught my
attention, but I could not say what exactly I do not know". The share of attention that turns into a curiosity
is in the report (`attention_to_curiosity`). Without this part, an unnamed question is born as in v9.

## Deciding whether to pursue (`decision`)

Being curious and acting on it are two things (İnan). When a curiosity is born, a separate step decides, by
rules, whether to pursue it now, and keeps the reason (`organism._weigh`):

1. a person's question is pursued;
2. a curiosity it doubts anyone could ever answer waits (see transience);
3. a curiosity it said it wants to come back to later waits, with its own reason;
4. one that no text it has speaks of waits, when it cannot fetch new texts;
5. at most `max_pursued` (6) are pursued at once: a weaker one waits, or a stronger one takes the place of
   the weakest.

A waiting curiosity is taken up again when what it reads speaks of it. The diary says "I am curious about Q7,
but I do not pursue it now, because ...". The report follows every curiosity it ever deferred to its end:
answered while it waited, answered after it was pursued again, pursued again, still waiting, faded or let go.
Without this part, every curiosity is pursued at once.

## Satisfaction and cessation (`satisfaction`)

After each heartbeat (`organism._satisfaction`):

1. **The curious one believes it** (step **SATISFY**): given the record and the passages it has just read, the
   model says whether a passage lets it identify the unknown, copies the exact words, and says what it is.
2. **The quote is verified**: the words must be in that passage (the person's own draft does not count).
3. **The blind judge checks it** (`Judge.satisfies`): could a reader of the quote alone say what the unknown
   is, in the way the answer form describes? A schema hint copied back ("yes or no: ...") is no verdict.

Only then does the curiosity end, and it is not pursued again: a question proposed later that repeats it is
not asked. A person can still ask it again. The same check runs for up to `satisfaction_checks` other
curiosities whose object the passages speak of, so a curiosity it is not pursuing can end too, when the answer
comes along while it reads for something else. Without this part, questions settle by confidence, as in v9.

## Transience (`transience`)

A curiosity lasts while attention stays with it (Loewenstein 1994; Ady et al.). What a heartbeat touches (the
question it looks at, curiosities whose object the passages speak of, a new question that repeats one) grows
stronger; the rest fades by `intensity_decay` (0.9) per heartbeat, and below `fade_below` (0.15) the curiosity
has faded and is gone (never a question it began with or a person asked).

Some unknowns cannot be made known ("the nearest Earth-orbiting teapot", Ady's thesis p. 204). The model is
asked whether the thing is something nobody could ever find out. Its doubt alone lets nothing go: such a
curiosity waits, and is let go only when a look finds that the texts say nothing of it. A curiosity looked at
`pursuit_limit` (6) times without an answer waits as well (committing to a goal helps, but not for ever; thesis
pp. 157, 203). This judgment replaces v9's list of words for "undiscovered" things. Without this part there is
no fading, no time limit, and the word list is back.

## Voluntary exposure and interests (`exposure`)

Ady's agent learns to return to the place that makes it curious (the bookstore), not to the page that
answered it. Each text keeps count of how often it was read, how many curiosities it awakened and how many it
answered (`MindState.sources`). With nothing to pursue, or after a rest, the organism goes back to the text with
the highest curiosity yield and reads on where it stopped (step **RETURN**). Its **interests** grow from the
words of the curiosities it pursued (weight 1) and had answered (weight 2); a new curiosity near them starts
stronger (Wade and Kidd 2019: prior knowledge predicts curiosity). The research map shows the texts that made it
curious and its interests. Without this part it never goes back and keeps no interests.

## What a session reports

`report.md` has lines for **Curiosities**, **Curious, but not acting**, **Attention and transience** and
**Voluntary exposure and interests**, and a section **Curiosities that ended**. `metrics.json` adds, among
others: `records_made`, `records_not_made`, `curiosities_satisfied`, `satisfied_while_waiting`,
`satisfaction_refused`, `looks_to_satisfaction`, `already_satisfied_not_asked`, `pursued_at_birth`,
`deferred_at_birth`, `deferral_reasons`, `pursued_after_waiting`, `deferred_fates`, `attention_events`,
`attention_to_curiosity`, `questions_faded`, `let_go_unreachable`, `deferred_after_looking`, `returns_to_texts`,
`read_from_curious_texts` (first and second half) and `interest_fit_of_new`. Every curiosity of the life, with
its record and fate, is written to `curiosity_records.jsonl` in the session folder. Satisfaction verdicts are
in `judge_check.csv` for a hand check (kind `satisfaction`). `scripts/compare_sessions.py` lists them side by
side.

## Evaluating it

There is no single score for curiosity; Ady says as much (thesis p. 211). Three questions, each with its own
measure and its own judge:

| question | measure | who judges |
|---|---|---|
| Does the mechanism behave as designed? | the numbers above, trap tests, switching parts off, forking | the program |
| Is what comes out curiosity in İnan's sense? | blind ratings of the records | a philosopher and students |
| Does it help research? | grounded answers, the exam, the time-split test | the researcher and the blind judge |

**Comparisons.** Same model, same library (no `--web`, or a fixed `--papers` folder), same number of
heartbeats (`--heartbeats`), a life of its own for each (`--home`), at least three seeds:

```bash
for s in 1 2 3; do
  python live.py --home runs/drive-$s   --architecture drive   --heartbeats 100 --seed $s --label drive
  python live.py --home runs/records-$s --architecture records --heartbeats 100 --seed $s --label records
  python live.py --home runs/no-decision-$s --architecture records --ablate decision --heartbeats 100 --seed $s --label no-decision
  python live.py --home runs/novelty-$s --policy novelty --heartbeats 100 --seed $s --label novelty
done
python scripts/compare_sessions.py runs/*/sessions/* --csv results.csv
```

**Trap tests** (`scripts/trap_tests.py`, `organism/traps.py`): situations whose right outcome is known.
*noise*: scrambled texts are new, so they draw attention, but no curiosity should come of them; *unreachable*: a
person asks what nobody could ever find out, and it should let it go within a few looks; *hidden*: the answer
to a waiting curiosity is planted where it will read for other questions, and it should recognize it without
pursuing it; *old*: shown the words that ended a curiosity, it should not take it up again. Traps can be set
in a copy of an existing life (`--from-home`), for example a researcher's.

```bash
python scripts/trap_tests.py --home runs/traps --model mistral:7b-instruct --heartbeats 8
```

**Forking** (`scripts/fork_life.py`, `organism/fork.py`): a life is copied at a deferred curiosity; one copy
lives on as decided, the other pursues that curiosity. The result says whether deferring was right, cost an
answer, or saved looks. A person cannot live the same afternoon twice; a program can.

**Blind ratings** (`scripts/rating_sheet.py`, `organism/rating.py`): questions from several runs are mixed and
their source hidden. Level 1 asks of each question: an intellectual curiosity? worth asking? Level 2 shows the
whole record with five 0-2 scores (a specific unknown; what it knows is correct; what would answer it is clear;
the interest is real; the reason is sensible). `score` gives the result per run and the agreement of two
raters (Cohen's kappa). **The model without the loop** (`scripts/plain_curiosity.py`) gives a baseline: the
same model reads the same library and is asked the same two questions, with no loop around it.

**Research value.** The exam (`--exam`) and the time-split test (`--until-year`, docs/RESEARCHER.md): it reads
only papers up to a year; do its curiosities, and the ones it deferred because no text spoke of them, point to
what was published afterwards?

**Check the instruments first.** Satisfaction and contradiction verdicts come from a model judge; check a
sample by hand (`judge_check.csv`, `scripts/judge_agreement.py`) before trusting the numbers built on them, and
use a larger judge (`--judge-model`) if it agrees with you less than about 80% of the time.

**Write the predictions down before running**, for example: the novelty policy dwells on the scrambled texts
and the records architecture does not; raters call the records' questions intellectual curiosity more often
than the drive's and the plain model's; grounded answers do not drop, and looks per answered curiosity fall.
A prediction that fails is a finding too.

## What the first real runs taught the prompts

With qwen2.5:3b on a CPU:

* Asked for an answer form "(a mechanism, a number, a condition, a name, ...)", it copied the list. The
  prompt now asks it to complete "a text or an observation saying ...", gives one worked example from
  another domain (a stain on a wall; a record that copies it is refused), and labels copied into a field
  ("why it matters to you now: ...") are taken off.
* Asked "could any text or observation ever answer it?", it said no to every question, a mechanism of
  whistler propagation included ("not well documented"). Asked instead whether it is "something nobody could
  ever find out", it said no to all of them: a small model's habit of answering no now falls on the right
  side. Its doubt alone still lets nothing go.
* It once claimed a passage answered its curiosity and quoted "the tyranny, 332", which is in no passage: the
  verification refused it before the judge was asked.

A 7B model makes better records (mistral, for the whistler question: "the specific changes in the paths of
lightning-generated whistlers due to plasmasphere refilling after a storm", answered by "detailed data on the
changes in frequency, amplitude and propagation direction ... and their correlation with the refilling").
Records, satisfaction checks and decisions add about three model calls to a heartbeat.

## Limits

* Whether a text "speaks of" a curiosity is decided by shared words (the topic's own words left out), not by
  meaning.
* A small judge can be wrong about satisfaction; that is why its verdicts are sampled for a hand check.
* Intensity, decay and the decision rules are simple and fixed; they are there to be compared, not tuned.
