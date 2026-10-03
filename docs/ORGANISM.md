# The Curiosity Organism (v9)

v9 keeps the v8 organism and fixes what a first real hour showed (see [CHANGELOG_v9.md](../CHANGELOG_v9.md)):
its predictions could never be wrong, it graded its own evidence too kindly, and it drifted off its topic
without noticing. It can now also study **your own research topic** ([RESEARCHER.md](RESEARCHER.md)).

## The idea

Versions 1 to 7 ask a language model to *write about* curiosity, then use more and more judge agents to
grade how deep the writing is. v8 changes what curiosity is in the system. Curiosity is no longer only
the topic. It is the **mechanism that decides what the system does next**.

The organism has a persistent **mind**: open questions, beliefs with evidence, and a temperament. At
every heartbeat a **curiosity drive**, computed from that mind, chooses the question that pulls hardest.
The organism commits to **predictions before it reads**, so the texts can **surprise** it. Surprise and
contradiction give birth to new questions. **Learning progress** and **boredom** decide where its
attention goes next. Every few heartbeats it **reflects on its own way of being curious** and adjusts
itself.

Each of these mechanisms comes from the philosophy or psychology of curiosity. The table below shows
where each one comes from.

## One heartbeat

```text
      +---------------- reflection, every few heartbeats -----------------+
      |  vital signs -> diagnosis -> adjust temperament -> theory         |
      v                                                                   |
  choose --> predict --> observe --> compare --> judge --> argue --> settle --> learn
  (drive,    (claims +   (read the   (verified    (blind     (Wonder,   (answer that  (Brier error,
   topic)     probabil-   library)    quotes)      check of   Skeptic,   could be      progress,
              ities)                               each       defend/    wrong, one    boredom)
                                                   quote)     revise/    new question)
                                                              concede)
```

| Step | What happens | Root |
|---|---|---|
| choose | Every open question gets a *pull* from the curiosity drive. Questions far from the main topic pull less. A softmax picks one, so the organism mostly follows the strongest pull and sometimes wanders. | Loewenstein, Oudeyer, Berlyne (below) |
| predict | Before reading, the organism writes its current answer, its confidence, and 2-4 **predictions**: who will say what, and the probability that the texts support it. "May", "might" and "could" are not allowed (a hedged guess can never be wrong); it is asked once more if it hedges. | Peirce: real doubt needs a real expectation; Popper: a claim must be able to fail |
| observe | BM25 search over `data/philosophy_corpus` (or, in researcher mode, your papers) and anything a human shared. In researcher mode what the person shares is their own work: at most one passage per heartbeat, read as their claims and never as evidence. | Dewey: curiosity seeks "material for thought" |
| compare | For each prediction: confirmed, contradicted, or not addressed. A claim counts only if its quote is **found in the passage**. | Peirce's "irritation of doubt" |
| judge | A separate **blind judge** sees only each claim and its quote (never the organism's reasoning or verdict) and decides: supports, contradicts, or neither. Only what the judge accepts counts as evidence. | A witness is not the judge of its own case |
| argue | Wonder proposes an explanation that could be wrong; Skeptic attacks it **with evidence** (a quote, or a named thinker who would disagree); Wonder answers with one word first: DEFEND, REVISE or CONCEDE. | Socratic elenchus; Mill: "He who knows only his own side of the case, knows little of that" |
| settle | Revised answer (specific enough to be wrong, with what would refute it) and confidence, new beliefs (grounded only with the judge's approval), doubts about old beliefs, and **at most one** new question, the one closest to the topic. | Dewey's fifth step: "acceptance or rejection" |
| learn | The visit's prediction error (Brier score) and change of confidence are stored. They feed learning progress, boredom, settling and dormancy. Grounded episodes become training data. | Oudeyer, Schmidhuber |

## Philosophy becomes mechanism

Line numbers point into the files in `data/philosophy_corpus/`.

| Idea | Source | What it does in the code |
|---|---|---|
| "Wonder is the feeling of a philosopher, and philosophy begins in wonder." | Plato, *Theaetetus* 155d (`plato_theaetetus_jowett.txt` l. 3915) | The organism is born with **questions, not answers**. Questions are the unit of its life. |
| "How will you enquire, Socrates, into that which you do not know?" | Plato, *Meno* 80d (`plato_meno_jowett.txt` l. 1529) | You can only search for what you partly know. The **information gap** is zero at confidence 0 and 1 and highest in between. A newborn question starts at 0.25 confidence, because being able to ask it already means knowing something. **Anchoring**: a gap is felt more strongly when related beliefs exist. |
| "I neither know nor think that I know." | Plato, *Apology* 21d (`plato_apology_jowett.txt` l. 669) | Every question and belief carries an explicit confidence. `mind.json` is a map of known unknowns. |
| "The irritation of doubt causes a struggle to attain a state of belief." | Peirce, *The Fixation of Belief* (1877), in *Chance, Love, and Logic* (`peirce_chance_love_logic.md` l. 1391) | **Predict, then compare.** Prediction error is surprise. A question **settles** when the organism is confident *and* no longer surprised; new contradicting evidence reopens it. |
| "Let us not pretend to doubt in philosophy what we do not doubt in our hearts." | Peirce, *Some Consequences of Four Incapacities* (1868) | No paper doubt and no paper confirmation: confirmations and contradictions count only with a **verified quote**. |
| Five steps of reflective thought: felt difficulty, definition, suggestion, reasoning, observation leading to "belief or disbelief". | Dewey, *How We Think* ch. 6 (`john_dewey_how_we_think.txt` l. 2376) | The heartbeat: chosen question (difficulty), expectations (definition), Wonder (suggestion), Skeptic (reasoning), compare and settle (observation and acceptance or rejection). |
| Curiosity becomes intellectual "when the question is not discharged by being asked of another" but held "in his own mind". | Dewey ch. 3 (l. 1188) | Questions persist across runs in an open-question graph, with parents and children. |
| The child experiments with objects "till they cease to yield new qualities". | Dewey ch. 3 (l. 1156) | **Novelty habituates**: 1 / (1 + visits). |
| Curiosity is lost "in indifference or carelessness", "frivolous flippancy", "hard dogmatism", "routine". | Dewey ch. 3 (l. 1204) | The self-diagnoses below. |
| The philosophic mind responds to "an inconsistency or a gap in its knowledge". | William James, *Principles of Psychology* (1890), ch. 24 (`james_principles_psychology_vol2.md` l. 17336) | Questions are born from **contradiction** and **gaps** (and from surprise and objections). |
| Reason "is called upon to consider questions, which it cannot decline ... but which it cannot answer". | Kant, *Critique of Pure Reason*, preface (`kant_critique_pure_reason.txt` l. 350) | A question can be marked **unanswerable** when the texts stay silent across visits, instead of looping forever. Your v7 run spent 3 of 4 iterations stuck in forced revision of one question. |
| *Curiositas*, the "lust of the eyes"; *studiositas* as its virtuous counterpart. | Augustine, *Confessions* X.35 (`augustine_confessions_pusey.md` l. 6365); Aquinas, *ST* II-II q.166-167 | Diagnosis **restless curiositas**: questions multiplying faster than any is settled, or drifting from the topic. |
| *Neugier*: curiosity that seeks novelty "not in order to understand" and never dwells. | Heidegger, *Being and Time* section 36 | The main reward is **learning progress, not novelty**; **boredom** ends unproductive visits; the **topic anchor** and **one new question per heartbeat** keep it dwelling. |
| "He who knows only his own side of the case, knows little of that." | Mill, *On Liberty* ch. 2 (`mill_on_liberty.md` l. 1720) | The Skeptic must bring the other side's evidence, not just say "it is more complex". |
| The torpedo fish that numbs those it touches. | Plato, *Meno* 80a (l. 1501) | Diagnosis **aporetic numbness**: perplexity that paralyzes. |
| Hume "interrupted my dogmatic slumber". | Kant, *Prolegomena* (1783) (`kant_prolegomena.md` l. 365) | Diagnosis **dogmatic slumber**: nothing surprises it anymore. |
| Curiosity is "a Lust of the mind, that by a perseverance of delight in the continuall and indefatigable generation of Knowledge, exceedeth the short vehemence of any carnall Pleasure." | Hobbes, *Leviathan* ch. 6 (`hobbes_leviathan.md` l. 1685) | Curiosity is a lasting appetite: the organism never runs out of questions, and with `--forever` it keeps living. |
| The love of truth resembles hunting: the truth must seem important, and finding it must take effort. | Hume, *Treatise* 2.3.10 (`hume_treatise_human_nature.md` l. 15322, l. 15426) | The **importance** term. Questions from humans start with high importance. |
| Information-gap theory; curiosity peaks at intermediate confidence. | Loewenstein (1994); Kang et al. (2009) | `information_gap(c) = 4c(1-c)` |
| Intrinsic motivation as learning progress / compression progress. | Oudeyer, Kaplan & Hafner (2007); Schmidhuber (2010) | `learning_progress()`: drop in prediction error plus change of belief. |
| The "noisy TV": a curious agent trapped by unpredictable noise. | Burda et al. (2018) | Unlearnable surprise gives no learning progress, so boredom takes over. See `test_noisy_tv_loses_to_a_learnable_question`. |
| Collative variables: novelty, surprise, conflict. | Berlyne (1960) | The novelty and surprise terms. |
| Incubation. | Wallas, *The Art of Thought* (1926) | **Dormant** questions wake up when enough new related beliefs have formed since they went to sleep. |

## The curiosity drive

```text
pull(q) = [ w_gap        * gap(confidence) * anchoring
          + w_progress   * learning_progress
          + w_surprise   * surprise
          + w_novelty    * novelty
          + w_importance * importance
          + news ] * (1 - boredom) * (1 - refractory)
                   * (1 - topic_anchor * (1 - relevance))
```

* `gap = 4c(1-c)`, where `c` is the confidence in the current answer.
* `anchoring = 0.4 + 0.6 * min(1, related_beliefs / 3)`.
* `learning_progress`: an optimistic prior before two visits; after that, the fall in prediction error
  (older half of the recent visits minus newer half) plus the average change of confidence.
* `surprise`: the last visit's surprise, fading by 0.85 per heartbeat. A newborn question inherits the
  surprise of the episode that created it. Each prediction `i` carries a probability `p_i`. A prediction
  the texts confirmed costs `(1 - p_i)^2`, one they contradicted costs `p_i^2` (the **Brier score**). For
  a visit with `n` predictions, `a` of them addressed (confirmed or contradicted, as the judge accepted),
  and `u` genuinely unexpected findings:

  ```text
  informativeness = (a + u/2)              / (n + u/2)   how much the texts spoke to my predictions
  surprise        = (sum of Brier costs + u/2) / (n + u/2)
  ```

  A confident prediction that comes true costs almost nothing; a confident one that fails costs a lot; a
  50/50 guess always costs 0.25. Texts that stay silent cannot be very surprising, and learning progress
  uses the error weighted by informativeness, so a silent visit cannot pass for progress.
* `novelty = 1 / (1 + visits)`.
* `boredom`: zero until `boredom_patience` visits without progress, then rising by 0.35 per visit.
* `refractory = 0.5` if the question was visited in the previous heartbeat, to discourage rumination.
* `news = 0.25` when texts were fetched for the question since its last visit: it goes back to see whether
  they answer it (no refractory pause then). Without this, a first research run fetched texts for a
  question and never looked at the question again.
* `relevance`: how directly the question serves the main topic (0 to 1). The judge rates every new
  question (0-3) in a call of its own. A question is **set aside** (not pursued) when both the judge's
  rating and the topic's own words put it below `topic.min_relevance`: a small judge sometimes rates
  everything 0, and one confused rating must not throw away a good question. Seed questions and
  questions from humans count as central. Older, unrated questions are estimated from the topic's words. `topic_anchor` (0.6 to start) is part of the temperament, so self-regulation can change it.

The code is in `curiosity_ai/organism/drive.py`. It is plain arithmetic and is unit-tested in
`tests/test_organism_drive.py`.

## Honesty: why it is harder to fool

Your v7 run showed the usual failure modes of a small model judging itself. Every thesis drifted to
"curiosity needs balance", the move detector gave 1.000 to a platitude, and the web search returned
papers on copyright law and medical curricula. v8 stopped asking the model to grade its own depth. The
first real v8 hour showed what was still too easy: 97% of its predictions were hedged ("the texts may
discuss..."), nothing contradicted it in 85 heartbeats, and about half of its "confirmations" did not
support the claim at all. v9 adds the last four points below.

* **Predictions come before evidence**, so surprise is measured rather than claimed.
* **Quotes are verified.** A confirmation or contradiction counts only if its quote is found in the
  passage that was read. Fabricated quotes are discarded and counted in the diary.
* **A blind judge decides what a quote proves** (`organism/judge.py`). It sees the claim, the quote and
  who wrote the quote, nothing else. A quote it finds beside the point confirms nothing; a belief is
  "grounded" only with a quote it accepted. It can be a different, larger model (`--judge-model`), and a
  sample of its decisions is written to `judge_check.csv` so a person can check it.
* **Predictions must be able to fail**: no "may" or "might"; doubt goes into a probability, and the
  Brier score rewards being right *and* knowing how sure to be. A prediction that only says one thing
  influences another ("the plasmapause significantly influences the propagation of whistlers") cannot
  fail either: whatever a paper reports agrees with it. Half of a 7B model's predictions were of this
  kind, and in 51 heartbeats the texts contradicted none. Such predictions are asked again with the
  hedged ones (one retry: say which way, how much, under which condition, or by which mechanism).
* **Only a test that could have failed counts** (Popper's severe test): a confirmation makes an answer
  surer only if its prediction was risky, that is neither hedged nor a mere claim of influence.
* **Answers must be specific enough to be wrong.** An answer leaning on "complex", "multifaceted",
  "various factors" earns no confidence, and one that cannot say what would refute it gains at most 0.05.
  So does a hedged answer ("the plasmapause may play a role", "could potentially affect"): like a hedged
  prediction, it survives any finding. The doubt belongs in the confidence, not in the wording. So does
  an answer that would be wrong only "if X played no role at all" or "if Y alone explained everything":
  it only claims that something plays some part, and no finding can refute that (Popper: an existential
  claim cannot be falsified). An answer that is itself a question is no answer; the old one stays.
* **No paper doubt** (Peirce): a belief that rests on a quote the judge accepted is doubted only when the
  judge accepted a contradiction in the same heartbeat. Without one, the doubt is refused and noted in the
  diary. Interpretations (beliefs without a quote) can be doubted freely. A hedged belief ("X may play a
  role") is not checked against a quote at all: almost any related quote supports it, so it stays an
  interpretation.
  A theory of curiosity that vague is not adopted at reflection.
* **Conceding is not winning**: after Wonder concedes an objection its confidence cannot rise, and
  after a revision it can rise by at most 0.10.
* **No citations from memory**: Wonder and Skeptic may refer only to the passages they were shown. A
  turn that cites other papers ("T. Nakamura et al., JGR vol. 82") is marked unverified in the diary
  and counted in the report: such references may be invented. A reference inside a quoted passage
  (the passage itself cites "Inan et al., 1990") is not counted.
* **Only confirmed predictions make an answer surer.** Without a confirmation the judge accepted,
  confidence rises by at most 0.05, however surprising the side findings were.
* **Confidence moves in bounded steps**: at most 0.25 per heartbeat, at most +0.05 without evidence,
  and at most +0.10 when heavily surprised.
* **Confidence belongs to the answer, and cannot outrun its evidence.** Each question keeps the quotes
  the judge accepted for its *current* answer (at most four). Every heartbeat the judge reads the answer
  against them and against this heartbeat's confirmations, in the same call that checks new beliefs; only
  the quotes it accepts are kept. Confidence can never be higher than `0.5 + 0.1 x` those quotes, so a new
  answer does not inherit what an old one earned (a 7B model held 0.70 for an answer a paper confirmed,
  replaced it with an unsupported guess and kept the 0.70), and settling (0.8) needs three. The research
  map lists each hypothesis's supporting quotes.
* **A pair that shares no real words never reaches the judge.** A belief is checked against a quote only
  if they share at least a fifth of the shorter one's content words, and at least one word that is not
  the topic's own: "The role of lightning polarization is context-dependent" is not grounded by "The
  large-scale plasma environment is expected to play a central role in selecting these propagation
  pathways", although a 7B judge accepted it. The same check applies to the quotes kept for an answer.
* **An answer that only says the question is open** ("further research is needed", "not yet clearly
  defined") counts as too vague to be wrong: it earns no confidence.
* **The comparison is told to be strict**: a passage on a related topic confirms nothing, and most
  expectations will not be addressed. The same quote cannot count as both expected and unexpected.
* **Beliefs without a verified quote are labelled "interpretation"** and start at lower confidence.
* **Contradictions can only target beliefs the model was actually shown.**
* **Web results must cover at least 25% of the question's content words**; off-topic results are dropped.
* **Citations point to real file lines**, so a human can check them.

## Self-regulation

At each reflection the organism measures its **vital signs** over recent episodes: progress rate (how
often **evidence** moved its mind: a belief grounded with the judge's approval, or a change of
confidence or a doubt in a heartbeat where the judge accepted a confirmation or contradiction), mean
surprise, diversity of questions visited, new questions per episode, questions settled per episode,
and closeness to the topic. A confidence bump with no evidence behind it does not count: in the first
real hour that was exactly what made every self-check say "healthy" while it drifted.
**Depth before breadth.** A first research run asked a new question at almost every heartbeat and
never returned to one; four of them were the same question reworded. Now a new question needs a reason:
a "surprise" needs evidence the judge accepted and a real prediction error, a "contradiction" needs a
contradiction the judge accepted (doubting an earlier belief counts only when the heartbeat brought such
evidence: a small model doubts its earlier guesses on nothing but newer guesses), an "objection" needs
the skeptic to have quoted a text, and a question born from a gap
waits until its parent has had a second look (`min_visits_before_children`, 2): a gap is a gap only if it
is still there when you look again, often with newly fetched texts. Proposals without a reason yet are
noted in the diary ("kept for a second look"). A question about things nobody has identified
("undiscovered plasma instabilities", "lesser-known structures", "yet-to-be-identified ...") is not asked
at all: no paper can answer it.

It then diagnoses its state and nudges its temperament in small, bounded steps:

| Diagnosis | Pattern | Response |
|---|---|---|
| Healthy wonder | evidence-backed progress, on topic | relax toward the baseline temperament |
| Restless curiositas | many questions with little progress; or new questions multiplying (0.8+ per heartbeat) while none is settled; or drifting from the topic (closeness below 0.5) | value progress over novelty; anchor closer to the topic; more patience |
| Dogmatic slumber | almost no surprise, few new questions | raise surprise and novelty; loosen the topic anchor; explore more |
| Aporetic numbness | high surprise, little progress | get bored sooner; prefer anchored questions |

The language model also writes a short reflection and updates the organism's **own theory of
curiosity** (in researcher mode: of its topic). It is shown, separately, the answers the texts support
(with a quote), the beliefs that rest on quotes, and its guesses, and is asked to build the theory on
the supported part and to call a guess a guess; a theory that only says "not yet understood" does not
replace the old one, and a focus question about undiscovered things is not adopted. (A 7B model had
built a whole hour's theory around one unsupported guess, "lightning polarization".) That theory is
shown to the Wonder voice at every heartbeat. What the organism comes to
believe about curiosity therefore shapes how it wonders, and its measured experience of inquiry
shapes how it allocates curiosity.

## Taking care of the computer it lives in

A local model keeps the GPU busy, and a laptop that thinks for days gets hot. So the organism also
regulates its **body**, the computer (`curiosity_ai/organism/body.py`, settings in `organism.body`):

| Protection | Default | How |
|---|---|---|
| Breath | 10 s after every heartbeat | time-based, works everywhere |
| Work and rest rhythm | 5 min rest after every 20 min of thinking | time-based, works everywhere |
| Temperature guard | stop at 80 °C, continue at 65 °C | reads the NVIDIA GPU with `nvidia-smi` (installed with the NVIDIA driver) before calls to the model, at most every 15 s |
| Power guard | think only while plugged in | Windows power status, Linux `/sys/class/power_supply`, macOS `pmset` |

When it starts, the console says which protections are active and shows the current GPU temperature,
or says that the temperature cannot be read on this machine. Every rest is written in the diary, for
example *"I rested 2.5 minutes, to let the computer cool down: GPU 82°C -> 64°C."* If a sensor cannot
be read, that guard does not block, and the time-based rhythm still protects the machine.

Notes:

* One call to the model cannot be stopped halfway, so the GPU can go slightly above the limit for the
  length of one call before the organism pauses.
* GPUs also slow themselves down near their own maximum, in the high 80s °C. These defaults keep the
  computer well below that. If it rests too often for your taste, raise `max_gpu_temp_c` a little or
  shorten `work_minutes`; if your laptop runs hot, lower them.
* Laptop tips: keep the vents free (not on a bed or a blanket), use a cooling pad for long runs, and
  pick a cooler thermal mode in your laptop's own tool (on a Dell G15: Alienware Command Center).
* `--no-rest` turns all of this off. Use it only for short experiments.

## Growing its own library

A curious reader whose books fall silent goes to the library. With `--web` (or
`organism.librarian.enabled: true`) the organism does the same
(`curiosity_ai/organism/librarian.py`):

1. **Hunger.** After a heartbeat in which the texts addressed only a small part of its expectations
   (informativeness at or below `hunger_informativeness`, 0.34 by default), it decides to look elsewhere.
   It goes at most once every 4 heartbeats for the same question.
2. **A reading wish.** The model names what it wants: up to 2 encyclopedia topics, 1 classic book
   (author and title), and search **keywords** for papers (an invented paper title finds nothing, so
   titles are cut down to keywords). It is told which texts it already owns and which searches found
   nothing before; a search that found nothing is not repeated in the same run.
3. **Acquisition.**
   * **Wikipedia**: the exact article first, then search results ranked by how well their titles fit.
     Disambiguation pages and reference sections are dropped. Stored with CC BY-SA attribution.
   * **Project Gutenberg**: Gutenberg's own catalogue file for programs, searched locally. A book is
     taken only when the author's **family name** and a title word match, so "curiosity" cannot bring in
     "The City Curious" and a first name like "John" cannot bring in a lawyer's diary from 1602. The text
     comes from a mirror, as Gutenberg asks of programs.
   * **Semantic Scholar**, then **arXiv**: paper abstracts. Semantic Scholar is often busy without a free
     API key in `SEMANTIC_SCHOLAR_API_KEY`; arXiv needs no key (one request every three seconds).
   * **Relevance check**: a found text is kept only if it shares at least two words with the question and
     the topic, beyond the words searched for ("Curiosity" the Mars rover shares only its name), and the
     judge, reading its title and beginning, does not rate it unrelated ("Magnetosphere of Saturn"
     shares many words with a question about Earth's inner magnetosphere). Background such as
     "Plasmasphere", which a small judge calls only loosely related, is kept. A title about another
     world, an astrophysical object or a laboratory device (Venus, Saturn, a magnetar, a tokamak, "in
     the solar wind") is refused unless the question or the topic names it: a small judge had let in
     papers on whistlers at Venus, in tokamaks and in magnetar magnetospheres.
   * **Busy is not "nothing found"**: when a source cannot be asked (rate limit, no network), the diary
     and the report say so, instead of claiming that the literature is silent.
4. **Provenance.** Every acquired text goes into `memory/organism/library/` with front matter giving
   its source, license, the question it was acquired for, the heartbeat, and the reason. It is
   indexed at once. The model sees what kind of source a passage comes from (book, encyclopedia
   article, paper abstract, or shared by a human), and citations show it: `[BOOK:...]`, `[WIKI:...]`,
   `[PAPER:...]`.

**Politeness.** It sends a User-Agent with contact details (`librarian.contact`, as Wikimedia and
Gutenberg ask), waits at least one second between requests to the same site, follows `Retry-After`
when a site says it is busy (and waits at least five seconds), rests a busy site for a while, and
keeps quotas **per hour** (12 articles, 2 books, 10 abstracts by default), so a long life keeps reading.
Nothing is downloaded twice.

You can also add books yourself: `python live.py --add-book "Hobbes Leviathan"` downloads the best
match into `data/philosophy_corpus/`. `scripts/download_real_corpus.py` downloads every work in
`data/real_corpus_manifest.json`.

## Sessions, reports and experiments

Every run that lets it think is a **session**. When the session ends (after `--minutes`, after
`--heartbeats`, or with Ctrl+C), it writes `memory/organism/sessions/<id>/`:

* `report.md`: a summary (predictions and Brier score, what the judge accepted, how the debates went,
  how close it stayed to its topic), its theory before and after, grounded beliefs with citations,
  doubted beliefs, the questions it gave birth to, library visits, and a table of every heartbeat;
* `metrics.json`: the numbers (definitions in [RESEARCH.md](RESEARCH.md));
* `config.json`: the exact settings, for reproducibility;
* `judge_check.csv`: 20 random decisions of the judge, with an empty column for your own verdict.
  `python scripts/judge_agreement.py <file>` then says how often the judge agreed with you.

It also rewrites `research_map.md` in the life's folder: open questions by pull, hypotheses with the
grounded beliefs they rest on and the predictions the texts contradicted, surprises, possible gaps, doubted beliefs and a reading list.

**The exam.** `--exam` gives it a fixed exam before and after the session
(`data/exams/philosophy_of_curiosity.json`: 21 questions whose answers stand in the corpus, each with
the file and lines). It answers from its own notes only, and a blind grader compares each answer with
the reference. The **grounded score** counts only answers that cite a belief backed by a verified
quote, so it measures what the organism learned by reading, not what the language model already knew.

For experiments, `--policy random` and `--policy novelty` replace the curiosity drive with baselines,
`--home` gives each condition its own life, and `scripts/compare_sessions.py` puts sessions side by
side. [RESEARCH.md](RESEARCH.md) has research questions and a 1-hour protocol.

## What "learning" means here

1. **Knowledge** (working now): beliefs with verified quotes, a question graph, answers with
   confidence. Everything persists in `memory/organism/mind.json`. The exam measures it.
2. **Character** (working now): the temperament changes with the organism's own experience.
3. **Weights** (the next step, not automated): every grounded episode is written to
   `memory/organism/experience/sft.jsonl`. Every episode where the organism was demonstrably wrong
   (a *verified* contradiction) and changed its answer is written to `preference_pairs.jsonl`, with
   the corrected answer as `chosen` and the earlier answer as `rejected`. These are standard formats
   for supervised fine-tuning and DPO (e.g. with Hugging Face TRL or Unsloth), and the resulting LoRA
   adapter can be loaded into Ollama. Training needs a GPU, an evaluation set, and care: a model that
   trains on its own words can degrade. The verified-quote filter exists for exactly this reason.

## Running it

```bash
python live.py                                        # live 5 heartbeats (heartbeats_per_run)
python live.py --minutes 60 --web                     # a 1-hour session that can grow its library; ends with a report
python live.py --heartbeats 20
python live.py --forever                              # keep living until Ctrl+C (the mind is saved every heartbeat)
python live.py --forever --pause 30                   # ...with a longer breath between heartbeats
python live.py --ask "Can a machine be curious, or only act as if it were?"
python live.py --feed-file my_notes.txt --title "My notes on boredom"
python live.py --status                               # open questions with their pulls, beliefs, temperament, theory
python live.py --reflect                              # reflect now
python live.py --model qwen2.5:7b                     # try another Ollama model
python live.py --new-life                             # archive this life and start a new one
python live.py --add-book "Augustine Confessions"     # download a public-domain book into the corpus
python live.py --check --web                          # check the computer and the online sources
python live.py --home memory/exp/random --policy random --minutes 60   # a baseline condition for an experiment
python live.py --minutes 60 --exam                    # take the exam before and after the session
python live.py --exam                                 # just take the exam now
python live.py --judge-model qwen2.5:14b              # a separate, larger model as the blind judge
python live.py --no-judge                             # the organism grades itself (only for ablation experiments)
python live.py --map                                  # write research_map.md now
python live.py --topic "How do lithium-ion batteries age?" --papers ~/papers --minutes 60 --web   # researcher mode
```

Files, in `memory/organism/`:

| File | Contents |
|---|---|
| `diary.md` | The readable life: why each question was chosen, what was expected and found, the inner dialogue, what changed |
| `mind.json` | The whole mind: questions, beliefs, temperament, self-model, vital signs |
| `episodes.jsonl` | One record per heartbeat |
| `inbox/` | What humans have shared (it also becomes part of the library) |
| `experience/` | Training data from grounded episodes |
| `library/` | Texts it acquired itself (`encyclopedia/`, `books/`, `papers/`) and `acquisitions.jsonl`, its reading history |
| `sessions/` | One folder per session: `report.md`, `metrics.json`, `config.json`, `judge_check.csv` |
| `research_map.md` | What it has found so far, on one page |
| `exams/` | Every exam it took: answers, grades, scores |
| `papers/` | Researcher mode: your papers, converted to text |

Run one process per organism home at a time.

## Limits, honestly

* This is a research prototype, not a conscious being. "Curious" means that its behavior is driven by
  measured gaps, surprise and learning progress, not by a script or by a human choosing every topic.
* The language model is the ceiling on the quality of each thought. The loop keeps a small model
  honest and moving; it does not make it brilliant. A stronger model (`--model`) will think better.
* BM25 matches words, not meanings. "search" does not find "enquire" (some old spellings are
  normalized). `retrieval: chroma` switches to v7's vector memory.
* Quote verification proves that the words are in the text. Whether they **support the claim** is now
  decided by the blind judge, which is better than the organism grading itself but is still a language
  model. Check it with `judge_check.csv`, and use a larger judge model when you can.
* The topic anchor keeps it close to its topic, which is what you want for research but can also stop
  a genuinely interesting side question. Lower `topic_anchor` to give it more freedom.
* Confidence is still the model's own estimate, but it is bounded and tied to evidence.
* With `trace_llm: true` every prompt and raw reply is written to `llm_trace.jsonl`. Read it when
  tuning prompts for your model.

## v7 and v8 side by side

| | v7 pipeline | v8 organism |
|---|---|---|
| Who chooses the next topic | a Planner prompt | the curiosity drive, computed from the mind |
| Quality signal | LLM judges grade depth (0-1) | prediction error, verified quotes, change of belief |
| Memory | journal + theory text rewritten each iteration | persistent beliefs and questions with evidence and history |
| When stuck | forced revision of the same topic | boredom, dormancy, incubation, "unanswerable" |
| LLM calls per step | about 16, with long prompts | 9 (10 when it must restate hedged predictions), with short prompts |
| Self-improvement | none | homeostasis of temperament; its own theory of curiosity |
| Literature | fixed corpus; web snippets per iteration | a library that grows when its books fall silent, with provenance |
| Evaluation | judge scores | session reports, metrics, a blind judge that can be checked by hand, an outside exam, baselines (`--policy`), comparison script |

v7 (`run.py`) is untouched and still works.

## References

* Berlyne, D. E. (1960). *Conflict, Arousal, and Curiosity*. McGraw-Hill.
* Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.
* Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. A. (2018). Large-scale study of curiosity-driven learning. arXiv:1808.04355.
* Dewey, J. (1910). *How We Think*. D. C. Heath.
* Gottlieb, J., Oudeyer, P.-Y., Lopes, M., & Baranes, A. (2013). Information-seeking, curiosity, and attention: computational and neural mechanisms. *Trends in Cognitive Sciences*, 17(11), 585-593.
* Inan, I. (2012). *The Philosophy of Curiosity*. Routledge.
* James, W. (1890). *The Principles of Psychology*, ch. 24.
* Kang, M. J., et al. (2009). The wick in the candle of learning: epistemic curiosity activates reward circuitry and enhances memory. *Psychological Science*, 20(8), 963-973.
* Kidd, C., & Hayden, B. Y. (2015). The psychology and neuroscience of curiosity. *Neuron*, 88(3), 449-460.
* Loewenstein, G. (1994). The psychology of curiosity: a review and reinterpretation. *Psychological Bulletin*, 116(1), 75-98.
* Mill, J. S. (1859). *On Liberty*, ch. 2.
* Oudeyer, P.-Y., Kaplan, F., & Hafner, V. V. (2007). Intrinsic motivation systems for autonomous mental development. *IEEE Transactions on Evolutionary Computation*, 11(2), 265-286.
* Park, J. S., et al. (2023). Generative agents: interactive simulacra of human behavior. *UIST 2023*.
* Peirce, C. S. (1868). Some consequences of four incapacities. *Journal of Speculative Philosophy*, 2, 140-157.
* Peirce, C. S. (1877). The fixation of belief. *Popular Science Monthly*, 12, 1-15.
* Popper, K. (1963). *Conjectures and Refutations*. Routledge.
* Rafailov, R., et al. (2023). Direct preference optimization: your language model is secretly a reward model. *NeurIPS 2023*.
* Schmidhuber, J. (2010). Formal theory of creativity, fun, and intrinsic motivation (1990-2010). *IEEE Transactions on Autonomous Mental Development*, 2(3), 230-247.
* Wallas, G. (1926). *The Art of Thought*. Harcourt Brace.
* Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). STaR: bootstrapping reasoning with reasoning. *NeurIPS 2022*.
