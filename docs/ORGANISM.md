# The Curiosity Organism (v8)

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
      +------------- reflection, every few heartbeats ---------------+
      |  vital signs -> diagnosis -> adjust temperament -> theory     |
      v                                                               |
  choose --> predict --> observe --> compare --> argue --> settle --> learn
  (drive)    (commit     (read the   (surprise,   (Wonder    (answer,   (progress,
              first)      library)    verified     vs         beliefs,   boredom,
                                      quotes)      Skeptic)   new        experience
                                                              questions)  data)
```

| Step | What happens | Root |
|---|---|---|
| choose | Every open question gets a *pull* from the curiosity drive. A softmax picks one, so the organism mostly follows the strongest pull and sometimes wanders. | Loewenstein, Oudeyer, Berlyne (below) |
| predict | Before reading, the organism writes its current answer, its confidence, and 2-4 expectations that a text could prove wrong. | Peirce: real doubt needs a real expectation |
| observe | BM25 search over `data/philosophy_corpus` and anything a human shared. Optionally the web, with off-topic results dropped. | Dewey: curiosity seeks "material for thought" |
| compare | For each expectation: confirmed, contradicted, or not addressed. A claim counts only if its quote is **found in the passage**. *Informativeness* is how much the texts addressed the expectations; *surprise* is how much of what was predicted they went against (see below). | Peirce's "irritation of doubt" |
| argue | Wonder proposes a bold explanation; Skeptic attacks its weakest point; Wonder answers. | Socratic elenchus; Dewey's "suggestion" and "reasoning" steps |
| settle | Revised answer and confidence, new beliefs (grounded or marked as interpretation), doubts about old beliefs, and new questions born from surprise, contradiction, gaps or objections. | Dewey's fifth step: "acceptance or rejection" |
| learn | The visit's prediction error and change of confidence are stored. They feed learning progress, boredom, settling and dormancy. Grounded episodes become training data. | Oudeyer, Schmidhuber |

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
| *Curiositas*, the "lust of the eyes"; *studiositas* as its virtuous counterpart. | Augustine, *Confessions* X.35 (`augustine_confessions_pusey.md` l. 6365); Aquinas, *ST* II-II q.166-167 | Diagnosis **restless curiositas**. |
| *Neugier*: curiosity that seeks novelty "not in order to understand" and never dwells. | Heidegger, *Being and Time* section 36 | The main reward is **learning progress, not novelty**, and **boredom** ends unproductive visits. |
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
          + w_importance * importance ] * (1 - boredom) * (1 - refractory)
```

* `gap = 4c(1-c)`, where `c` is the confidence in the current answer.
* `anchoring = 0.4 + 0.6 * min(1, related_beliefs / 3)`.
* `learning_progress`: an optimistic prior before two visits; after that, the fall in prediction error
  (older half of the recent visits minus newer half) plus the average change of confidence.
* `surprise`: the last visit's surprise, fading by 0.85 per heartbeat. A newborn question inherits the
  surprise of the episode that created it. For a visit with `n` expectations, `x` contradicted, `c`
  confirmed and `u` genuinely unexpected findings:

  ```text
  informativeness = (c + x + u/2) / (n + u/2)      how much the texts spoke to my predictions
  surprise        = (x + u/2)     / (n + u/2)      = prediction error x informativeness
  ```

  Texts that stay silent, or only say unrelated things, cannot be very surprising. Learning progress
  uses the prediction error weighted by informativeness, so a silent visit cannot pass for progress.
* `novelty = 1 / (1 + visits)`.
* `boredom`: zero until `boredom_patience` visits without progress, then rising by 0.35 per visit.
* `refractory = 0.5` if the question was visited in the previous heartbeat, to discourage rumination.

The code is in `curiosity_ai/organism/drive.py`. It is plain arithmetic and is unit-tested in
`tests/test_organism_drive.py`.

## Honesty: why it is harder to fool than v7

Your v7 run showed the usual failure modes of a small model judging itself. Every thesis drifted to
"curiosity needs balance", the move detector gave 1.000 to a platitude, and the web search returned
papers on copyright law and medical curricula. v8 does not ask the model to grade its own depth.
Instead:

* **Predictions come before evidence**, so surprise is measured rather than claimed.
* **Quotes are verified.** A confirmation or contradiction counts only if its quote is found in the
  passage that was read. Fabricated quotes are discarded and counted in the diary.
* **Confidence moves in bounded steps**: at most 0.25 per heartbeat, at most +0.05 without evidence,
  and at most +0.10 when heavily surprised.
* **Confidence cannot outrun the evidence**: it can never rise above `0.5 + 0.1 x` the number of
  verified confirmations gathered for that question. Settling (0.8) therefore needs several.
* **The comparison is told to be strict**: a passage on a related topic confirms nothing, and most
  expectations will not be addressed. The same quote cannot count as both expected and unexpected.
* **Beliefs without a verified quote are labelled "interpretation"** and start at lower confidence.
* **Contradictions can only target beliefs the model was actually shown.**
* **Web results must cover at least 25% of the question's content words**; off-topic results are dropped.
* **Citations point to real file lines**, so a human can check them.

## Self-regulation

At each reflection the organism measures its **vital signs** over recent episodes: progress rate (how
often its mind actually moved: confidence changed, an old belief was doubted, or a belief backed by a
verified quote was formed), mean surprise, diversity of questions visited, and new questions per
episode.
It then diagnoses its state and nudges its temperament (the drive weights) in small, bounded steps:

| Diagnosis | Pattern | Response |
|---|---|---|
| Healthy wonder | progress is happening | relax toward the baseline temperament |
| Restless curiositas | many different questions, little progress | value progress over novelty; more patience |
| Dogmatic slumber | almost no surprise, few new questions | raise surprise and novelty; explore more |
| Aporetic numbness | high surprise, little progress, few questions | get bored sooner; prefer anchored questions |

The language model also writes a short reflection and updates the organism's **own theory of
curiosity**. That theory is shown to the Wonder voice at every heartbeat. What the organism comes to
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
   (author and title), and 1 search phrase for papers. It is told which texts it already owns.
3. **Acquisition.**
   * **Wikipedia**: the exact article first, then search results ranked by how well their titles fit.
     Disambiguation pages and reference sections are dropped. Stored with CC BY-SA attribution.
   * **Project Gutenberg**: Gutenberg's own catalogue file for programs, searched locally. A book is
     taken only when both an author and a title word match, so a vague wish such as "curiosity"
     cannot bring in "The City Curious". The text comes from a mirror, as Gutenberg asks of programs.
   * **Semantic Scholar**: paper abstracts, optional. Without a free API key in
     `SEMANTIC_SCHOLAR_API_KEY` it is often busy, and is then skipped quietly.
4. **Provenance.** Every acquired text goes into `memory/organism/library/` with front matter giving
   its source, license, the question it was acquired for, the heartbeat, and the reason. It is
   indexed at once. The model sees what kind of source a passage comes from (book, encyclopedia
   article, paper abstract, or shared by a human), and citations show it: `[BOOK:...]`, `[WIKI:...]`,
   `[PAPER:...]`.

**Politeness.** It sends a User-Agent with contact details (`librarian.contact`, as Wikimedia and
Gutenberg ask), waits at least one second between requests to the same site, follows `Retry-After`
when a site says it is busy (and waits at least five seconds), rests a busy site for a while, and
keeps quotas per run (12 articles, 3 books, 8 abstracts by default). Nothing is downloaded twice.

You can also add books yourself: `python live.py --add-book "Hobbes Leviathan"` downloads the best
match into `data/philosophy_corpus/`. `scripts/download_real_corpus.py` downloads every work in
`data/real_corpus_manifest.json`.

## Sessions, reports and experiments

Every run that lets it think is a **session**. When the session ends (after `--minutes`, after
`--heartbeats`, or with Ctrl+C), it writes `memory/organism/sessions/<id>/`:

* `report.md`: a summary, its theory of curiosity before and after, grounded beliefs with citations,
  doubted beliefs, the questions it gave birth to, library visits, and a table of every heartbeat;
* `metrics.json`: the numbers (definitions in [RESEARCH.md](RESEARCH.md));
* `config.json`: the exact settings, for reproducibility.

For experiments, `--policy random` and `--policy novelty` replace the curiosity drive with baselines,
`--home` gives each condition its own life, and `scripts/compare_sessions.py` puts sessions side by
side. [RESEARCH.md](RESEARCH.md) has research questions and a 1-hour protocol.

## What "learning" means here

1. **Knowledge** (working now): beliefs with verified quotes, a question graph, answers with
   confidence. Everything persists in `memory/organism/mind.json`.
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
| `sessions/` | One folder per session: `report.md`, `metrics.json`, `config.json` |

Run one process per organism home at a time.

## Limits, honestly

* This is a research prototype, not a conscious being. "Curious" means that its behavior is driven by
  measured gaps, surprise and learning progress, not by a script or by a human choosing every topic.
* The language model is the ceiling on the quality of each thought. The loop keeps a small model
  honest and moving; it does not make it brilliant. A stronger model (`--model`) will think better.
* BM25 matches words, not meanings. "search" does not find "enquire" (some old spellings are
  normalized). `retrieval: chroma` switches to v7's vector memory.
* Quote verification proves that the words are in the text, **not that they support the claim**.
  Whether a quote confirms an expectation is still the model's judgment. Small models say "confirmed"
  too easily, which is why confidence is capped by the amount of evidence.
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
| LLM calls per step | about 16, with long prompts | 6, with short prompts |
| Self-improvement | none | homeostasis of temperament; its own theory of curiosity |
| Literature | fixed corpus; web snippets per iteration | a library that grows when its books fall silent, with provenance |
| Evaluation | judge scores | session reports, metrics, baselines (`--policy`), comparison script |

v7 (`run.py`) is untouched and still works.

## References

* Berlyne, D. E. (1960). *Conflict, Arousal, and Curiosity*. McGraw-Hill.
* Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. A. (2018). Large-scale study of curiosity-driven learning. arXiv:1808.04355.
* Dewey, J. (1910). *How We Think*. D. C. Heath.
* Gottlieb, J., Oudeyer, P.-Y., Lopes, M., & Baranes, A. (2013). Information-seeking, curiosity, and attention: computational and neural mechanisms. *Trends in Cognitive Sciences*, 17(11), 585-593.
* Inan, I. (2012). *The Philosophy of Curiosity*. Routledge.
* James, W. (1890). *The Principles of Psychology*, ch. 24.
* Kang, M. J., et al. (2009). The wick in the candle of learning: epistemic curiosity activates reward circuitry and enhances memory. *Psychological Science*, 20(8), 963-973.
* Kidd, C., & Hayden, B. Y. (2015). The psychology and neuroscience of curiosity. *Neuron*, 88(3), 449-460.
* Loewenstein, G. (1994). The psychology of curiosity: a review and reinterpretation. *Psychological Bulletin*, 116(1), 75-98.
* Oudeyer, P.-Y., Kaplan, F., & Hafner, V. V. (2007). Intrinsic motivation systems for autonomous mental development. *IEEE Transactions on Evolutionary Computation*, 11(2), 265-286.
* Park, J. S., et al. (2023). Generative agents: interactive simulacra of human behavior. *UIST 2023*.
* Peirce, C. S. (1868). Some consequences of four incapacities. *Journal of Speculative Philosophy*, 2, 140-157.
* Peirce, C. S. (1877). The fixation of belief. *Popular Science Monthly*, 12, 1-15.
* Rafailov, R., et al. (2023). Direct preference optimization: your language model is secretly a reward model. *NeurIPS 2023*.
* Schmidhuber, J. (2010). Formal theory of creativity, fun, and intrinsic motivation (1990-2010). *IEEE Transactions on Autonomous Mental Development*, 2(3), 230-247.
* Wallas, G. (1926). *The Art of Thought*. Harcourt Brace.
* Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). STaR: bootstrapping reasoning with reasoning. *NeurIPS 2022*.
