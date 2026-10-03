# Doing research with the Curiosity Organism

The organism makes curiosity **measurable**. Every session writes a report and a `metrics.json`
(see [ORGANISM.md](ORGANISM.md)), the way it chooses questions can be switched for baselines, and
each experimental condition can live in its own folder. This makes controlled experiments possible
in **1-hour sessions on a laptop**.

## Research questions you can test now

**RQ1. Does curiosity-driven choice make a language-model agent learn more than random or
novelty-driven choice?**
Compare `--policy curiosity`, `--policy random` and `--policy novelty` (least-visited first).
*Expected:* curiosity yields more grounded beliefs per hour and a larger drop in surprise on revisited
questions. Novelty visits more distinct questions but learns less from each one: the restless
curiosity that Augustine and Heidegger describe, and the "noisy TV" problem in AI research.

**RQ2. Does growing its own library help?**
Compare sessions with and without `--web`.
*Expected:* with the web library, the texts address more of its expectations (informativeness) in the
second half of the session than in the first, and fewer questions end up dormant or unanswerable.

**RQ3. How does model size change the quality of curiosity?**
Run the same protocol with different models (`--model qwen2.5:3b`, `mistral:7b-instruct`,
a 14B model, and so on).
*Expected:* larger models invent fewer quotes (fabrication rate), form more grounded beliefs, and
doubt their earlier beliefs more often and more appropriately.

**RQ4. What theory of curiosity does it reach, and is it faithful to what it read?** (qualitative)
Each report shows its theory of curiosity before and after the session, and the grounded beliefs with
citations. Compare the theories reached by different lives and models, and check them against the
texts.

**RQ5. Does self-regulation work?** (needs longer runs of several hours)
Do the diagnoses (healthy wonder, restless curiositas, dogmatic slumber, aporetic numbness) match what
the diary shows, and do the temperament changes move it out of stuck states?
*A first answer from the v8 hour:* no. A 7B model left alone fell into exactly the vice the
philosophers describe (113 new questions in 85 heartbeats, none settled, drifting to "adaptive learning
strategies for diverse cultural contexts"), and all 17 self-checks said "healthy wonder", because any
confidence bump counted as progress. v9 counts only evidence the judge accepted. Whether that fixes the
diagnosis is now testable: a finding either way.

**RQ6. Does an agent that grades itself measure itself wrongly?**
Compare runs with the blind judge (default) and without it (`--no-judge`), and check a sample of both
by hand (`judge_check.csv`).
*Expected:* without the judge, grounded beliefs, confirmations and confidence are inflated (in the v8
hour, half of a hand-checked sample of "confirmations" did not support the claim at all).

**RQ7. Can a curious agent find where a research field went next?** (researcher mode, see
[RESEARCHER.md](RESEARCHER.md))
Give it only papers up to a year (`--until-year 2020`), let it work, and compare its top questions and
"possible gaps" with what was published afterwards. Compare `--policy curiosity` with `random`.

## A 1-hour experiment, step by step

Change one thing only and keep everything else the same: model, seed questions, `--seed`, corpus,
and body-care settings. Give every condition a **fresh life of its own** with `--home`.

```bash
python live.py --home memory/exp/curiosity-1 --policy curiosity --minutes 60 --seed 1 --label curiosity --exam
python live.py --home memory/exp/random-1    --policy random    --minutes 60 --seed 1 --label random --exam
python scripts/compare_sessions.py memory/exp/curiosity-1/sessions/* memory/exp/random-1/sessions/*
```

`compare_sessions.py` prints a Markdown table (add `--csv results.csv` to save it). `--exam` gives
the exam before and after each session (exam time does not count toward the 60 minutes). Then:

* **Check the judge first.** Open each session's `judge_check.csv`, write your own verdict in the last
  column (supports, contradicts or neither) and run
  `python scripts/judge_agreement.py memory/exp/*/sessions/*/judge_check.csv`. If the judge agrees with
  you much less than about 80% of the time, use a larger judge model (`--judge-model`) before trusting
  any number that depends on it.

* **Repeat each condition at least 3 times** with different seeds (`--seed 2`, `--seed 3`). A
  language model's answers vary, so one run proves nothing.
* Report the mean and the spread of each metric, not just the best run.
* Read the diaries (`memory/exp/*/diary.md`) for concrete examples: a doubted belief, a surprising
  passage, a question born from a contradiction.

Your first hour with v8 gave 85 heartbeats with mistral 7B on an RTX 4060. v9 asks the model three
more short things per heartbeat (the judge's checks), so expect roughly 55-65 heartbeats per hour.

## What the numbers mean

| Metric (`metrics.json`) | Meaning |
|---|---|
| `heartbeats`, `heartbeats_per_hour` | acts of inquiry completed |
| `rest_minutes`, `cooling_rests` | time spent protecting the computer |
| `distinct_questions` | how many different questions it visited |
| `questions_born`, `questions_born_by_trigger` | new questions, by what caused them (surprise, contradiction, gap, objection, ...) |
| `beliefs_new`, `beliefs_grounded`, `beliefs_interpretive` | beliefs formed; grounded ones carry a quote verified in a source |
| `beliefs_doubted` | earlier beliefs it lowered after contradicting evidence: self-correction |
| `quotes_verified`, `quotes_rejected`, `fabrication_rate` | honesty: claimed quotes found / not found in the sources |
| `mean_surprise`, `surprise_first_half`, `surprise_second_half` | how much the texts went against its predictions |
| `mean_informativeness` | how much of its expectations the texts addressed at all |
| `revisited_questions`, `surprise_change_on_revisits` | on questions visited more than once: negative means it now predicts the texts better (learning) |
| `mean_confidence_change` | how much its answers moved per heartbeat |
| `library_visits`, `acquisitions`, `acquisitions_by_kind` | how often its books were silent, and what it fetched |
| `library_rejected`, `library_busy` | texts found but off topic (not kept); searches impossible because a source was busy |
| `diagnoses` | its self-diagnoses during the session |
| `predictions`, `hedged_rate` | predictions made; the share still hedged ("may", "might") after it was asked to restate them |
| `predictions_confirmed`, `predictions_contradicted` | what the texts confirmed or contradicted, as the judge accepted |
| `mean_brier`, `brier_first_half`, `brier_second_half` | Brier score of the addressed predictions (0 is perfect, 0.25 is a coin flip); falling means it predicts the texts better |
| `judge_pairs`, `judge_agreement`, `judge_rejected` | claim/quote pairs the judge saw; how often it agreed with the organism's own claim; quotes it found beside the point |
| `stances`, `concession_rate` | how Wonder answered the skeptic: defended, revised or conceded |
| `skeptic_quoted_rate` | share of debates where the skeptic quoted a real passage |
| `vague_answers` | answers too vague to be wrong (no confidence gained) |
| `mean_on_topic`, `questions_set_aside` | closeness to the topic of the questions it worked on (0-1); proposed questions set aside as off topic |
| `exam_before`, `exam_after`, `exam_gain` | exam score before and after the session (partly correct counts half) |
| `exam_grounded_before`, `exam_grounded_after` | the same, counting only answers backed by a verified quote: what it learned by reading |

Rates per hour make runs of different length comparable; `compare_sessions.py` adds grounded beliefs
per hour and questions born per hour.

## Threats to validity (say these in a paper)

* **A model judges the evidence.** Quote verification proves that the quoted words exist in a source.
  Whether they support the claim is decided by the blind judge, which is stricter than the organism
  grading itself but is still a language model. Report the judge's agreement with your own hand-check.
* **The exam is small.** 21 questions about a fixed corpus measure one kind of learning. Report it with
  the other metrics, not alone.
* **Small samples.** One hour is about 50 heartbeats. Use several seeds per condition.
* **The corpus shapes the results.** Report which books were in the library; with `--web`, the
  library changes during the run (each report lists what was acquired).
* **Randomness.** `--seed` fixes the organism's own choices, not the language model's text. Ollama
  samples with temperature, so runs differ even with the same seed.
* **Time and compute.** Rests and GPU speed change heartbeats per hour; compare per heartbeat as well.
* **Prompt sensitivity.** Different models follow the prompts differently. Keep `trace_llm: true`
  in a pilot run and read `llm_trace.jsonl`.

## Running it on a computing cluster (for example VALAR)

A cluster is worth it for what a laptop cannot do: larger models (32B-70B), a larger judge model, and
many seeds and conditions in parallel. Three practical points:

* **Freeze the library.** Compute nodes often have no internet. Run without `--web`, so every condition
  reads exactly the same books; that is also better science. Researcher mode can use a fixed `--papers`
  folder.
* **Compare by heartbeats, not minutes.** Models of different sizes think at different speeds:
  `--heartbeats 100` gives every condition the same amount of inquiry.
* **One life per folder.** Use `--home` per condition and seed, for example
  `--home runs/qwen32b-curiosity-seed3`, and collect all `sessions/*/metrics.json` with
  `compare_sessions.py --csv`.

Ollama runs on a cluster too (a single program; point `OLLAMA_MODELS` to your project storage). A
server that runs many lives on one GPU at once (such as vLLM) would be faster; it is not built in yet.

## Ideas for extending the research

* Ablation switches: no prediction before reading, no inner dialogue, no evidence ceiling. Which
  part matters most?
* The "sleep" step: fine-tune a new model generation on `experience/` data, then test whether it
  predicts the texts better on questions it has not seen.
* Blind human ratings of insights from curiosity versus random runs.
* Long lives (days): memory consolidation and forgetting.

## Related work

* Intrinsic motivation and curiosity in machines: Oudeyer, Kaplan & Hafner (2007); Schmidhuber (2010);
  Pathak et al., *Curiosity-driven exploration by self-supervised prediction* (ICML 2017);
  Burda et al., *Large-scale study of curiosity-driven learning* (2018).
* Autotelic and open-ended agents: Colas et al., *Autotelic agents with intrinsically motivated
  goal-conditioned reinforcement learning: a short survey* (JAIR 2022); Zhang et al., *OMNI:
  Open-endedness via models of human notions of interestingness* (2023); Wang et al., *Voyager* (2023).
* Language agents with memory and reflection: Park et al., *Generative agents* (2023); Shinn et al.,
  *Reflexion* (2023). Automated research: Lu et al., *The AI Scientist* (2024).
* Psychology of curiosity: Loewenstein (1994); Kang et al. (2009); Gottlieb et al. (2013); Kidd &
  Hayden (2015).
* Philosophy of curiosity: Inan, *The Philosophy of Curiosity* (2012); Whitcomb, *Curiosity was
  framed* (Philosophy and Phenomenological Research, 2010); and the primary texts in the corpus
  (Plato, Augustine, Hobbes, Hume, Kant, Peirce, James, Dewey).
