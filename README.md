# Curiosity AI

A local-first research system that tries to make an AI **curious**, not only able to talk about curiosity.

## v9: the Curiosity Organism, now measured honestly, and a researcher mode

v1 to v7 ask a language model to write about curiosity and then grade the writing. v8 added an
organism whose **behavior is driven by curiosity**. v9 makes its measurements trustworthy and lets it
study **your own research topic** (see [CHANGELOG_v9.md](CHANGELOG_v9.md)):

- Predictions must be able to fail: who will say what, with a **probability**, scored with the Brier
  score. Hedges like "may" and "might" are refused.
- A **blind judge** (optionally a larger model) decides what each quote really supports; a sample is
  saved so you can check the judge by hand.
- It **stays on its topic** (the judge rates every new question; at most one is born per heartbeat),
  and its **self-diagnosis** now counts only progress backed by evidence.
- A **real debate**: the Skeptic must bring evidence, Wonder must defend, revise or concede, and an
  answer too vague to be wrong earns no confidence.
- An **outside exam** before and after a session measures what it learned by reading.
- **Researcher mode**: `--topic "..." --papers <folder>` turns it into a curious assistant that reads
  your PDFs and keeps a **research map** of open questions, hypotheses, surprises and gaps. See
  **[docs/RESEARCHER.md](docs/RESEARCHER.md)**.

What v8 brought:

- It keeps a persistent **mind**: open questions, beliefs with evidence, and a temperament.
- A **curiosity drive** chooses what to think about next: the information gap, learning progress,
  surprise, novelty, importance, and boredom.
- It **predicts before it reads**, so the classic texts can surprise it. Surprise and contradiction
  give birth to new questions.
- An **inner dialogue** between Wonder and Skeptic argues over every finding.
- **Only quotes that really appear in the source count as evidence**, and confidence can never rise
  above what the evidence allows.
- Every few heartbeats it **reflects on its own way of being curious**. It diagnoses itself (healthy
  wonder, restless *curiositas*, dogmatic slumber, aporetic numbness) and adjusts its temperament.
- Grounded episodes are exported as **training data** for a future fine-tune.
- It **takes care of your computer**: it rests regularly, stops when the NVIDIA GPU gets hot (read
  with `nvidia-smi`) until it has cooled down, and on a laptop thinks only while plugged in.
- It **grows its own library** (`--web`): when its books are silent on a question, it looks things up
  on Wikipedia, downloads public-domain books from Project Gutenberg, and reads paper abstracts
  (Semantic Scholar, arXiv), and it records why it acquired each text.
- Every session ends with a **report and metrics**, and baselines (`--policy random|novelty`) make
  controlled experiments possible. See **[docs/RESEARCH.md](docs/RESEARCH.md)**.

```bash
pip install -r requirements.txt
ollama pull mistral:7b-instruct
python live.py                      # live 5 heartbeats
python live.py --minutes 60 --web --exam   # a 1-hour session that can grow its library, with the exam before and after
python live.py --topic "How do lithium-ion batteries age?" --papers ~/papers --minutes 60 --web   # researcher mode
python live.py --add-book "Hobbes Leviathan"   # add a public-domain book to the library
python live.py --status             # look inside its mind
python live.py --ask "Can a machine be curious, or only act as if it were?"
python live.py --forever            # let it live until Ctrl+C (it rests to keep the PC cool)
python live.py --check              # check this computer's setup and say what to fix
```

On Windows, run it inside WSL with your NVIDIA GPU: see **[docs/WSL_SETUP.md](docs/WSL_SETUP.md)**.

Then read `memory/organism/diary.md`. **[docs/ORGANISM.md](docs/ORGANISM.md)** explains the design:
every mechanism and the philosophical idea it comes from (Plato, Meno's paradox, Peirce, Dewey,
James, Kant, Augustine, Heidegger, and the psychology of curiosity).

Run the tests with `python -m pytest`. They need no Ollama.

The v7 pipeline below (`run.py`) is unchanged.

---

# Curiosity AI v6 — Aggressive Depth Philosophical Loop

This is a local-first closed-loop philosophical AI system designed for **autonomous curiosity-driven inquiry**, not normal ask-answer chat.

v6 adds **Aggressive Depth Mode**: the system now generates radical questions, forces a strong philosophical position, detects whether a real philosophical move was made, penalizes excessive hedging, tracks a cumulative theory of curiosity, and revises weak outputs instead of simply moving on.

## Core loop

```text
Topic
↓
Source Gate
↓
Reader / Grounding Agent
↓
Explorer Agent
↓
Radical Question Agent
↓
Question Critic
↓
Research Agent
↓
Textual Analyst
↓
Argument Reconstructor
↓
Strong Position Enforcer
↓
Deep Synthesizer
↓
Argument Critic
↓
Philosophical Depth Judge
↓
Real Philosophical Move Detector
↓
Forced Revision Gate
↓
Curiosity Theory Tracker
↓
Evaluator
↓
Planner
↓
Memory / Dataset / Reports
```

## Why v6 is different

Previous versions could produce good research skeletons but still sounded safe and mechanical. v6 explicitly tries to avoid that by requiring:

- a bold but defensible thesis,
- one real philosophical move per iteration,
- less hedging,
- direct source interpretation,
- a strong objection and reply,
- a cumulative update to the theory of curiosity,
- forced revision if quality is not high enough.

## Install

```bash
cd curiosity_ai_project_v6
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull mistral:7b-instruct
```

## Run

Start fresh:

```bash
python scripts/reset_memory.py
python run.py --topic "Is curiosity a virtue, a method, or a danger to certainty?" --iterations 3
```

Open the generated files:

```text
reports/<run_id>/discussion_transcript.md
reports/<run_id>/iteration_001.md
memory/curiosity_theory_state_v6.json
```

## Important files

```text
curiosity_ai/agents.py          # agent implementations
curiosity_ai/controller.py      # loop + forced revision logic
curiosity_ai/schema.py          # output schemas
curiosity_ai/prompts.py         # agent system prompts
curiosity_ai/scoring.py         # heuristic question scoring
config.yaml                     # thresholds and mode settings
```

## Corpus

The active corpus is in:

```text
data/philosophy_corpus/
```

Included public-domain / Project Gutenberg style texts include:

- Plato — *Republic*
- Aristotle — *Nicomachean Ethics*
- Descartes — *Discourse on Method*
- Hume — *Enquiry Concerning Human Understanding*
- Kant — *Critique of Pure Reason*
- Nietzsche — *Beyond Good and Evil*
- Dewey — *How We Think*
- Confucius — *Analects*, Legge translation
- Curated retrieval aid: `curiosity_core_reader.md`

The manifest also includes recommended additional primary texts, including Plato's *Theaetetus*, *Meno*, and *Apology*. Try:

```bash
python scripts/download_real_corpus.py
```

The downloader preserves Gutenberg/source text. Verify copyright/public-domain status in your jurisdiction before redistribution.

## Benchmark

Run a small benchmark over fixed topics:

```bash
python scripts/run_benchmark.py --limit 3 --iterations 2
```

Benchmark topics live at:

```text
data/evaluation/curiosity_benchmark_topics.json
```

## Dataset generation

The system exports SFT examples only when the output is strong enough:

```text
datasets/curiosity_sft_v6.jsonl
```

v6 requires:

- philosophical depth pass,
- real philosophical move pass,
- strong position present,
- evaluator approval,
- score above threshold.

Do not fine-tune until you have hundreds of high-quality examples.

## Recommended development path

1. Run 3-iteration experiments.
2. Read `discussion_transcript.md`.
3. Check `Philosophical Depth Judge` and `Real Philosophical Move Detector`.
4. Tune thresholds in `config.yaml`.
5. Add targeted primary texts.
6. Build a benchmark table.
7. Only then consider QLoRA fine-tuning.

---

# v7 Notes — Serious Depth Controls

v7 adds three important protections:

1. **Argument Integrity Gate:** the system cannot display an empty conclusion / assumptions / gaps / objection in the Argument Reconstructor section. If the local LLM omits them, the code fills conservative repair statements and forces the synthesis to treat the argument as incomplete.

2. **Balance Stress Tester:** if the system says “balanced approach” or “tempered curiosity,” it must explain exactly what is balanced, who decides the balance, when curiosity becomes a virtue, when it becomes dangerous, and what fails when balance fails.

3. **Hybrid research mode:** `config_hybrid_web.yaml` enables web/academic search. Tavily is used if `TAVILY_API_KEY` is set. Crossref and Semantic Scholar metadata can also be searched for secondary literature discovery.

Current local-only runs show `Web sources: 0` by design. To enable web/academic research, run:

```bash
python run.py --config config_hybrid_web.yaml --topic "Is curiosity a virtue, method, or danger?" --iterations 3
```

For best quality, use web sources as secondary/contextual support and primary texts as the basis of philosophical interpretation.
