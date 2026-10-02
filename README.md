# Curiosity AI

A local-first research system that tries to make an AI **curious**, not only able to talk about curiosity.

## v8: the Curiosity Organism

v1 to v7 ask a language model to write about curiosity and then grade the writing. v8 adds an
organism whose **behavior is driven by curiosity**:

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

```bash
pip install -r requirements.txt
ollama pull mistral:7b-instruct
python live.py                      # live 5 heartbeats
python live.py --status             # look inside its mind
python live.py --ask "Can a machine be curious, or only act as if it were?"
python live.py --forever --pause 30 # let it live until Ctrl+C
```

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
