# Changelog v6 — Aggressive Depth Mode

v6 is designed to move the system from a source-grounded research-loop MVP toward a more serious philosophical reasoning engine.

## New architecture modules

1. **Radical Question Agent**
   - Generates dangerous but defensible questions and theses.
   - Forces paradoxes, dilemmas, contradictions, reversals, and hidden assumptions.

2. **Strong Position Enforcer**
   - Prevents endless safe hedging.
   - Forces the system to defend one of: curiosity as virtue, method, danger, conditional transformation, or revisionary concept.

3. **Real Philosophical Move Detector**
   - Requires at least one real philosophical move per iteration.
   - Detects distinction, paradox, contradiction, dilemma, reversal, genealogy, hidden assumption, objection/reply, or criterion-setting.
   - Adds deterministic hedging analysis.

4. **Curiosity Theory Tracker**
   - Maintains `memory/curiosity_theory_state_v6.json`.
   - Tracks the evolving definition of curiosity, discovered properties, unresolved tensions, and the next required move.

5. **Forced Revision Gate**
   - If depth, curiosity progress, or real-move score is too low, the system revises before planning onward.
   - If still weak, the next topic becomes a revision-focused version of the same question.

## Scoring redesign

- Novelty and researchability are now less dominant.
- Philosophical depth, genuine tension, argument quality, source interpretation, curiosity progress, and hedging penalties matter more.
- Dataset export now requires: depth pass, real philosophical move pass, strong position, evaluator approval, and high final score.

## Data upgrades

- Added `curiosity_core_reader.md`, a curated retrieval aid built from the included public-domain primary texts.
- Expanded corpus manifest with additional Plato inquiry texts: *Theaetetus*, *Meno*, and *Apology*.
- Added `data/evaluation/curiosity_benchmark_topics.json` for fixed evaluation topics.
- Added `scripts/run_benchmark.py` for repeatable benchmark runs.

## Recommended run command

```bash
python scripts/reset_memory.py
python run.py --topic "Is curiosity a virtue, a method, or a danger to certainty?" --iterations 3
```

Open:

```text
reports/<run_id>/discussion_transcript.md
memory/curiosity_theory_state_v6.json
```
