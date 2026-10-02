# v3 Changelog

## Main upgrades

- Added visible multi-agent transcript files:
  - `discussion_001.md`, `discussion_002.md`, ...
  - `discussion_transcript.md`
- Added transcript links inside every iteration report and the final report.
- Strengthened Critic Agent prompt so it rejects vague, fake-deep, repetitive, or weakly grounded questions.
- Strengthened Planner Agent prompt so the system does not get trapped only in loop-controller engineering.
- Moved small generated seed notes out of the active corpus folder into `data/seed_examples/`.
- Added `data/real_corpus_manifest.json` with real primary philosophical readings.
- Added `scripts/download_real_corpus.py` to download public-domain Project Gutenberg texts locally.
- Added `scripts/reset_memory.py`.
- Changed Chroma collection prefix to `curiosity_ai_v3` and journal path to `memory/journal_v3.jsonl`.
- Disabled publishing by default to prioritize quality inspection.

## Recommended first command

```bash
python scripts/reset_memory.py
python scripts/download_real_corpus.py --clear-existing
python run.py --topic "Is curiosity a virtue, a method, or a danger to certainty?" --iterations 5
```
