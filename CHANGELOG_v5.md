# CHANGELOG v5 — Philosophical Depth Control

This version responds to the v4 evaluation: the system had source grounding and visible debate, but still produced outputs that were sometimes too mechanical, shallow, or drifted away from the central curiosity theme.

## Added

- **Philosophical Depth Judge Agent**
  - Scores each iteration on:
    1. concept definition
    2. anachronism control
    3. genuine philosophical tension
    4. counterargument strength
    5. source interpretation
    6. curiosity-theme progress
  - Produces a strict final depth score and verdict: `pass`, `revise`, or `block`.

- **Curiosity Theme Preservation**
  - The planner is now instructed to keep returning to curiosity, wonder, inquiry, doubt, learning, or knowledge-seeking.
  - The depth judge penalizes generic philosopher comparisons that do not explain how the curiosity project advanced.

- **Score Calibration**
  - Dataset export now uses the minimum of curiosity score, evaluator score, and philosophical depth score.
  - Weak depth judgments prevent publishing and dataset export by default.
  - The evaluator score is capped when philosophical depth is weak.

- **Report Visibility**
  - Iteration reports and discussion transcripts now include a new `Philosophical Depth Judgment` section.
  - Terminal output now prints depth diagnostics after synthesis/critique.

## Changed

- Deep synthesis target length increased from ~1200 to ~1500 words.
- Dataset export now requires philosophical depth pass by default.
- Memory collection changed to `curiosity_ai_v5`.
- Journal changed to `memory/journal_v5.jsonl`.

## Why this matters

The v4 CuriosityScore measured whether a question was promising. It did not reliably measure whether the produced philosophical answer was actually deep. v5 separates those two tasks:

- `CuriosityScore` = question promise / research potential
- `PhilosophicalDepthJudgment` = actual philosophical quality of the produced discussion
- `Evaluation` = final publish/continue decision

This should reduce false confidence, superficial synthesis, and drift away from the central curiosity theme.
