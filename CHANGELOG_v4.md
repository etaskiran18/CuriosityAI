# Changelog v4 — Deep Philosophy Mode

This version upgrades the system from a research-agenda generator toward a deeper philosophical reasoning engine.

## Added

- **Source Gate** before synthesis.
  - Counts primary local passages, memory passages, and web sources.
  - Warns when the system is relying on memory instead of primary texts.

- **Textual Analyst Agent**.
  - Performs close reading of retrieved passages.
  - Extracts literal claim, key concepts, relevance, limitations, and anachronism risk.

- **Argument Reconstructor Agent**.
  - Converts the inquiry into premises, conclusion, hidden assumptions, inferential gaps, objection, and reply.

- **Deep Synthesis Mode**.
  - Writes long-form Markdown reports with required sections: problem framing, textual evidence, conceptual distinction, reconstructed argument, objection, reply, inquiry update, unresolved tensions, next directions.

- **Argument Critic Agent**.
  - Attacks the synthesis after it is written.
  - Can accept, revise, or block a synthesis.
  - Detects unsupported claims, weak citations, logical gaps, anachronism, and shallow summaries.

- **Revision Pass**.
  - If the argument critic says `revise`, the synthesizer rewrites the report.

- **Real public-domain corpus included**.
  - Plato — The Republic
  - Aristotle — Nicomachean Ethics
  - Descartes — Discourse on Method
  - Hume — Enquiry Concerning Human Understanding
  - Nietzsche — Beyond Good and Evil
  - Dewey — How We Think
  - Confucius — Analects, Legge translation

## Changed

- Default memory collection prefix is now `curiosity_ai_v4`.
- Default dataset export path is now `datasets/curiosity_sft_v4.jsonl`.
- Publishing remains disabled by default.
- Curiosity scoring now penalizes memory-only grounding more strongly.

## Important

Run `python scripts/reset_memory.py` before your first v4 run so old v3 memory does not dominate the new source-gated loop.
