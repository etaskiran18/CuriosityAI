# Patch Notes v2.1

This patch improves robustness for local instruct models such as `mistral:7b-instruct`.

## Fixed

- Prevented crashes when a local LLM returns a string where the schema expects a list.
- Added tolerant Pydantic validators for:
  - `Critique.main_objections`
  - `Critique.accepted_questions`
  - `Critique.rejected_questions`
  - `QuestionCandidate.assumptions`, `likely_sources`, `tags`
  - `Synthesis.argument`, `counterarguments`, `unresolved_tensions`, `next_directions`, citation fields
  - `Evaluation` booleans and lists
  - `PlanningDecision` list/mode/value fields
- Made Explorer question parsing tolerant of dict/string/list variations.

## Why this matters

Mistral 7B can be strong for reasoning, but it sometimes violates strict JSON schemas by returning:

```json
{"main_objections": "Questions were too vague or repetitive."}
```

instead of:

```json
{"main_objections": ["Questions were too vague or repetitive."]}
```

The loop now coerces these harmless schema variations instead of stopping.
