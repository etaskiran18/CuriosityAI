# Changelog v7 — Argument Integrity + Balance Stress + Hybrid Research

v7 focuses on making the philosophical reasoning stricter rather than merely adding more agents.

## Main upgrades

1. **Argument Integrity Gate**
   - `ArgumentReconstruction` can no longer silently leave conclusion, hidden assumptions, inferential gaps, strongest objection, or possible reply blank.
   - A deterministic repair layer fills mandatory fields and makes weaknesses explicit.

2. **Balance Stress Tester**
   - Added a new agent section that interrogates "balanced approach", "tempered curiosity", and "guided curiosity" claims.
   - It forces the system to answer:
     - What exactly is balanced?
     - Who or what decides the balance?
     - When does curiosity become a virtue?
     - When does curiosity become dangerous?
     - What fails when balance fails?

3. **Stricter publish/revision policy**
   - If Argument Critic says `revise` or `block`, publish is forced false.
   - If Balance Stress Tester detects an undefined balance claim, publish is forced false.
   - Weak answers now revise the same topic instead of escaping to a new topic.

4. **Hybrid web/academic grounding support**
   - Existing Tavily support remains.
   - Added optional Crossref metadata search.
   - Added optional Semantic Scholar metadata search.
   - Added `config_hybrid_web.yaml`.

5. **Expanded corpus**
   - Added local Project Gutenberg primary texts:
     - Plato — Theaetetus
     - Plato — Meno
     - Plato — Apology
     - William James — The Will to Believe
   - These are especially useful for wonder, aporia, inquiry, belief, doubt, and knowledge-seeking.

## Recommended use

Run local-only first:

```bash
python scripts/reset_memory.py
python run.py --topic "When does curiosity stop being a virtue and become epistemic instability?" --iterations 3
```

For hybrid web/academic search:

```bash
# Optional for Tavily live web search
set TAVILY_API_KEY=your_key_here

# Optional for higher Semantic Scholar limits
set SEMANTIC_SCHOLAR_API_KEY=your_key_here

python scripts/reset_memory.py
python run.py --config config_hybrid_web.yaml --topic "When does curiosity stop being a virtue and become epistemic instability?" --iterations 3
```
