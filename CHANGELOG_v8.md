# Changelog v8: the Curiosity Organism

v8 makes curiosity the mechanism of the system rather than only its topic. The v7 pipeline
(`run.py`) is unchanged. The organism lives next to it (`live.py`, `curiosity_ai/organism/`).

## Why

The v7 run in `reports/run_1777754905_07c570e3` showed the limits of asking a small model to judge its
own depth:

- every thesis converged on "curiosity needs balance with critical thinking";
- 3 of 4 iterations were forced revisions of the same Kant/Dewey question;
- the move detector gave 1.000 to a platitude, and the theory state reached confidence 1.0 while depth failed;
- the web search returned papers on copyright law and medical curricula.

## Added

- **Persistent mind** (`memory/organism/mind.json`): open questions with confidence, answers, visit
  history and parents; beliefs with verified evidence and revision history; temperament; self-model.
- **Curiosity drive** (`organism/drive.py`): information gap (inverted U), anchoring, learning progress,
  surprise, novelty with habituation, importance, boredom and a refractory period. Softmax choice.
- **Predict, observe, compare**: expectations are committed before reading; prediction error and
  informativeness are measured from verified quotes only.
- **Quote verification**: confirmations and contradictions count only if the quote is found in the
  passage. Fabricated quotes are discarded and reported.
- **Evidence-bounded confidence**: bounded steps per heartbeat, +0.05 at most without evidence, and a
  ceiling that rises with verified confirmations.
- **Inner dialogue**: Wonder, then Skeptic, then Wonder.
- **Question life cycle**: born from surprise, contradiction, gaps, objections, humans, observations and
  reflection; open, settled (Peirce), dormant (boredom, incubation), unanswerable (Kant).
- **Self-regulation**: vital signs, diagnosis (healthy wonder, restless curiositas, dogmatic slumber,
  aporetic numbness), bounded temperament changes, and an evolving theory of curiosity.
- **Human interaction**: `--ask`, `--feed`/`--feed-file` (shared observations join the library),
  `--status`, `--reflect`, `--new-life`.
- **BM25 library** with no extra dependencies: strips Gutenberg boilerplate, normalizes old spellings
  (enquire/inquire, sceptic/skeptic), takes at most 2 passages per source, and gives citations with
  real file line numbers. `retrieval: chroma` reuses v7's vector memory.
- **Web relevance filter**: results must cover at least 25% of the question's content words.
- **Experience export**: `experience/sft.jsonl` and `experience/preference_pairs.jsonl` (verified
  corrections) for future fine-tuning.
- **Diary** (`memory/organism/diary.md`): why each question was chosen, what was expected and found,
  the dialogue, and what changed.
- **Body care** (`organism/body.py`, `organism.body` in config): a breath after every heartbeat, a
  rest after every stretch of thinking, a GPU temperature guard (`nvidia-smi`; stop at 80 °C, continue
  at 65 °C), and a battery guard for laptops. Rests are written in the diary. `--no-rest` turns it off;
  `--pause` sets the breath.
- **Setup check** (`python live.py --check`): Python, library, Ollama, model, a test answer with its
  time, how much of the model Ollama reports on the GPU, GPU temperature, power, and write access.
- **WSL**: the battery guard asks Windows for the charger state (cached for a minute);
  `docs/WSL_SETUP.md` is a step-by-step guide for Windows laptops with an NVIDIA GPU.
- **LLM trace** (`trace_llm: true`): every prompt and raw reply in `llm_trace.jsonl`, for tuning prompts.
- `docs/ORGANISM.md`: architecture and the philosophy-to-mechanism map, with corpus line references.
- Tests: `tests/test_organism_*.py` (a scripted fake model, so no Ollama is needed).

## Changed

- `OllamaClient.chat/json_chat` accept an optional `max_tokens` (backward compatible).
- `config.yaml` / `config_hybrid_web.yaml` gain an `organism:` section; all fields have defaults.
- `tests/test_config` was renamed to `tests/test_config.py` so pytest collects it; `pytest.ini` added.
- `.gitignore` added; committed `__pycache__` files removed from the repository.
- `scripts/reset_memory.py` now also removes the v7 journal and theory state.
