# The one test

Everything below is one run of one hour. It uses all the changes made after the four research runs on
"How does lightning illuminate the inner magnetosphere?" (see `CHANGELOG_v9.md`).

## 1. Update

In WSL, one line:

```bash
cd ~/CuriosityAI && source .venv/bin/activate && git pull && pip install -r requirements.txt
```

## 2. A bigger judge (recommended)

The judge decides which quotes count as evidence, and a 7B judge accepts too much. If your GPU has
about 12 GB of memory or more, download a bigger model once:

```bash
ollama pull qwen2.5:14b
```

If it does not fit, skip this step: mistral then judges its own work, as before.

Optional: with a free Semantic Scholar API key (semanticscholar.org/product/api) it finds more papers.
Set it once in the same terminal before the run: `export SEMANTIC_SCHOLAR_API_KEY=your_key`.

## 3. Run

One line (add `--judge-model qwen2.5:14b` at the end if you did step 2):

```bash
python live.py --new-life --topic "How does lightning illuminate the inner magnetosphere?" --papers ~/core_density_inversion/core_density_inversion/paper/article --feed-file ~/core_density_inversion/core_density_inversion/paper/article/main.pdf --title "My article (draft)" --home memory/research/lightning --minutes 60 --web
```

`--new-life` is needed: lives started with older versions keep confidences and beliefs made under the
old rules. The old life is moved to an archive folder, not deleted.

On a laptop, keep the charger connected: on battery it rests until the charger is connected (it says so).

When it starts, read the line **"It takes the topic to mean: ..."**. If that is not what your field
means, stop it (Ctrl+C) and start again with the topic in your field's words.

## 4. Send back

From `memory/research/lightning/`:

* `research_map.md`
* `diary.md`
* `sessions/<newest>/report.md`
* `sessions/<newest>/judge_check.csv`, if you have ten minutes: write `supports`, `contradicts` or
  `neither` in the last column (`your_verdict`) for each row. That measures how far the judge can be
  trusted (`python scripts/judge_agreement.py <that file>`).

## 4b. The curiosity life (the philosophy of curiosity), after the research run

Run it after the research run, not at the same time (both would share the GPU and both would be slow):

```bash
python live.py --new-life --minutes 60 --web
```

Add `--judge-model qwen2.5:14b` here too if you downloaded it. This life lives in `memory/organism/`
(`--new-life` moves the old one to an archive folder). Send back from `memory/organism/`:
`research_map.md`, `diary.md` and `sessions/<newest>/report.md`.

## 5. What to look at yourself

1. **"Your draft's claims it met, and what the other texts say"** in the research map: does what it
   found agree with what you know of the literature?
2. **Hypotheses**: every answer lists the quotes that support it ("Supported by"). Do they?
3. **Open questions**: are they questions you would find worth asking?
4. **Report, "Predictions"**: how many said only that something has an influence (few is good), and
   whether the texts *contradicted* any prediction (that is a real surprise; before, there were none).

## What is new since your last run

* Confidence belongs to an answer and the quotes that support it; a new answer does not inherit the old
  one's confidence.
* A quote reaches the judge only if it shares real words with the claim.
* Your draft is a set of claims to test, not evidence; at most one passage of it per heartbeat.
* Predictions that only say "X influences Y" are asked again; only predictions that could fail count.
* Answers that only say "further research is needed" earn no confidence.
* Questions about "undiscovered" or "lesser-known" things are not asked.
* Its theory is built from what the texts support; guesses must be called guesses.
* Texts about Venus, magnetars, tokamaks and the like are refused for an Earth topic.
* Paper titles are read correctly from more PDFs.
* Reference lists are not read: a reference ("Nunn, D. and Smith, A.J.: 1996, ...") is not evidence.
* Quotes name the text they come from; a label such as "[S1]" no longer appears in the map.
