# Researcher mode: a curious assistant on your own topic

The organism that studies the philosophy of curiosity can study **any research topic you give it**.
It reads your papers, asks its own questions about them, predicts what the literature will say, gets
surprised, argues with itself, and keeps a map of what it has found. It does not answer one question
and stop: it lives with your topic, session after session.

## Start

```bash
python live.py --topic "How do lithium-ion batteries age?" --papers ~/papers/batteries --minutes 60 --web
```

* `--topic`: your topic, in your own words. A new life is born in `memory/research/<topic>/` (or in
  `--home`). It first reads how your papers begin (your own article first, if you share it), then
  writes what it takes the topic to mean in your field, a short title, key terms and five first
  questions for itself. Check that meaning when it is printed: a field's words can have an everyday
  sense too (given only "How does lightning illuminate the inner magnetosphere?", a 7B model studied
  light emission). If the meaning is wrong, start again with `--new-life` and write the topic in your
  field's words.
* `--papers`: a folder with your papers: PDF, `.txt` or `.md`. They are converted to text once, into
  `<home>/papers/`, with page markers such as `[page 3]` so every quote can be found again.
  Scanned PDFs (pictures of pages) contain no text; they are listed so you can run OCR on them first.
* `--web` (optional): when your papers say little about a question, it looks for more: paper abstracts
  from Semantic Scholar and arXiv, and encyclopedia articles. Get a free Semantic Scholar API key
  (semanticscholar.org/product/api) and put it in `SEMANTIC_SCHOLAR_API_KEY`; without one that source
  is often busy. arXiv needs no key.

Run the same command again later to continue the same life. Add papers at any time with `--papers`.

**Folders in WSL.** Write folders with `/` (bash removes `\`), and keep the command on one line. A
folder on Windows such as `C:\Users\you\papers` is `/mnt/c/Users/you/papers`;
`wslpath 'C:\Users\you\papers'` prints it for you. A path in quotes typed the Windows way also works.

## Your own article

If you are writing an article on the topic, share it so the organism knows which paper is yours:

```bash
python live.py --topic "How does lightning illuminate the inner magnetosphere?" --papers ~/my_project/paper/article --feed-file ~/my_project/paper/article/main.pdf --title "My article (draft)" --minutes 60 --web
```

Your article then counts as something a person shared (cited as `INBOX`), not as one of the papers
(cited as `DOC`), and it is not read twice. A new life reads its beginning first, to learn what the
topic means in your field. The organism asks its own questions about it, and its
claims are checked against the literature like any other claim: its predictions about what the papers
say can be contradicted, and the research map shows where the literature is silent (possible gaps) or
disagrees (surprises). That is useful for an introduction or a discussion section, but every quote it
gives must still be checked by you.

## Talk to it

```bash
python live.py --topic "How do lithium-ion batteries age?" --ask "Does calendar aging matter more than cycling for grid storage?"
python live.py --topic "How do lithium-ion batteries age?" --feed-file my_notes.txt --title "Lab notes, March"
python live.py --topic "How do lithium-ion batteries age?" --status
```

Your questions are always treated as central to the topic.

## What you get

| File in the life's folder | What it is |
|---|---|
| `research_map.md` | **The main result.** Open questions ranked by how strongly they pull it; hypotheses (answers it holds with confidence 0.5 or more), what would refute each, the quoted beliefs they rest on, and the predictions the papers contradicted; **surprises** (predictions the papers contradicted); **possible gaps** (questions its papers kept silent about, even after searching); beliefs it came to doubt; a reading list. |
| `diary.md` | Every heartbeat in words: what it predicted (with probabilities), what it read, what the judge accepted, the debate, what changed. |
| `sessions/<id>/report.md` | Each session's summary and numbers. |
| `mind.json` | Everything it knows and wonders about, as data. |

Every quote in the map was found in the source and accepted by a separate, blind judge. A "possible
gap" means *its* library is silent, not that nobody has studied the question; it is a place to look.

## How it goes deep

* After it fetches texts for a question, it goes back to that question to read them.
* A new question needs a reason: evidence the judge accepted, or a gap that is still there on a second
  look. So it works on a few questions several times instead of asking a new one at every heartbeat.
* A question that stays silent after several looks gets boring and rests ("dormant"); it wakes up again
  when new related beliefs arrive.

## How it stays useful

The same safeguards as the philosophy organism (see [ORGANISM.md](ORGANISM.md)):

* predictions must be able to fail, with a probability; hedged guesses ("may", "might") are refused;
* a blind judge decides what each quote really supports; a sample is saved for you to check;
* answers must say what would prove them wrong; vague answers ("complex", "many factors") earn no confidence;
* new questions must stay close to your topic: the judge rates each one, and off-topic ones are set aside;
* texts it fetches must not be unrelated to the question, as the judge reads their title and beginning
  (background articles are kept);
* the debate may cite only the passages it read; references from memory are marked unverified;
* at most one new question per heartbeat: depth before breadth.

## Your own exam

To measure whether it learns your topic, write an exam: questions whose answers stand in your papers,
with a reference answer and where it is written.

```json
{
  "title": "Battery aging basics",
  "questions": [
    {"id": "B1", "question": "Which layer grows on the anode and consumes lithium during cycling?",
     "reference": "The solid electrolyte interphase (SEI).", "source": "Smith 2021, p. 3"}
  ]
}
```

```bash
python live.py --topic "How do lithium-ion batteries age?" --minutes 60 --exam my_exam.json
```

It answers from its own notes only (not from the papers directly), before and after the session. A
blind grader compares each answer with your reference. The **grounded score** counts only answers
backed by a quote it verified, so it shows what it learned by reading rather than what the language
model already knew.

## A test reviewers like: the time split

Give it only the literature up to a year, and see whether its questions point where the field
actually went:

```bash
python live.py --topic "..." --papers papers_until_2020/ --until-year 2020 --minutes 120 --web
```

With `--until-year` it fetches only papers published up to that year, and no encyclopedia articles
(today's articles describe later work). Afterwards, compare the top questions and "possible gaps" in
`research_map.md` with what was published after that year. This needs no expert ratings and can be
repeated for many topics, for example on a computing cluster.

## A stronger model

Research questions need a stronger model than a 7B one. On a bigger machine:

```bash
python live.py --topic "..." --model qwen2.5:32b --judge-model qwen2.5:72b --minutes 120
```

The judge can be a different, larger model than the organism; that makes its decisions, and so all
the measurements, more trustworthy. Check the judge with `scripts/judge_agreement.py` (see
[RESEARCH.md](RESEARCH.md)).

## Limits, honestly

* It can only be as good as its model and its papers. A 7B model reads carefully but thinks plainly.
* It reads text. Figures, tables and equations in PDFs are often lost or garbled.
* Abstracts are not papers: a belief grounded in an abstract is weaker evidence than one grounded in
  the full text. Add the PDFs of key papers to `--papers` when you can.
* The judge is a language model too. It is much stricter than the organism grading itself, but
  check a sample by hand before you trust the numbers.
* It is an assistant for finding questions, surprises and gaps. It is not a source of facts: always
  check the quotes it gives you, which is why it gives them.
