# v9: honest measurements, and a researcher mode

The first real hour of v8 (mistral 7B on an RTX 4060, 85 heartbeats, web library on) ran without a
single failure. Its diary also showed that the main measurements did not yet measure what they
claimed. v9 fixes that, and adds a researcher mode on the same engine.

## What the hour showed, and what changed

| What the diary showed | What v9 does |
|---|---|
| 180 of 186 predictions were hedged ("the texts may discuss..."). Nothing contradicted it in 85 heartbeats, so surprise stayed near 0 and the learning measure at -0.00. | **Predictions that can fail.** Each one names who will say what, with a probability. Hedged predictions are asked for once more. Surprise is the **Brier score**, which rewards being right and knowing how sure to be. (`organism.py`, `textutil.is_hedged`) |
| In a hand-checked sample of 16 "confirmations", 8 did not support the claim at all (Plato's "Whereas the physician and the carpenter have different natures?" was taken to confirm "cultural differences in novelty and understanding"). | **A blind judge** (`organism/judge.py`) sees only claim, quote and author, and decides supports, contradicts or neither. Only accepted quotes count; beliefs are grounded only with its approval. It can be a larger model (`--judge-model`). Each session saves 20 of its decisions in `judge_check.csv` for a hand-check (`scripts/judge_agreement.py`). |
| 113 new questions, none settled. Half began "How can we..." or "What strategies...". Philosophy words appeared in 7 of the first 56 new questions and in none of the last 57. Meno's question was never visited. | **Staying on topic.** The judge rates every new question against the topic; a question is set aside when the judge and the topic's own words agree that it is off topic. At most one question is born per heartbeat (the most central), a child never matters more than its parent, and a **topic anchor** in the drive lowers the pull of distant questions. |
| All 17 self-checks said "Healthy wonder" while it drifted: any +0.05 confidence step counted as progress. | **Honest self-check.** Progress counts only with evidence the judge accepted. New vital signs (questions settled per heartbeat, closeness to the topic) let it diagnose **restless curiositas** when questions multiply without being settled or when it drifts; the response now also tightens the topic anchor. |
| In 75% of the debates Wonder answered "a valid point" and added more factors. Its theory froze at "a dynamic, multifaceted process... cognitive, motivational, and environmental factors". | **A real debate.** The Skeptic must bring a passage or a named thinker. Wonder starts with DEFEND, REVISE or CONCEDE; after conceding, confidence cannot rise. Every answer must say what would prove it wrong; answers (and theories) too vague to be wrong earn no confidence and are not adopted. |
| The library fetched the Mars rover "Curiosity", "Spark-gap transmitter" and a lawyer's diary from 1602; found 0 papers (Semantic Scholar was most likely busy, but the report said "not found"); and its per-run quota ran out at heartbeat 25. | **Library fixes.** A relevance check before keeping a text; Gutenberg matches the author's family name, not a first name; "busy" is reported as busy; paper searches use keywords, not invented titles; failed searches are not repeated; quotas are per hour; **arXiv** is a second paper source. |
| No measure of learning that the organism could not talk up. | **An outside exam** (`organism/exam.py`, `data/exams/philosophy_of_curiosity.json`): 21 questions whose answers stand in the corpus, with file and line. It answers from its own notes; a blind grader compares with the reference. The **grounded score** counts only answers backed by a verified quote. `--exam` runs it before and after a session. |

## New: researcher mode

`python live.py --topic "..." --papers <folder>` turns the organism into a curious assistant on your
own topic: it writes its own first questions and key terms, reads your PDFs (`organism/papers.py`,
with pypdf), looks for more on Semantic Scholar and arXiv with `--web`, and keeps a
**research map** (`research_map.md` in its folder; code in `organism/research_map.py`): open questions, hypotheses with the quotes they rest on and what contradicted them,
surprises, possible gaps and a reading list. `--until-year` restricts it to older papers for
time-split tests. See [docs/RESEARCHER.md](docs/RESEARCHER.md).

## Also

* Session reports gain lines for predictions, the judge, the debate and staying on topic;
  `metrics.json` gains the matching numbers, and `scripts/compare_sessions.py` shows them.
* `live.py`: `--exam`, `--judge-model`, `--no-judge` (ablation), `--topic`, `--papers`, `--until-year`,
  `--map`; `--status` shows the topic and each question's closeness to it; `--check` checks pypdf, the
  judge model and arXiv.
* Each heartbeat now makes 9 calls to the model (10 when it must restate hedged predictions), up from 6: the judge checks the quotes, then the new beliefs, then rates the new question, each in a short call of its own (a small model mixes up two tasks in one call).
* After updating, start a new life (`--new-life`): beliefs in an old life were graded by the old,
  more lenient rules.
* Tests: 180, all offline (the scripted model now also plays the judge, the grader and the topic setup).

## Tested on a real model

A 5-heartbeat run with a small model (qwen2.5:3b, on a CPU) showed the new machinery working:
0 of 17 predictions hedged (the v8 hour: 97%), closeness to the topic 0.97, the blind judge rejecting
spurious "confirmations" (a Republic passage about "contemplating the expanse of heaven" offered as
proof that "philosophy begins in wonder"), and the self-check diagnosing **restless curiositas**
(a new question every heartbeat, none settled, no evidence-backed progress) and adjusting itself.
The exam went from 0% to 14%, but its grounded score stayed at 0%: the correct answers came from the
language model's own knowledge (and a lenient small grader), not from what it read. That difference is
what the grounded score is for.

The run also found problems, fixed before release:

* A "finding" that only copied its quote let the judge approve a text against itself; the judge now
  checks whether such a passage bears on the question at all.
* Fragments such as "Yes, my boy, outer barbarians." passed as findings; unexpected findings now need a
  quote of at least 8 words, and findings of fewer than 4 words fall back to their quote.
* Side findings raised confidence; now only confirmed predictions can raise it by more than 0.05.
* A small judge mixed up checking quotes and rating questions in one call (it rated a question about
  curiosity "off topic"); rating is now a call of its own, and a question is set aside only when the
  judge and the topic's own words agree.
* Long predictions crowded the question out of the search (asked about Plato's "philosophy begins in
  wonder", it never read the *Theaetetus*); half the passages now come from the question alone.
* A pair the judge skipped counted as before; now nothing counts without the judge's approval.
* Copied instructions and schema hints ("DEFEND (the objection fails: ...)", "only if something
  genuinely surprised you") are removed.
* The report said "mean Brier score 0.00" when no prediction was addressed; it now says there is no
  score yet.

## After the first research run (mistral 7B, a space-physics topic)

* **Deeper investigation.** It visited 9 questions in 9 heartbeats and returned to none; 4 of its new
  questions were one question reworded. Now it returns to a question when texts were fetched for it, and
  a new question needs a reason (see "Depth before breadth" in docs/ORGANISM.md).
* **A doubt without evidence is no contradiction.** A check run still gave birth to 4 questions in 5
  heartbeats, all as "contradictions", with no quote verified: the model doubted one of its earlier
  guesses at each heartbeat, and each doubt let a question be born at once. A doubt now counts as a
  reason only when the judge accepted evidence in the same heartbeat; the doubt itself still lowers the
  belief.
* **Replies cut off by the length limit** ("Could not extract JSON object") are repaired, keeping every
  complete value; the limit is raised to 1000 tokens, and authors are asked for as one family name.
* **"Carpenter will discuss X" counts as hedged**: it says what a text is about, not what it claims, and
  can never be contradicted. It is asked again as a claim.
* **References from memory** in the debate ("T. Nakamura et al., JGR vol. 82, 1977") are marked
  unverified; the voices are told to cite only the passages they read. A reference inside a quoted
  passage is not counted.
* **The judge checks fetched texts** and refuses those it rates unrelated ("Magnetosphere of Saturn"):
  one short extra call per text a search finds. A first threshold, "useful" (2 of 3), refused even
  "Plasmasphere" and left the library empty, so background (1 of 3) is kept.
* **Paper titles** no longer come from journal headers ("SCIENCE ADVANCES | RESEARCH ARTICLE").
* **It learns what the topic means before it asks.** Given only "How does lightning illuminate the
  inner magnetosphere?", it took "illuminate" for light and asked about light emission and "faint
  emissions" for a whole run. A new research life now reads how the person's article and papers begin
  and writes one sentence on what the topic means in its field; that sentence goes into every prompt,
  the judge's view of the topic, the diary and the research map. The wording of the request matters
  with a small model: asked plainly, a 3B model repeated "illuminate" in 3 of 3 tries and its first
  questions stayed on "illumination"; asked for the meaning "in the terms its papers use rather than the
  topic's own words", it did so in 0 of 3, and its questions were about how lightning-generated whistler
  and VLF waves propagate and are detected.
* **Quotes from PDFs**: a sentence copied cleanly from text with layout noise inside (citation marks, a
  word from the next column) counts; 11 of its 12 claimed quotes had been rejected.
* **A busy source is named** in the diary (Semantic Scholar, arXiv or Wikipedia), and a busy site rests
  for 2 minutes instead of 5.

## After the second research run (mistral 7B, 55 heartbeats, the same topic and papers)

It understood the topic this time (lightning-generated VLF whistlers in the plasmasphere), returned to
its questions (up to 4 visits each), grounded beliefs in quotes from the person's article and papers,
and hedged only 4% of its predictions. Three things went wrong:

* **Its own labels became things.** The voices were told "the passages did not clearly address: E1,
  E2, E3", without what E1, E2 and E3 had said. The model took them for unexplained structures in the
  magnetosphere and asked about "the lesser-known structures E1, E2, E3 and E4" for the rest of the hour,
  and the questions about "overlooked" or "yet-to-be-identified" structures grew from there. The voices
  now see each prediction written out, never only its label.
* **Hedged answers settled questions.** "The plasmapause ... may collectively guide or distort
  whistlers, potentially contributing to space weather" settled a question at 0.85. A hedged answer can
  never be wrong, so it now gains at most 0.05 per heartbeat (as an answer without a falsifier does); the
  report counts such answers, and the model is asked to state answers plainly.
* **New questions should be answerable**: "a specific new question that a paper could answer".

## After the third research run (53 heartbeats, the same topic and papers)

The label leak was gone, no question was settled on a hedged answer, only 3 of 198 predictions were
hedged, and it fetched mostly relevant papers (whistler-particle interactions, bounce resonance,
field-line curvature scattering). But 16 of its 19 grounded beliefs quoted the person's own article, and:

* **Paper doubt.** It doubted beliefs 36 times while the judge accepted no contradiction at all, among
  them the article's central claim ("The plasmapause organizes the pathways ..."). A belief that rests on
  an accepted quote is now doubted only when the judge accepted a contradiction in that heartbeat
  (Peirce: doubt needs a surprising fact). Refused doubts are noted in the diary and counted.
* **Falsifiers that test nothing.** 30 of 53 answers would have been wrong only "if X played no role at
  all", "if Y were the only factor" or "regardless of conditions". Such an answer claims only that
  something plays some part, which no finding can refute; it now gains at most 0.05 per heartbeat, like
  a hedged answer. A check on the 53 falsifiers flagged 30 and none of six that name a real finding.
* **Hedged beliefs were grounded.** 6 of the 19 grounded beliefs were hedged ("could play a crucial
  role"); almost any related quote supports such a sentence. Hedged beliefs now stay interpretations.
* **Answers that were questions** ("Can we infer ... thresholds ...?") replaced real answers; they no
  longer do.

## After the fourth research run (51 heartbeats, local PC)

The rules from the third run held: 8 doubts of quoted beliefs refused, 17 strawman falsifiers and 18
hedged answers capped, 1 hedged prediction in 192. But the whole map came to rest on "lightning
polarization", a guess Wonder made at heartbeat 17 that no text supports:

* **Borrowed confidence.** Q3 reached 0.70 with an answer about field-aligned irregularities that a
  paper confirmed; at heartbeat 21 the answer became a polarization guess and kept 0.70 (later 0.75),
  because the irregularity predictions were confirmed again. Confidence now belongs to the answer: the
  question keeps the quotes the judge accepted for its current answer, re-checked every heartbeat in the
  call that already checks beliefs, and confidence cannot exceed `0.5 + 0.1 x` those quotes.
* **Quotes accepted for claims they do not mention.** One sentence of the person's article ("The
  large-scale plasma environment is expected to play a central role ...") grounded five beliefs, among
  them "The role of lightning polarization is context-dependent". A belief and a quote now reach the judge
  only if they share a fifth of their words, one of them not a topic word. On the 35 grounded pairs of
  the second to fourth runs this refuses 13, all unsupported, and keeps every pair a reader would accept.
  (On the 44 confirmed predictions of this run the same check would also refuse two real confirmations,
  so predictions are left to the judge.)
* **Non-answers held at 0.6-0.7** ("Further research is crucial to uncover ...", "are yet to be clearly
  defined") now count as too vague to be wrong.
* Still open: 44 confirmations and no contradiction in 51 heartbeats (predictions general enough that
  texts can only agree or stay silent), and 11 of its 12 grounded beliefs quote the person's own article.

## Before the one test: the open items

* **Predictions a text can contradict.** In the fourth run the texts confirmed 44 predictions and
  contradicted none: half of them only said that one thing influences another. Such predictions are
  now asked again together with hedged ones (one retry), and a confirmation counts as support only if
  its prediction could have failed (Popper's severe test).
* **The person's own draft is a claim to test, not evidence.** At most one passage of it per heartbeat;
  a prediction it matches is recorded as "your draft says so"; it grounds no belief and supports no
  answer. The research map sets each claim of the draft against what the other texts said.
* **Guesses do not take over.** Reflection builds the theory on the answers the texts support and calls
  a guess a guess; questions about undiscovered or lesser-known things are not asked; a bare "REVISE"
  takes no stance.
* **The library refuses texts about another world, star or device** (Venus, magnetars, tokamaks) unless
  the topic names it.
* **PDF titles and words**: journal banners, Cyrillic metadata titles, titles in capitals and titles
  wrapped over two lines are read correctly; letter-spaced words ("interactio n") are rejoined. Checked
  on seven arXiv space-physics PDFs.
* `docs/ONE_TEST.md`: one command, what to send back, and what to look at.
* **Found in a validation run with mistral 7B** (seven arXiv space-physics PDFs, one as a stand-in draft,
  web on): a probability written into the claim ("..., with a probability of 0.75") is taken out of the
  text; "the paper discusses the role of X" counts as hedged like "will discuss"; junk PDF author fields
  ("iitm", "user") are dropped; a draft without an "Abstract" heading no longer shows the topic setup an
  address instead of its abstract; and a `--judge-model` that is not installed stops the run at the
  start instead of silently wasting it (nothing counts as evidence without the judge).
* **Also from the validation run**: mistral turned its predictions into denials when asked for claims a
  text could contradict, so a retry is now kept only if it tests more of what it expects (p >= 0.5), and a
  contradiction counts as a surprise only if the prediction was expected; a fetched paper must be rated
  "useful" (mistral rated a solar-wind paper "loosely related" while writing "not in the Earth's
  magnetosphere"), while encyclopedia articles and books stay background; and the map lists as
  hypotheses only answers that a quote supports.
* **From the final validation run**: predictions name only authors the library has (mistral wrote "Dewey
  found that ...", taken from the template's example, and "Smith's research shows that ..."); the
  person's own work found online is not fetched again as somebody's paper (the stand-in draft came back
  from arXiv); "would be wrong if ... independently of ..." counts as a strawman; a source label such as
  "[S1]" becomes the text's short title (cutting it out left beliefs such as "The model in considers ...",
  and answers in the map kept a bare "[S1]" that meant nothing outside its heartbeat).
* **Also from the final validation run**: the judge accepted a line of a reference list ("Nunn, D. and
  Smith, A.J.: 1996, ...") as a confirmation. Reference lists are full of the topic's words, so the word
  search found them often (40% of one review paper was its reference list). They are no longer read: a
  passage ends at the "References" heading, and stretches dense with authors' initials and years are
  left out (an appendix after the list stays); lines that hold only numbers, such as a figure's axes, are
  dropped from passages. On 49 texts this left out 1.1% of the passages, all reference lists or
  bibliographies. Quotes in the map, the diary and the report now name their text instead of the
  program's key ("[DOC:Unknown:NUMERICAL_SIMULATIONS_OF_THE_EFF:chunk0:L7-L27]"). And when Ollama fails for
  a moment (it answered one call with an error after the process running the model was killed), the call
  is tried again twice, 5 and 10 seconds later.
* **A last check with all of the above** (mistral 7B, 3 heartbeats, same papers and stand-in draft): 2 of 11
  predictions were hedged (in the run before, 10 of 15), the judge agreed with 6 of 8 claims (before, 5 of
  11), and nothing crashed. Too short to measure anything, but it showed four more faults, now fixed: a
  belief is no longer doubted in a heartbeat in which a quote the judge accepted says much the same
  (mistral retracted "the interaction between electrons and the primary whistler wave packet plays a
  role ..." while the judge was accepting "the electrons interacted with the primary whistler wave
  packet", and a "contradiction" question was born from that doubt); predictions name only the library's
  authors, not words from its titles ("Thunderstorms" became an author); "Singh et al. (1992)" in the debate
  is not marked as invented when a passage it read cites Singh itself; and the theory says "(a guess)"
  instead of "(guess: [B4])".

