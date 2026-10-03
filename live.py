"""Live, talk to, and observe the curiosity organism (v8).

Examples:
    python live.py                                   # live a few heartbeats
    python live.py --heartbeats 20
    python live.py --forever                         # keep living until Ctrl+C (it rests to keep the PC cool)
    python live.py --ask "Can a machine be curious, or only act as if it were?"
    python live.py --feed-file my_notes.txt --title "My notes on boredom"
    python live.py --status                          # look inside its mind
    python live.py --new-life                        # archive this life and start again
    python live.py --check                           # check this computer's setup and say what to fix
    python live.py --minutes 60 --web --exam         # a 1-hour test, with the exam before and after

Researcher mode (a curious assistant on your own topic, reading your papers):
    python live.py --topic "How do lithium-ion batteries age?" --papers ~/papers/batteries --minutes 60 --web
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from curiosity_ai.config import AppConfig, load_config
from curiosity_ai.llm import OllamaClient
from curiosity_ai.organism import CuriosityOrganism, OrganismError
from curiosity_ai.organism.body import SystemSensors
from curiosity_ai.organism.librarian import Librarian, user_agent
from curiosity_ai.organism.drive import DIAGNOSES
from curiosity_ai.organism.exam import run_exam
from curiosity_ai.organism.papers import ingest_papers, pdf_text
from curiosity_ai.organism.research_map import write_research_map
from curiosity_ai.organism.state import Episode
from curiosity_ai.organism.textutil import one_line

console = Console()


def check_ollama(config: AppConfig) -> bool:
    base = config.llm.base_url.rstrip("/")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        console.print(f"[red]Cannot reach Ollama at {escape(base)}[/red] ({escape(str(exc))}).\nStart it with:  [bold]ollama serve[/bold]")
        return False
    names = {m.get("name", "") for m in resp.json().get("models", [])}
    model = config.llm.model
    if not _is_model(names, model):
        console.print(f"[red]Model '{escape(model)}' is not installed in Ollama.[/red] Install it with:  [bold]ollama pull {escape(model)}[/bold]")
        return False
    judge = config.organism.judge
    on_this_server = not judge.base_url or judge.base_url.rstrip("/") == base
    if judge.enabled and judge.model and on_this_server and not _is_model(names, judge.model):
        # Without its judge nothing would count as evidence: better to stop now than to waste the hour.
        console.print(f"[red]Judge model '{escape(judge.model)}' is not installed in Ollama.[/red] Install it with:  "
                      f"[bold]ollama pull {escape(judge.model)}[/bold]  (or leave out --judge-model)")
        return False
    return True


def _is_model(names: set[str], model: str) -> bool:
    return model in names or f"{model}:latest" in names or any(n.startswith(model + ":") for n in names)


def run_checks(config: AppConfig, online: bool = False) -> int:
    """Check this computer's setup, one line per item, and say what to fix."""
    rows: list[tuple[str, str, str]] = []

    v = sys.version_info
    rows.append(("ok" if v >= (3, 10) else "FIX", "Python", f"{v.major}.{v.minor}.{v.micro}" + ("" if v >= (3, 10) else ": needs 3.10 or newer")))
    try:
        import chromadb

        rows.append(("ok", "chromadb", f"{chromadb.__version__} (used by run.py and retrieval: chroma)"))
    except Exception:
        rows.append(("note", "chromadb", "not installed: only needed for the v7 run.py or retrieval: chroma"))

    try:
        import pypdf

        rows.append(("ok", "pypdf", f"{pypdf.__version__} (reads your PDF papers in researcher mode)"))
    except Exception:
        rows.append(("note", "pypdf", "not installed: only needed for PDF papers in researcher mode (pip install pypdf)"))

    corpus = Path(config.corpus.path)
    extensions = {e.lower() for e in config.corpus.accepted_extensions}
    texts = [f for f in corpus.glob("*") if f.suffix.lower() in extensions and not f.name.lower().startswith("readme")] if corpus.exists() else []
    rows.append(("ok" if texts else "FIX", "Library", f"{len(texts)} texts in {corpus}" if texts else f"no texts found in {corpus}"))

    base, model = config.llm.base_url.rstrip("/"), config.llm.model
    try:
        names = {m.get("name", "") for m in requests.get(f"{base}/api/tags", timeout=5).json().get("models", [])}
        rows.append(("ok", "Ollama", f"running at {base}"))
    except Exception:
        names = None
        rows.append(("FIX", "Ollama", f"not reachable at {base}. Start it in another terminal with: ollama serve"))
    judge = config.organism.judge
    if judge.enabled and judge.model and names is not None and not judge.base_url:
        if _is_model(names, judge.model):
            rows.append(("ok", "Judge model", judge.model))
        else:
            rows.append(("FIX", "Judge model", f"'{judge.model}' is not installed. Run: ollama pull {judge.model}"))
    elif judge.enabled:
        rows.append(("ok", "Judge", "on, using the main model (a larger judge model makes measurements stronger)" if not judge.model else f"on, {judge.model}"))
    else:
        rows.append(("note", "Judge", "off: the organism grades its own evidence (only for ablation experiments)"))
    if names is not None and not _is_model(names, model):
        rows.append(("FIX", "Model", f"'{model}' is not installed. Run: ollama pull {model}"))
    elif names is not None:
        rows.append(("ok", "Model", model))
        try:
            start = time.time()
            OllamaClient(config.llm).json_chat("You answer in JSON.", 'Reply with {"ok": true}.', '{"ok": true}', temperature=0.0, max_tokens=20)
            rows.append(("ok", "Model answers", f"in {time.time() - start:.1f} s (the first call also loads the model)"))
            rows.append(_gpu_share(base, model))
        except Exception as exc:
            rows.append(("FIX", "Model answers", f"the model did not answer: {one_line(str(exc), 120)}"))

    sensors = SystemSensors()
    temp = sensors.gpu_temperature()
    if temp is not None:
        rows.append(("ok", "GPU temperature", f"{temp:.0f}°C now; it pauses at {config.organism.body.max_gpu_temp_c:g}°C"))
    else:
        rows.append(("note", "GPU temperature", "not readable (nvidia-smi not found); the rest rhythm still protects the PC"))
    power = sensors.on_battery()
    rows.append((
        "ok" if power is False else "note",
        "Power",
        {False: "plugged in", True: "on battery: it will wait for the charger", None: "unknown (no battery found, or not readable)"}[power],
    ))

    home = Path(config.organism.home)
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        rows.append(("ok", "Organism home", str(home)))
    except OSError as exc:
        rows.append(("FIX", "Organism home", f"cannot write to {home}: {exc}"))

    if online:
        rows += _online_rows(config)
    else:
        rows.append(("note", "Web library", "not checked; add --web to test Wikipedia, Project Gutenberg and Semantic Scholar"))

    table = Table(title="Curiosity organism: setup check")
    for col in ("", "item", "detail"):
        table.add_column(col)
    colors = {"ok": "green", "FIX": "red", "note": "yellow"}
    for status, item, detail in rows:
        table.add_row(f"[{colors[status]}]{status}[/{colors[status]}]", escape(item), escape(detail))
    console.print(table)
    problems = sum(1 for status, _, _ in rows if status == "FIX")
    if problems:
        console.print(f"[red]{problems} thing(s) to fix, see above.[/red]")
        return 1
    console.print("[green]Ready. Try:  python live.py --heartbeats 3[/green]")
    return 0


def _online_rows(config: AppConfig) -> list[tuple[str, str, str]]:
    """One quick request to each source the librarian uses."""
    lib = config.organism.librarian
    headers = {"User-Agent": user_agent(lib.contact)}
    key = os.environ.get(lib.semantic_scholar_api_key_env)
    probes = [
        ("Wikipedia", "https://en.wikipedia.org/w/api.php",
         {"action": "query", "list": "search", "srsearch": "curiosity", "srlimit": 1, "format": "json"}, {}, True),
        ("Project Gutenberg", f"{lib.gutenberg_mirror.rstrip('/')}/cache/epub/1643/pg1643.txt", None, {}, True),
        ("Semantic Scholar", "https://api.semanticscholar.org/graph/v1/paper/search",
         {"query": "curiosity", "limit": 1, "fields": "title"}, {"x-api-key": key} if key else {}, False),
        ("arXiv", "https://export.arxiv.org/api/query", {"search_query": "all:curiosity", "max_results": 1}, {}, False),
    ]
    rows = []
    for name, url, params, extra, needed in probes:
        try:
            resp = requests.get(url, params=params, headers={**headers, **extra}, timeout=20, stream=True)
            code = resp.status_code
            resp.close()
        except Exception as exc:
            rows.append(("FIX" if needed else "note", name, f"unreachable: {one_line(str(exc), 90)}"))
            continue
        if code < 400:
            rows.append(("ok", name, "reachable"))
        elif code == 429:
            hint = "" if name != "Semantic Scholar" else " (optional; a free API key in SEMANTIC_SCHOLAR_API_KEY raises the limit)"
            rows.append(("note", name, f"busy right now (429 Too Many Requests); it waits and retries{hint}"))
        else:
            rows.append(("FIX" if needed else "note", name, f"answered HTTP {code}"))
    return rows


def _gpu_share(base: str, model: str) -> tuple[str, str, str]:
    """How much of the loaded model Ollama keeps in GPU memory."""
    try:
        loaded = requests.get(f"{base}/api/ps", timeout=5).json().get("models", [])
    except Exception:
        return ("note", "GPU use", "could not ask Ollama where the model runs")
    for m in loaded:
        if not _is_model({m.get("name") or m.get("model") or ""}, model):
            continue
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        if size and vram >= 0.99 * size:
            return ("ok", "GPU use", "Ollama reports 100% of the model on the GPU")
        if size and vram:
            return ("note", "GPU use", f"Ollama reports {vram / size:.0%} on the GPU and the rest on the CPU (slower); a smaller model would fit fully")
        return ("note", "GPU use", "Ollama reports the model on the CPU only (slow); check the NVIDIA driver with: nvidia-smi")
    return ("note", "GPU use", "the model is not loaded right now")


def print_heartbeat(organism: CuriosityOrganism, ep: Episode | None) -> None:
    if ep is None:
        console.print("[dim]No open question; the organism rests.[/dim]")
        return
    if ep.status_after == "no-thought":
        console.print(f"[red]\\[heartbeat {ep.heartbeat}] could not think:[/red] {escape('; '.join(ep.errors))}")
        return
    console.print(
        f"[bold cyan]\\[heartbeat {ep.heartbeat}][/bold cyan] {ep.question_id} "
        f"[dim](curiosity {ep.drive.get('total', 0):.2f})[/dim] {escape(ep.question)}"
    )
    parts = [
        f"surprise {ep.prediction_error:.2f}",
        f"confidence {ep.prior_confidence:.2f} -> {ep.confidence:.2f}",
    ]
    if ep.support or ep.contradicted:
        parts.append(f"evidence +{ep.support}/-{ep.contradicted}")
    rejected = sum(1 for j in ep.judgments if j.get("judge") == "neither")
    if rejected:
        parts.append(f"judge rejected {rejected} quote(s)")
    if ep.stance:
        parts.append(f"wonder {ep.stance}s")
    if ep.vague:
        parts.append("answer too vague to be wrong")
    if ep.set_aside_questions:
        parts.append(f"{len(ep.set_aside_questions)} off-topic question(s) set aside")
    if ep.new_belief_ids:
        parts.append(f"+{len(ep.new_belief_ids)} belief(s)")
    if ep.doubted_belief_ids:
        parts.append(f"doubting {', '.join(ep.doubted_belief_ids)}")
    if ep.new_question_ids:
        parts.append(f"+{len(ep.new_question_ids)} question(s)")
    if ep.rejected_quotes:
        parts.append(f"{ep.rejected_quotes} unverifiable quote(s) discarded")
    if ep.status_after != "open":
        parts.append(f"now {ep.status_after}")
    console.print("   " + escape(" | ".join(parts)))
    if ep.insight:
        console.print(f"   [italic]{escape(ep.insight)}[/italic]")
    if ep.errors:
        console.print(f"   [yellow]trouble: {escape('; '.join(ep.errors))}[/yellow]")
    if ep.heartbeat % max(1, organism.oc.reflect_every) == 0 and organism.oc.reflect_every > 0:
        sm = organism.state.self_model
        console.print(f"   [magenta]reflection: {DIAGNOSES[sm.diagnosis]['name']}[/magenta] - {escape(sm.last_reflection)}")


def print_status(organism: CuriosityOrganism) -> None:
    st = organism.state
    counts: dict[str, int] = {}
    for q in st.questions.values():
        counts[q.status] = counts.get(q.status, 0) + 1
    belief_counts: dict[str, int] = {}
    for b in st.beliefs.values():
        belief_counts[b.status] = belief_counts.get(b.status, 0) + 1
    diagnosis = DIAGNOSES.get(st.self_model.diagnosis, DIAGNOSES["healthy_wonder"])
    t = st.temperament
    years = f" (papers up to {st.topic.until_year})" if st.topic.until_year else ""
    if organism.research:
        theory = f"[bold]What it thinks about its topic:[/bold] {escape(organism._understanding())}"
    else:
        theory = f"[bold]What it thinks curiosity is:[/bold] {escape(st.self_model.understanding_of_curiosity)}"
    text = (
        f"[bold]{escape(st.name)}[/bold], {st.heartbeat} heartbeats old (born {st.born_at[:10]}), studying "
        f"[bold]{escape(st.topic.title)}[/bold]{escape(years)}\n"
        f"Questions: {', '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'none'}\n"
        f"Beliefs: {', '.join(f'{k} {v}' for k, v in sorted(belief_counts.items())) or 'none'}\n"
        f"State of curiosity: [magenta]{diagnosis['name']}[/magenta] - {diagnosis['meaning']}\n"
        f"Temperament: gap {t.gap:.2f}, learning progress {t.learning_progress:.2f}, surprise {t.surprise:.2f}, "
        f"novelty {t.novelty:.2f}, importance {t.importance:.2f}, patience {t.boredom_patience}, "
        f"exploration {t.exploration_temperature:.2f}, topic anchor {t.topic_anchor:.2f}\n\n{theory}"
    )
    console.print(Panel(text, title="Curiosity organism" + (" (researcher)" if organism.research else "")))
    table = Table(title="Open questions, by the pull they exert")
    for col in ("id", "pull", "gap", "progress", "surprise", "novelty", "bored", "topic", "conf", "question"):
        table.add_column(col, justify="right" if col not in ("id", "question") else "left")
    for q, r in organism.drives()[:12]:
        table.add_row(
            q.id, f"{r.total:.2f}", f"{r.gap:.2f}", f"{r.learning_progress:.2f}", f"{r.surprise:.2f}",
            f"{r.novelty:.2f}", f"{r.boredom:.2f}", f"{r.relevance:.2f}", f"{q.confidence:.2f}", escape(one_line(q.text, 120)),
        )
    console.print(table)
    beliefs = sorted(st.held_beliefs(), key=lambda b: -b.confidence)[:10]
    if beliefs:
        btable = Table(title="Strongest beliefs")
        for col in ("id", "conf", "grounding", "belief"):
            btable.add_column(col)
        for b in beliefs:
            grounding = f"{len(b.evidence)} quote(s)" if b.evidence else "interpretation"
            btable.add_row(b.id, f"{b.confidence:.2f}", grounding, escape(one_line(b.statement, 140)))
        console.print(btable)
    print_library(organism)
    console.print(f"Diary: {escape(str(organism.home / 'diary.md'))}   Mind: {escape(str(organism.mind_path))}")
    if (organism.home / "research_map.md").exists():
        console.print(f"Research map: {escape(str(organism.home / 'research_map.md'))}")


def print_library(organism: CuriosityOrganism) -> None:
    """What it can read: the shared corpus, what humans shared, and what it acquired itself."""
    config = organism.config
    extensions = {e.lower() for e in config.corpus.accepted_extensions}
    corpus = [f for f in Path(config.corpus.path).glob("*") if f.suffix.lower() in extensions and not f.name.lower().startswith("readme")]
    shared = list(organism.inbox_dir.glob("*.md")) if organism.inbox_dir.exists() else []
    acquired = {kind: len(list((organism.library_dir / kind).glob("*.md"))) for kind in ("encyclopedia", "books", "papers")}
    own = f"{len(list(organism.papers_dir.glob('*.md')))} of your papers" if organism.research else f"{len(corpus)} texts in the corpus"
    console.print(
        f"[bold]Library:[/bold] {own}, {len(shared)} shared by humans, acquired by itself: "
        f"{acquired['encyclopedia']} encyclopedia articles, {acquired['books']} books, {acquired['papers']} paper abstracts."
    )
    log = organism.library_dir / "acquisitions.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()] if log.exists() else []
    for row in rows[-5:]:
        console.print(f"[dim]  heartbeat {row.get('heartbeat')}: {escape(row.get('title', ''))} ({row.get('kind')}) because {escape(row.get('reason', ''))}[/dim]")


def add_books(config: AppConfig, queries: list[str]) -> int:
    """Download public-domain books from Project Gutenberg into the shared corpus."""
    corpus = Path(config.corpus.path)
    librarian = Librarian(config.organism.librarian, Path(config.organism.home) / "library")
    console.print("[dim]Searching Project Gutenberg's catalogue (the first time, it downloads the 6 MB catalogue)...[/dim]")
    missing = 0
    for query in queries:
        acquisition, candidates = librarian.add_book(query, corpus)
        if acquisition is not None:
            console.print(f"[green]Added[/green] {escape(acquisition.title)} by {escape(acquisition.author)} -> {escape(acquisition.path)}")
        elif candidates:
            best = candidates[0]
            console.print(f"[yellow]Not added[/yellow] '{escape(query)}': best match {escape(best.title)} by {escape(best.author)} (#{best.ebook_id}) is already in the library or could not be downloaded.")
        else:
            missing += 1
            console.print(f"[red]No book found for[/red] '{escape(query)}'. Try author and title, e.g. \"Hobbes Leviathan\".")
        if len(candidates) > 1:
            others = "; ".join(f"{c.title} by {c.author} (#{c.ebook_id})" for c in candidates[1:4])
            console.print(f"[dim]  other matches: {escape(others)}[/dim]")
    console.print("[dim]New books are read at the organism's next start.[/dim]")
    return 1 if missing else 0


def print_session(report) -> None:
    m = report.metrics
    judge = f"judge agreed {m['judge_agreement']:.0%} of {m['judge_pairs']}" if m.get("judge_pairs") else "judge: nothing to check"
    exam = ""
    if "exam_after" in m:
        exam = f"\nexam: {m.get('exam_before', 0):.0%} -> {m['exam_after']:.0%} (grounded {m.get('exam_grounded_before', 0):.0%} -> {m['exam_grounded_after']:.0%})"
    console.print(
        Panel(
            f"{m['heartbeats']} heartbeats in {m['minutes']:g} min (rested {m['rest_minutes']:g} min) | "
            f"questions born {m['questions_born']}, set aside {m['questions_set_aside']} | new beliefs {m['beliefs_new']} "
            f"({m['beliefs_grounded']} grounded) | doubted {m['beliefs_doubted']}\n"
            f"predictions {m['predictions']} (hedged {m['hedged_rate']:.0%}), confirmed {m['predictions_confirmed']}, "
            f"contradicted {m['predictions_contradicted']} | Brier {'-' if m['mean_brier'] is None else format(m['mean_brier'], '.2f')} | {judge} | "
            f"on topic {m['mean_on_topic']:.2f} | invented quotes caught {m['quotes_rejected']} | texts acquired {m['acquisitions']}{exam}",
            title=f"Session {escape(report.session_id)}",
        )
    )
    console.print(f"Report: {escape(str(report.directory / 'report.md'))}")
    research_map = Path(m["home"]) / "research_map.md"
    if research_map.exists():
        console.print(f"Research map: {escape(str(research_map))}")


def print_exam(result) -> None:
    table = Table(title=f"Exam: {result.exam} - score {result.score:.0%}, grounded {result.grounded_score:.0%}")
    for col in ("id", "grade", "grounded", "answer"):
        table.add_column(col)
    for a in result.answers:
        table.add_row(a.id, a.grade, "yes" if a.grounded else "", escape(one_line(a.answer, 110)))
    console.print(table)
    console.print(f"Saved: {escape(str(result.path))}")


def local_path(text: str) -> Path:
    """A path typed the Windows way also works in WSL: C:\\Users\\me\\papers -> /mnt/c/Users/me/papers.

    (Typed without quotes, bash removes the backslashes before Python sees them,
    so a missing folder gets a hint instead.)
    """
    text = text.strip().strip('"').strip("'")
    if os.name != "nt":
        drive = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
        if drive:
            text = f"/mnt/{drive.group(1).lower()}/" + drive.group(2).replace("\\", "/")
        elif "\\" in text and not Path(text).expanduser().exists():
            text = text.replace("\\", "/")
    return Path(text).expanduser()


PATH_HINT = (
    "In WSL, write folders with / (not \\): bash removes backslashes. Put the whole command on one line.\n"
    "A folder on Windows such as C:\\Users\\you\\papers is /mnt/c/Users/you/papers in WSL; "
    "the command wslpath 'C:\\Users\\you\\papers' converts it. Check a folder with: ls <folder>"
)


def ingest(config: AppConfig, folder: str, skip: set[Path] | None = None) -> int:
    """Researcher mode: convert the person's papers into the organism's library."""
    source = local_path(folder)
    if not source.is_dir():
        console.print(f"[red]There is no folder at {escape(str(source))}.[/red]")
        console.print(escape(PATH_HINT))
        return 1
    target = Path(config.organism.home) / "papers"
    console.print(f"[dim]Reading your papers from {escape(str(source))} ...[/dim]")
    report = ingest_papers(source, target, max_pages=config.organism.research.max_pdf_pages, skip=skip)
    console.print(
        f"Papers: {len(report.added)} added, {len(report.already)} already in its library, {len(report.failed)} could not be read"
        + (f", {len(report.skipped)} left out (shared as yours with --feed-file)" if report.skipped else "") + "."
    )
    for name, why in report.failed[:10]:
        console.print(f"[yellow]  {escape(name)}: {escape(why)}[/yellow]")
    return 0 if report.added or report.already else 1


def read_shared_file(path: Path) -> tuple[str, str]:
    """(title, text) of a file a person shares: a PDF (for example your own article) or a text file."""
    if path.suffix.lower() == ".pdf":
        meta, text = pdf_text(path)
        return meta.get("title") or path.stem, text
    return path.stem, path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live, talk to, and observe the curiosity organism.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--heartbeats", type=int, default=None, help="How many acts of inquiry to live now")
    parser.add_argument("--minutes", type=float, default=None, help="Live this many minutes (rests included), then stop and write the session report")
    parser.add_argument("--forever", action="store_true", help="Keep living until Ctrl+C")
    parser.add_argument("--web", action="store_true", help="Let it grow its library from the internet: Wikipedia, Project Gutenberg books, paper abstracts")
    parser.add_argument("--policy", choices=["curiosity", "random", "novelty"], default=None, help="How it chooses its next question; random and novelty are baselines for experiments")
    parser.add_argument("--home", default=None, help="Folder of this organism's life (use separate folders for experiments); default memory/organism")
    parser.add_argument("--label", default=None, help="A name added to this session's report folder")
    parser.add_argument("--pause", type=float, default=None, help="Seconds to breathe between heartbeats (organism.body.breath_seconds)")
    parser.add_argument("--no-rest", action="store_true", help="Turn off rests and the temperature/battery guards (for short experiments only)")
    parser.add_argument("--ask", action="append", default=[], help="Give the organism a question (repeatable)")
    parser.add_argument("--feed", default=None, help="Share an observation as text")
    parser.add_argument("--feed-file", default=None, help="Share a file with it: text, markdown or PDF (for example your own article)")
    parser.add_argument("--title", default=None, help="Title for --feed/--feed-file")
    parser.add_argument("--add-book", action="append", default=[], metavar="QUERY", help='Download a public-domain book into the corpus, e.g. "Hobbes Leviathan" (repeatable)')
    parser.add_argument("--status", action="store_true", help="Show the organism's mind and library, then exit")
    parser.add_argument("--reflect", action="store_true", help="Reflect now")
    parser.add_argument("--new-life", action="store_true", help="Archive the current life and start a new one")
    parser.add_argument("--model", default=None, help="Override llm.model (e.g. mistral:7b-instruct, qwen2.5:3b)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible choices")
    parser.add_argument("--check", action="store_true", help="Check this computer's setup (with --web, also the online sources) and exit")
    parser.add_argument("--exam", nargs="?", const="default", default=None, metavar="FILE",
                        help="Take the exam (default data/exams/philosophy_of_curiosity.json); with a session, before and after it")
    parser.add_argument("--judge-model", default=None, help="A separate, ideally larger, model for the blind judge (e.g. qwen2.5:14b)")
    parser.add_argument("--no-judge", action="store_true", help="Let the organism grade its own evidence (only for ablation experiments)")
    parser.add_argument("--topic", default=None, help='Researcher mode: the research topic of a new life, e.g. "How do lithium-ion batteries age?"')
    parser.add_argument("--papers", default=None, metavar="FOLDER", help="Researcher mode: a folder of your papers (PDF, .txt, .md) to read")
    parser.add_argument("--until-year", type=int, default=None, help="Researcher mode: read only papers published up to this year (time-split tests)")
    parser.add_argument("--map", action="store_true", help="Write the research map (research_map.md) now and exit")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    if args.model:
        config.llm.model = args.model
    if args.seed is not None:
        config.organism.random_seed = args.seed
    if args.pause is not None:
        config.organism.body.breath_seconds = args.pause
    if args.no_rest:
        config.organism.body.enabled = False
    if args.home:
        config.organism.home = args.home
    if args.policy:
        config.organism.policy = args.policy
    if args.web:
        config.organism.librarian.enabled = True
    if args.judge_model:
        config.organism.judge.model = args.judge_model
    if args.no_judge:
        config.organism.judge.enabled = False
    if args.topic:
        if not args.home:
            slug = re.sub(r"[^a-z0-9]+", "-", args.topic.lower()).strip("-")[:40].strip("-") or "topic"
            config.organism.home = f"memory/research/{slug}"
        config.organism.research.topic = args.topic
        config.organism.research.until_year = args.until_year
    elif args.papers or args.until_year:
        console.print("[red]--papers and --until-year belong to researcher mode: add --topic \"your topic\".[/red]")
        return 1
    if args.check:
        return run_checks(config, online=args.web)
    if args.add_book:
        added = add_books(config, args.add_book)
        if not (args.heartbeats is not None or args.minutes or args.forever):
            return added

    if args.new_life:
        archived = CuriosityOrganism.archive(config)
        console.print(f"Previous life archived to {escape(str(archived))}." if archived else "There was no previous life to archive.")

    home = Path(config.organism.home)
    feed_path = local_path(args.feed_file) if args.feed_file else None
    if feed_path is not None and not feed_path.is_file():
        console.print(f"[red]There is no file at {escape(str(feed_path))}.[/red]")
        console.print(escape(PATH_HINT))
        return 1
    if args.topic and feed_path is not None:
        config.organism.research.own_article = str(feed_path)  # a new life reads its beginning to learn what the topic means
    # A paper you share as yours (--feed-file) is not read a second time as one of the papers.
    if args.papers and ingest(config, args.papers, skip={feed_path.resolve()} if feed_path else None):
        console.print("[red]No papers could be read; check the folder.[/red]")
        return 1
    if args.topic and not (home / "mind.json").exists():
        console.print(f"[dim]A new researcher life will be born in {escape(str(home))}; it first prepares its questions.[/dim]")
        if not check_ollama(config):
            return 1

    organism = CuriosityOrganism(config)
    organism.body.log = lambda message: console.print(f"[blue]{escape(message)}[/blue]")
    if organism.newborn:
        console.print(f"[green]{escape(organism.state.name)} is born[/green] to study [bold]{escape(organism.state.topic.title)}[/bold] "
                      f"with {len(organism.state.questions)} questions:")
        for q in organism.state.questions.values():
            console.print(f"  {q.id} {escape(q.text)}")
        if organism.state.topic.meaning:
            console.print(f"It takes the topic to mean: [italic]{escape(organism.state.topic.meaning)}[/italic]")
            console.print("[dim]If that is not what you mean, start again with --new-life and write the topic in your field's words.[/dim]")
    elif args.topic and organism.state.topic.description != one_line(args.topic, 600):
        console.print(f"[yellow]This life already studies '{escape(organism.state.topic.title)}'; it continues. "
                      "Use --new-life (or another --home) for a new topic.[/yellow]")
    if args.map:
        console.print(f"Research map: {escape(str(write_research_map(organism)))}")
        return 0

    for question in args.ask:
        q = organism.ask(question)
        console.print(f"Asked: [bold]{q.id}[/bold] {escape(q.text)}")
    if args.feed or feed_path:
        if feed_path is not None:
            title, text = read_shared_file(feed_path)
            title = args.title or title
        else:
            title, text = args.title, args.feed
        if not text.strip():
            console.print(f"[red]No text found in {escape(str(feed_path))} (a scanned PDF needs OCR first).[/red]")
            return 1
        path = organism.feed(text, title, own=organism.research)  # in researcher mode: your own work, a claim to test
        console.print(f"Shared with it: {escape(str(path))} (it will be noticed at the next heartbeat)")

    if args.status:
        print_status(organism)
        return 0

    exam_path = None
    if args.exam:
        exam_path = Path(config.organism.exam_path if args.exam == "default" else args.exam)
        if args.exam == "default" and organism.research:
            console.print("[red]The default exam is about the philosophy of curiosity. Write one for your topic and pass it: --exam my_exam.json[/red]")
            return 1
        if not exam_path.is_file():
            console.print(f"[red]No exam file at {escape(str(exam_path))}[/red]")
            return 1
    living = bool(args.forever or args.minutes or args.heartbeats is not None)
    if exam_path and not living:
        if not check_ollama(config):
            return 1
        print_exam(run_exam(organism, exam_path, label="exam"))
        return 0

    talked = bool(args.ask or args.feed or args.feed_file or args.new_life or args.papers)
    wants_to_live = args.forever or args.minutes or args.heartbeats is not None or args.reflect or not talked
    if not wants_to_live:
        console.print("Run [bold]python live.py[/bold] to let it think about this.")
        return 0
    if not check_ollama(config):
        return 1
    console.print(f"[dim]{escape(organism.body.describe())}[/dim]")
    if organism.librarian is not None:
        console.print("[dim]Web library: on (Wikipedia, Project Gutenberg, Semantic Scholar, arXiv) - it looks things up when its texts are silent.[/dim]")
    console.print(f"[dim]Judge: {'off (self-graded)' if organism.judge is None else (config.organism.judge.model or config.llm.model)}[/dim]")
    if exam_path:
        console.print(f"[dim]Exam before and after: {escape(str(exam_path))} (exam time does not count toward the session).[/dim]")
    if args.minutes:
        console.print(f"[dim]Living for {args.minutes:g} minutes, then it stops and writes a report. Ctrl+C stops earlier.[/dim]")

    code = 0
    try:
        if args.reflect:
            organism.reflect()
            sm = organism.state.self_model
            console.print(f"[magenta]{DIAGNOSES[sm.diagnosis]['name']}[/magenta]: {escape(sm.last_reflection)}")
            console.print(f"[bold]Curiosity, as it now understands it:[/bold] {escape(sm.understanding_of_curiosity)}")
        if args.forever or args.minutes or args.heartbeats is not None or not args.reflect:
            organism.run_session(
                args.heartbeats,
                forever=args.forever,
                minutes=args.minutes,
                label=args.label,
                on_heartbeat=lambda ep: print_heartbeat(organism, ep),
                exam_path=exam_path,
            )
    except KeyboardInterrupt:
        console.print("\n[dim]Paused. The mind is saved; run again to continue its life.[/dim]")
    except OrganismError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        code = 1
    if organism.last_report is not None:
        print_session(organism.last_report)
    console.print(f"Diary: {escape(str(organism.home / 'diary.md'))}")
    return code


if __name__ == "__main__":
    sys.exit(main())
