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
"""
from __future__ import annotations

import argparse
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
from curiosity_ai.organism.drive import DIAGNOSES
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
    if _is_model(names, model):
        return True
    console.print(f"[red]Model '{escape(model)}' is not installed in Ollama.[/red] Install it with:  [bold]ollama pull {escape(model)}[/bold]")
    return False


def _is_model(names: set[str], model: str) -> bool:
    return model in names or f"{model}:latest" in names or any(n.startswith(model + ":") for n in names)


def run_checks(config: AppConfig) -> int:
    """Check this computer's setup, one line per item, and say what to fix."""
    rows: list[tuple[str, str, str]] = []

    v = sys.version_info
    rows.append(("ok" if v >= (3, 10) else "FIX", "Python", f"{v.major}.{v.minor}.{v.micro}" + ("" if v >= (3, 10) else ": needs 3.10 or newer")))
    try:
        import chromadb

        rows.append(("ok", "chromadb", f"{chromadb.__version__} (used by run.py and retrieval: chroma)"))
    except Exception:
        rows.append(("note", "chromadb", "not installed: only needed for the v7 run.py or retrieval: chroma"))

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
    console.print(
        Panel(
            f"[bold]{escape(st.name)}[/bold], {st.heartbeat} heartbeats old (born {st.born_at[:10]})\n"
            f"Questions: {', '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'none'}\n"
            f"Beliefs: {', '.join(f'{k} {v}' for k, v in sorted(belief_counts.items())) or 'none'}\n"
            f"State of curiosity: [magenta]{diagnosis['name']}[/magenta] - {diagnosis['meaning']}\n"
            f"Temperament: gap {t.gap:.2f}, learning progress {t.learning_progress:.2f}, surprise {t.surprise:.2f}, "
            f"novelty {t.novelty:.2f}, importance {t.importance:.2f}, patience {t.boredom_patience}, "
            f"exploration {t.exploration_temperature:.2f}\n\n"
            f"[bold]What it thinks curiosity is:[/bold] {escape(st.self_model.understanding_of_curiosity)}",
            title="Curiosity organism",
        )
    )
    table = Table(title="Open questions, by the pull they exert")
    for col in ("id", "pull", "gap", "progress", "surprise", "novelty", "bored", "conf", "question"):
        table.add_column(col, justify="right" if col not in ("id", "question") else "left")
    for q, r in organism.drives()[:12]:
        table.add_row(
            q.id, f"{r.total:.2f}", f"{r.gap:.2f}", f"{r.learning_progress:.2f}", f"{r.surprise:.2f}",
            f"{r.novelty:.2f}", f"{r.boredom:.2f}", f"{q.confidence:.2f}", escape(one_line(q.text, 120)),
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
    console.print(f"Diary: {escape(str(organism.home / 'diary.md'))}   Mind: {escape(str(organism.mind_path))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live, talk to, and observe the curiosity organism.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--heartbeats", type=int, default=None, help="How many acts of inquiry to live now")
    parser.add_argument("--forever", action="store_true", help="Keep living until Ctrl+C")
    parser.add_argument("--pause", type=float, default=None, help="Seconds to breathe between heartbeats (organism.body.breath_seconds)")
    parser.add_argument("--no-rest", action="store_true", help="Turn off rests and the temperature/battery guards (for short experiments only)")
    parser.add_argument("--ask", action="append", default=[], help="Give the organism a question (repeatable)")
    parser.add_argument("--feed", default=None, help="Share an observation as text")
    parser.add_argument("--feed-file", default=None, help="Share an observation from a text/markdown file")
    parser.add_argument("--title", default=None, help="Title for --feed/--feed-file")
    parser.add_argument("--status", action="store_true", help="Show the organism's mind and exit")
    parser.add_argument("--reflect", action="store_true", help="Reflect now")
    parser.add_argument("--new-life", action="store_true", help="Archive the current life and start a new one")
    parser.add_argument("--model", default=None, help="Override llm.model (e.g. mistral:7b-instruct, qwen2.5:3b)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible choices")
    parser.add_argument("--check", action="store_true", help="Check this computer's setup (Python, Ollama, model, GPU, power) and exit")
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
    if args.check:
        return run_checks(config)

    if args.new_life:
        archived = CuriosityOrganism.archive(config)
        console.print(f"Previous life archived to {escape(str(archived))}." if archived else "There was no previous life to archive.")

    organism = CuriosityOrganism(config)
    organism.body.log = lambda message: console.print(f"[blue]{escape(message)}[/blue]")
    if organism.newborn:
        console.print(f"[green]{escape(organism.state.name)} is born[/green] with {len(organism.state.questions)} questions.")

    for question in args.ask:
        q = organism.ask(question)
        console.print(f"Asked: [bold]{q.id}[/bold] {escape(q.text)}")
    if args.feed_file and not Path(args.feed_file).is_file():
        console.print(f"[red]No such file: {escape(args.feed_file)}[/red]")
        return 1
    if args.feed or args.feed_file:
        text = args.feed or Path(args.feed_file).read_text(encoding="utf-8", errors="ignore")
        title = args.title or (Path(args.feed_file).stem if args.feed_file else None)
        path = organism.feed(text, title)
        console.print(f"Shared an observation: {escape(str(path))} (it will be noticed at the next heartbeat)")

    if args.status:
        print_status(organism)
        return 0

    talked = bool(args.ask or args.feed or args.feed_file or args.new_life)
    wants_to_live = args.forever or args.heartbeats is not None or args.reflect or not talked
    if not wants_to_live:
        console.print("Run [bold]python live.py[/bold] to let it think about this.")
        return 0
    if not check_ollama(config):
        return 1
    console.print(f"[dim]{escape(organism.body.describe())}[/dim]")

    try:
        if args.reflect:
            organism.reflect()
            sm = organism.state.self_model
            console.print(f"[magenta]{DIAGNOSES[sm.diagnosis]['name']}[/magenta]: {escape(sm.last_reflection)}")
            console.print(f"[bold]Curiosity, as it now understands it:[/bold] {escape(sm.understanding_of_curiosity)}")
        if args.forever or args.heartbeats is not None or not args.reflect:
            organism.live(
                args.heartbeats,
                forever=args.forever,
                on_heartbeat=lambda ep: print_heartbeat(organism, ep),
            )
    except KeyboardInterrupt:
        organism.save()
        console.print("\n[dim]Paused. The mind is saved; run again to continue its life.[/dim]")
    except OrganismError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return 1
    console.print(f"Diary: {escape(str(organism.home / 'diary.md'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
