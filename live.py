"""Live, talk to, and observe the curiosity organism (v8).

Examples:
    python live.py                                   # live a few heartbeats
    python live.py --heartbeats 20
    python live.py --forever --pause 30              # keep living until Ctrl+C
    python live.py --ask "Can a machine be curious, or only act as if it were?"
    python live.py --feed-file my_notes.txt --title "My notes on boredom"
    python live.py --status                          # look inside its mind
    python live.py --new-life                        # archive this life and start again
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from curiosity_ai.config import AppConfig, load_config
from curiosity_ai.organism import CuriosityOrganism, OrganismError
from curiosity_ai.organism.drive import DIAGNOSES
from curiosity_ai.organism.state import Episode

console = Console()


def check_ollama(config: AppConfig) -> bool:
    base = config.llm.base_url.rstrip("/")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        console.print(f"[red]Cannot reach Ollama at {base}[/red] ({exc}).\nStart it with:  [bold]ollama serve[/bold]")
        return False
    names = {m.get("name", "") for m in resp.json().get("models", [])}
    model = config.llm.model
    if model in names or f"{model}:latest" in names or any(n.startswith(model + ":") for n in names):
        return True
    console.print(f"[red]Model '{model}' is not installed in Ollama.[/red] Install it with:  [bold]ollama pull {model}[/bold]")
    return False


def print_heartbeat(organism: CuriosityOrganism, ep: Episode | None) -> None:
    if ep is None:
        console.print("[dim]No open question; the organism rests.[/dim]")
        return
    if ep.status_after == "no-thought":
        console.print(f"[red][heartbeat {ep.heartbeat}] could not think:[/red] {'; '.join(ep.errors)}")
        return
    console.print(
        f"[bold cyan][heartbeat {ep.heartbeat}][/bold cyan] {ep.question_id} "
        f"[dim](curiosity {ep.drive.get('total', 0):.2f})[/dim] {ep.question}"
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
    console.print("   " + " | ".join(parts))
    if ep.insight:
        console.print(f"   [italic]{ep.insight}[/italic]")
    if ep.errors:
        console.print(f"   [yellow]trouble: {'; '.join(ep.errors)}[/yellow]")
    if ep.heartbeat % max(1, organism.oc.reflect_every) == 0 and organism.oc.reflect_every > 0:
        sm = organism.state.self_model
        console.print(f"   [magenta]reflection: {DIAGNOSES[sm.diagnosis]['name']}[/magenta] - {sm.last_reflection}")


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
            f"[bold]{st.name}[/bold], {st.heartbeat} heartbeats old (born {st.born_at[:10]})\n"
            f"Questions: {', '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'none'}\n"
            f"Beliefs: {', '.join(f'{k} {v}' for k, v in sorted(belief_counts.items())) or 'none'}\n"
            f"State of curiosity: [magenta]{diagnosis['name']}[/magenta] - {diagnosis['meaning']}\n"
            f"Temperament: gap {t.gap:.2f}, learning progress {t.learning_progress:.2f}, surprise {t.surprise:.2f}, "
            f"novelty {t.novelty:.2f}, importance {t.importance:.2f}, patience {t.boredom_patience}, "
            f"exploration {t.exploration_temperature:.2f}\n\n"
            f"[bold]What it thinks curiosity is:[/bold] {st.self_model.understanding_of_curiosity}",
            title="Curiosity organism",
        )
    )
    table = Table(title="Open questions, by the pull they exert")
    for col in ("id", "pull", "gap", "progress", "surprise", "novelty", "bored", "conf", "question"):
        table.add_column(col, justify="right" if col not in ("id", "question") else "left")
    for q, r in organism.drives()[:12]:
        table.add_row(
            q.id, f"{r.total:.2f}", f"{r.gap:.2f}", f"{r.learning_progress:.2f}", f"{r.surprise:.2f}",
            f"{r.novelty:.2f}", f"{r.boredom:.2f}", f"{q.confidence:.2f}", q.text[:90],
        )
    console.print(table)
    beliefs = sorted(st.held_beliefs(), key=lambda b: -b.confidence)[:10]
    if beliefs:
        btable = Table(title="Strongest beliefs")
        for col in ("id", "conf", "grounding", "belief"):
            btable.add_column(col)
        for b in beliefs:
            grounding = f"{len(b.evidence)} quote(s)" if b.evidence else "interpretation"
            btable.add_row(b.id, f"{b.confidence:.2f}", grounding, b.statement[:110])
        console.print(btable)
    console.print(f"Diary: {organism.home / 'diary.md'}   Mind: {organism.mind_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live, talk to, and observe the curiosity organism.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--heartbeats", type=int, default=None, help="How many acts of inquiry to live now")
    parser.add_argument("--forever", action="store_true", help="Keep living until Ctrl+C")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to rest between heartbeats")
    parser.add_argument("--ask", action="append", default=[], help="Give the organism a question (repeatable)")
    parser.add_argument("--feed", default=None, help="Share an observation as text")
    parser.add_argument("--feed-file", default=None, help="Share an observation from a text/markdown file")
    parser.add_argument("--title", default=None, help="Title for --feed/--feed-file")
    parser.add_argument("--status", action="store_true", help="Show the organism's mind and exit")
    parser.add_argument("--reflect", action="store_true", help="Reflect now")
    parser.add_argument("--new-life", action="store_true", help="Archive the current life and start a new one")
    parser.add_argument("--model", default=None, help="Override llm.model (e.g. mistral:7b-instruct, qwen2.5:3b)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible choices")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    if args.model:
        config.llm.model = args.model
    if args.seed is not None:
        config.organism.random_seed = args.seed

    if args.new_life:
        archived = CuriosityOrganism.archive(config)
        console.print(f"Previous life archived to {archived}." if archived else "There was no previous life to archive.")

    organism = CuriosityOrganism(config)
    if organism.newborn:
        console.print(f"[green]{organism.state.name} is born[/green] with {len(organism.state.questions)} questions.")

    for question in args.ask:
        q = organism.ask(question)
        console.print(f"Asked: [bold]{q.id}[/bold] {q.text}")
    if args.feed_file and not Path(args.feed_file).is_file():
        console.print(f"[red]No such file: {args.feed_file}[/red]")
        return 1
    if args.feed or args.feed_file:
        text = args.feed or Path(args.feed_file).read_text(encoding="utf-8", errors="ignore")
        title = args.title or (Path(args.feed_file).stem if args.feed_file else None)
        path = organism.feed(text, title)
        console.print(f"Shared an observation: {path} (it will be noticed at the next heartbeat)")

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

    try:
        if args.reflect:
            organism.reflect()
            sm = organism.state.self_model
            console.print(f"[magenta]{DIAGNOSES[sm.diagnosis]['name']}[/magenta]: {sm.last_reflection}")
            console.print(f"[bold]Curiosity, as it now understands it:[/bold] {sm.understanding_of_curiosity}")
        if args.forever or args.heartbeats is not None or not args.reflect:
            organism.live(
                args.heartbeats,
                forever=args.forever,
                pause_seconds=args.pause,
                on_heartbeat=lambda ep: print_heartbeat(organism, ep),
            )
    except KeyboardInterrupt:
        organism.save()
        console.print("\n[dim]Paused. The mind is saved; run again to continue its life.[/dim]")
    except OrganismError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    console.print(f"Diary: {organism.home / 'diary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
