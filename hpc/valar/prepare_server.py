"""Get an Ollama server ready for a run on a computing cluster.

    python hpc/valar/prepare_server.py --url http://127.0.0.1:11500
    python hpc/valar/prepare_server.py --url http://127.0.0.1:11500 --models qwen2.5:7b qwen2.5:32b

It waits until the server answers. With --models it checks that they are installed, loads them into memory
(where they stay for the whole job, so loading does not count toward the run's time) and says whether each
sits on the GPU, from the server's log (--log) where it can. Exit code 0: ready. 2: the server does not
answer or a model is missing. 3: a model sits partly on the CPU, which makes it many times slower
(allowed with --allow-cpu).
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import requests


def wait(url: str, seconds: float) -> str | None:
    """The server's version once it answers, or None if it does not answer in time."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            reply = requests.get(f"{url}/api/version", timeout=5)
            if reply.ok:
                return reply.json().get("version", "?")
        except requests.RequestException:
            pass
        time.sleep(1)
    return None


def installed(url: str) -> set[str]:
    return {m.get("name", "") for m in requests.get(f"{url}/api/tags", timeout=30).json().get("models", [])}


def known_as(names: set[str] | dict, model: str) -> str | None:
    """The name under which Ollama knows a model ("mistral" is "mistral:latest")."""
    for name in (model, f"{model}:latest"):
        if name in names:
            return name
    return None


def load(url: str, model: str, num_ctx: int) -> float:
    """Load a model into memory for good; returns the seconds it took.

    With the context length the run will ask for: with another, its first question would load it again.
    """
    start = time.monotonic()
    reply = requests.post(f"{url}/api/generate", json={"model": model, "keep_alive": -1, "options": {"num_ctx": num_ctx}},
                          timeout=1800)
    reply.raise_for_status()
    return time.monotonic() - start


# Every model load writes a line like "load_tensors: offloaded 29/29 layers to GPU" to the server's log.
OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
# Without such a line: the share of the model's memory that is on the GPU. A model wholly on the GPU can show
# less than 100%, because its token table and some work space stay in the computer's memory.
LEAST_SHARE = 0.8


def log_end(log: Path | None) -> int:
    return log.stat().st_size if log is not None and log.is_file() else 0


def offloaded_since(log: Path | None, start: int) -> tuple[int, int] | None:
    """(layers on the GPU, layers) from the last load written to the log after ``start``."""
    if log is None or not log.is_file():
        return None
    with log.open("rb") as f:
        f.seek(start)
        found = OFFLOADED.findall(f.read().decode("utf-8", "replace"))
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def residency(url: str) -> dict[str, tuple[float, float]]:
    """Loaded model -> (size in GB, share of it on the GPU)."""
    loaded = {}
    for m in requests.get(f"{url}/api/ps", timeout=30).json().get("models", []):
        size = m.get("size") or 0
        loaded[m.get("name", "")] = (size / 1e9, (m.get("size_vram") or 0) / size if size else 0.0)
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Get an Ollama server ready for a run.")
    parser.add_argument("--url", required=True, help="e.g. http://127.0.0.1:11500")
    parser.add_argument("--models", nargs="*", default=[], help="models to check and load (the organism's and the judge's)")
    parser.add_argument("--wait", type=float, default=180, help="seconds to wait for the server to answer")
    parser.add_argument("--allow-cpu", action="store_true", help="go on even if a model sits partly on the CPU")
    parser.add_argument("--log", default=None, help="the server's log, to read how many layers went to the GPU")
    parser.add_argument("--num-ctx", type=int, default=8192, help="the context length the run uses (llm.num_ctx)")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    version = wait(url, args.wait)
    if version is None:
        print(f"Ollama did not answer at {url} within {args.wait:g} s; its log says why.")
        return 2
    print(f"Ollama {version} answers at {url}.")
    models = list(dict.fromkeys(args.models))  # the judge may be the organism's own model
    if not models:
        return 0

    names = installed(url)
    missing = [m for m in models if known_as(names, m) is None]
    if missing:
        print(f"Not downloaded yet: {', '.join(missing)}. On the login node run:  bash hpc/valar/setup.sh models {' '.join(missing)}")
        return 2
    log = Path(args.log) if args.log else None
    layers: dict[str, tuple[int, int] | None] = {}
    for model in models:
        print(f"Loading {model} ...", flush=True)
        start = log_end(log)
        seconds = load(url, model, args.num_ctx)
        time.sleep(1)  # let the server finish writing its log
        layers[model] = offloaded_since(log, start)
        print(f"  loaded in {seconds:.0f} s", flush=True)

    loaded = residency(url)
    partly_on_cpu = False
    for model in models:
        name = known_as(loaded, model)
        size, share = loaded[name] if name else (0.0, 0.0)
        if layers[model] is not None:
            on_gpu, total = layers[model]
            partly_on_cpu |= on_gpu < total
            where = f"{on_gpu} of {total} layers on the GPU"
        else:
            partly_on_cpu |= name is not None and share < LEAST_SHARE
            where = f"{share:.0%} of its memory on the GPU" if name else "layers on the GPU not known"
        print(f"{model}: {where}" + (f", {size:.1f} GB in memory" if name else ""))
        if name is None:
            print(f"  {model} is not in memory now: the GPUs cannot hold all the models at once, so they will "
                  "take turns, which is slower; more GPUs would help.")
    if partly_on_cpu and not args.allow_cpu:
        if all(v is not None and v[0] == 0 for v in layers.values()):
            print("No layer went to a GPU at all: check that the job has a GPU (the gpus line of run_info.txt) and "
                  "what the server's log says after 'discovering available GPUs'.")
        else:
            print("A model does not fit on the GPU(s) and would run partly on the CPU, many times slower. Ask for "
                  "more or larger GPUs (GPU_TYPE and GPU_MEM_GB in hpc/valar/my_settings.sh).")
        print("To run anyway, many times slower: ALLOW_CPU=1.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
