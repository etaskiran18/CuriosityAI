"""The cluster scripts in hpc/valar: GPU sizing and gathering a batch of runs to share."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from curiosity_ai.organism import CuriosityOrganism

from .organism_fakes import ScriptedLLM

VALAR = Path(__file__).resolve().parents[1] / "hpc" / "valar"


def _collect_module():
    spec = importlib.util.spec_from_file_location("valar_collect", VALAR / "collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bash(command: str, store: Path) -> str:
    env = {**os.environ, "STORE": str(store)}
    return subprocess.run(["bash", "-c", f"source '{VALAR / 'common.sh'}' && {command}"],
                          capture_output=True, text=True, check=True, env=env).stdout.strip()


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
@pytest.mark.parametrize("model, judge, gpus", [
    ("qwen2.5:3b", "qwen2.5:32b", "1"),
    ("qwen2.5:14b", "qwen2.5:32b", "1"),
    ("qwen2.5:32b", "qwen2.5:32b", "1"),   # the judge is the organism's own model: loaded once
    ("qwen2.5:72b", "qwen2.5:32b", "2"),   # about 78 GB: two 48 GB cards
    ("mistral:7b-instruct", "qwen2.5:32b", "1"),
    ("mistral-large:123b", "qwen2.5:32b", "3"),
    ("llama3.1:405b", "qwen2.5:32b", "7"),
])
def test_a_job_asks_for_as_many_48_gb_gpus_as_its_models_need(model, judge, gpus, tmp_path):
    assert _bash(f"GPU_MEM_GB=48 gpus_needed {model} {judge}", tmp_path) == gpus


@needs_bash
def test_the_size_of_a_downloaded_model_comes_from_its_manifest(tmp_path):
    """A name need not say the size (mistral-large is 123B): what was downloaded does."""
    manifest = tmp_path / "models" / "manifests" / "registry.ollama.ai" / "library" / "mistral-large" / "latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [{"size": 72_000_000_000}, {"size": 1_000_000_000}, {"size": 500}]}))
    assert _bash("memory_needed mistral-large qwen2.5:32b", tmp_path) == "104"  # 73 + 3 + 2.9, and the judge's 25
    assert _bash("GPU_MEM_GB=48 gpus_needed mistral-large qwen2.5:32b", tmp_path) == "3"
    assert _bash("ram_needed mistral-large qwen2.5:32b; ram_needed qwen2.5:3b qwen2.5:3b", tmp_path) == "130\n64"


@needs_bash
def test_model_sizes_are_read_from_their_names(tmp_path):
    assert _bash("model_billions qwen2.5:72b; model_billions qwen2.5:1.5b; model_billions mistral", tmp_path) == "72\n1.5\n8"
    assert _bash("slug qwen2.5:7b", tmp_path) == "qwen2.5-7b"
    assert _bash("GPU_MEM_GB=80 gpus_needed qwen2.5:32b qwen2.5:14b", tmp_path) == "1"


def _run(batch: Path, name: str, config, model: str) -> None:
    run = batch / name
    run.mkdir(parents=True)
    (run / "run_info.txt").write_text(f"model: {model}\njudge: qwen2.5:32b\ngpus: NVIDIA L40S, 46068 MiB\n"
                                      "finished: 2026-10-26 16:00:00, exit code 0\n", encoding="utf-8")
    config.organism.home = str(run / "life")
    CuriosityOrganism(config, llm=ScriptedLLM()).run_session(1, label=name)


def test_a_batch_is_gathered_for_reading_and_sharing(config, tmp_path):
    batch = tmp_path / "runs" / "valar" / "ladder"
    _run(batch, "qwen2.5-14b", config, "qwen2.5:14b")
    _run(batch, "qwen2.5-3b", config, "qwen2.5:3b")
    traps = batch / "traps-qwen2.5-3b" / "traps"
    traps.mkdir(parents=True)
    (traps / "trap_results.md").write_text("# Trap tests\n", encoding="utf-8")
    (batch / "failed-before-start").mkdir()  # a run that wrote nothing is left out

    archive = _collect_module().collect(batch)
    share = batch / "share"
    assert sorted(p.name for p in share.iterdir()) == [
        "README.md", "comparison.csv", "comparison.md", "qwen2.5-14b", "qwen2.5-3b", "traps-qwen2.5-3b"]
    files = {p.name for p in (share / "qwen2.5-3b").iterdir()}
    assert {"diary.md", "research_map.md", "report.md", "metrics.json", "judge_check.csv", "run_info.txt"} <= files
    assert (share / "traps-qwen2.5-3b" / "trap_results.md").is_file()

    readme = (share / "README.md").read_text(encoding="utf-8")
    assert readme.index("| qwen2.5-3b |") < readme.index("| qwen2.5-14b |")  # smaller models first
    assert "| qwen2.5-3b | qwen2.5:3b | qwen2.5:32b | 1 |" in readme and "What to look for in a diary" in readme
    table = (share / "comparison.md").read_text(encoding="utf-8")
    assert table.startswith("| metric |") and "Written to" not in table and "heartbeats per hour" in table
    with tarfile.open(archive) as tar:
        assert "ladder-share/qwen2.5-3b/diary.md" in tar.getnames()
