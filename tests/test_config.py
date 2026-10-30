from pathlib import Path

from curiosity_ai.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_config():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.llm.model
    assert cfg.loop.max_iterations > 0


def test_organism_settings_have_defaults_and_load_from_yaml():
    for name in ("config.yaml", "config_hybrid_web.yaml"):
        cfg = load_config(ROOT / name)
        assert cfg.organism.seed_questions
        assert cfg.organism.retrieval in {"lexical", "chroma"}
        weights = cfg.organism.temperament
        assert abs(weights.gap + weights.learning_progress + weights.surprise + weights.novelty + weights.importance - 1.0) < 1e-6
