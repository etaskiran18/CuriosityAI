from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "mistral:7b-instruct"
    temperature: float = 0.35
    top_p: float = 0.9
    num_ctx: int = 8192
    timeout_seconds: int = 240
    json_repair_attempts: int = 2


class LoopConfig(BaseModel):
    max_iterations: int = 5
    min_curiosity_score: float = 0.35
    stop_if_redundancy_above: float = 0.82
    stop_if_no_progress_for: int = 2
    short_term_memory_items: int = 10
    candidate_questions_per_iteration: int = 8
    next_topics_per_iteration: int = 5


class CorpusConfig(BaseModel):
    path: str = "data/philosophy_corpus"
    chunk_size_chars: int = 2400
    chunk_overlap_chars: int = 450
    accepted_extensions: list[str] = Field(default_factory=lambda: [".txt", ".md", ".markdown"])
    auto_ingest_on_start: bool = True


class MemoryConfig(BaseModel):
    backend: str = "chroma"
    persist_dir: str = "memory/chroma"
    journal_path: str = "memory/journal_v6.jsonl"
    collection_prefix: str = "curiosity_ai_v6"
    top_k_passages: int = 12
    top_k_memories: int = 3


class WebGroundingConfig(BaseModel):
    enabled: bool = False
    # tavily = live web snippets; crossref/semantic_scholar = academic metadata; hybrid = all available
    provider: str = "tavily"
    max_results: int = 5
    search_depth: str = "basic"
    include_answer: bool = False
    include_raw_content: bool = False
    strict_citations: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    timeout_seconds: int = 30
    tavily_api_key_env: str = "TAVILY_API_KEY"
    semantic_scholar_api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"
    academic_search_enabled: bool = True
    academic_max_results: int = 3
    use_web_when_source_gate_limited: bool = True


class AgentConfig(BaseModel):
    enabled: bool = True
    exploration_temperature: float | None = None


class AgentsConfig(BaseModel):
    reader: AgentConfig = Field(default_factory=AgentConfig)
    explorer: AgentConfig = Field(default_factory=AgentConfig)
    radicalizer: AgentConfig = Field(default_factory=AgentConfig)
    critic: AgentConfig = Field(default_factory=AgentConfig)
    research: AgentConfig = Field(default_factory=AgentConfig)
    textual_analyst: AgentConfig = Field(default_factory=AgentConfig)
    argument_reconstructor: AgentConfig = Field(default_factory=AgentConfig)
    position_enforcer: AgentConfig = Field(default_factory=AgentConfig)
    balance_stress_tester: AgentConfig = Field(default_factory=AgentConfig)
    synthesizer: AgentConfig = Field(default_factory=AgentConfig)
    argument_critic: AgentConfig = Field(default_factory=AgentConfig)
    philosophical_depth_judge: AgentConfig = Field(default_factory=AgentConfig)
    philosophical_move_detector: AgentConfig = Field(default_factory=AgentConfig)
    curiosity_theory_tracker: AgentConfig = Field(default_factory=AgentConfig)
    evaluator: AgentConfig = Field(default_factory=AgentConfig)
    planner: AgentConfig = Field(default_factory=AgentConfig)


class ScoringConfig(BaseModel):
    # v6: novelty and researchability are useful, but philosophical quality matters more.
    novelty_weight: float = 0.12
    depth_weight: float = 0.24
    epistemic_tension_weight: float = 0.24
    researchability_weight: float = 0.10
    generativity_weight: float = 0.08
    citation_grounding_weight: float = 0.18
    redundancy_penalty_weight: float = 0.45
    depth_quality_weight: float = 0.30
    argument_quality_weight: float = 0.25
    curiosity_progress_weight: float = 0.20
    hedging_penalty_weight: float = 0.18
    unsupported_inference_penalty_weight: float = 0.22


class DepthConfig(BaseModel):
    enabled: bool = True
    min_primary_sources: int = 2
    min_textual_analyses: int = 3
    min_synthesis_words: int = 1400
    target_synthesis_words: int = 1900
    max_synthesis_words: int = 2800
    block_synthesis_without_primary_sources: bool = False
    require_argument_reconstruction: bool = True
    require_post_synthesis_critique: bool = True
    prefer_primary_over_memory: bool = True
    min_philosophical_depth_score: float = 0.76
    min_curiosity_progress_score: float = 0.70
    block_if_curiosity_theme_missing: bool = True
    require_depth_judge: bool = True
    require_real_philosophical_move: bool = True


class AggressiveModeConfig(BaseModel):
    enabled: bool = True
    radical_questions: int = 4
    require_strong_position: bool = True
    require_binary_or_thesis_position: bool = True
    require_curiosity_continuity: bool = True
    forbid_safe_synthesis: bool = True
    banned_safe_patterns: list[str] = Field(default_factory=lambda: [
        "could potentially", "might suggest", "may suggest", "it can be interpreted",
        "possibly", "perhaps", "in some way", "to some extent", "it remains unclear"
    ])
    hedge_words: list[str] = Field(default_factory=lambda: [
        "might", "could", "may", "possibly", "potentially", "perhaps", "seems",
        "suggests", "it can be inferred", "to some extent", "in some way"
    ])
    max_hedge_ratio: float = 0.030
    min_real_move_score: float = 0.72


class BalanceConfig(BaseModel):
    enabled: bool = True
    trigger_words: list[str] = Field(default_factory=lambda: ["balanced", "balance", "tempered", "moderated", "guided", "disciplined"] )
    require_if_position_is_conditional: bool = True
    force_revision_if_balance_undefined: bool = True


class RevisionConfig(BaseModel):
    enabled: bool = True
    max_revision_rounds: int = 2
    revise_if_depth_below: float = 0.76
    revise_if_curiosity_progress_below: float = 0.70
    revise_if_move_below: float = 0.72
    force_same_topic_on_failure: bool = True
    same_topic_prefix: str = "REVISION REQUIRED — deepen without drifting: "


class CuriosityStateConfig(BaseModel):
    enabled: bool = True
    path: str = "memory/curiosity_theory_state_v6.json"
    require_property_update: bool = True


class DatasetConfig(BaseModel):
    enabled: bool = True
    output_path: str = "datasets/curiosity_sft_v6.jsonl"
    include_web_sources: bool = False
    include_low_score_items: bool = False
    min_score_for_dataset: float = 0.80
    use_evaluation_score: bool = True
    require_philosophical_depth_pass: bool = True
    require_real_move_pass: bool = True
    require_strong_position: bool = True


class ReportsConfig(BaseModel):
    output_dir: str = "reports"
    write_iteration_reports: bool = True
    write_final_report: bool = True
    include_citations: bool = True


class PublishingConfig(BaseModel):
    enabled: bool = False
    require_human_approval: bool = True
    min_score_to_queue: float = 0.82
    queue_dir: str = "publications/pending"
    published_dir: str = "publications/published"
    platform: str = "local_markdown"
    public_site_dir: str = "publications/site"
    webhook_url_env: str = "CURIOSITY_PUBLISH_WEBHOOK"


class TemperamentConfig(BaseModel):
    """Starting weights of the organism's curiosity drive (see organism/drive.py)."""
    gap: float = 0.30
    learning_progress: float = 0.30
    surprise: float = 0.15
    novelty: float = 0.10
    importance: float = 0.15
    boredom_patience: int = 3
    exploration_temperature: float = 0.15


class BodyConfig(BaseModel):
    """Taking care of the computer the organism lives in (see organism/body.py)."""
    enabled: bool = True
    breath_seconds: float = 10          # short pause after every heartbeat
    work_minutes: float = 20            # after this much thinking...
    rest_minutes: float = 5             # ...rest this long (0 turns the rhythm off)
    gpu_temperature_guard: bool = True  # read the NVIDIA GPU temperature with nvidia-smi
    max_gpu_temp_c: float = 80          # stop thinking at this GPU temperature...
    resume_gpu_temp_c: float = 65       # ...until the GPU has cooled to this
    check_every_seconds: float = 15
    pause_on_battery: bool = True       # laptops: think only while plugged in

    @model_validator(mode="after")
    def resume_below_max(self):
        if self.resume_gpu_temp_c >= self.max_gpu_temp_c:
            self.resume_gpu_temp_c = self.max_gpu_temp_c - 10
        return self


class OrganismConfig(BaseModel):
    """v8 curiosity organism: a persistent mind whose behavior is driven by curiosity."""
    home: str = "memory/organism"
    name: str = "Curiosity"
    # lexical = BM25 over the corpus (no extra dependencies); chroma = reuse v7 vector memory
    retrieval: str = "lexical"
    # Compilations of excerpts would duplicate the primary texts and hide who really said what.
    library_exclude: list[str] = Field(default_factory=lambda: ["curiosity_core_reader.md"])
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 200
    evidence_passages: int = 5
    max_passages_per_source: int = 2
    passage_chars: int = 1100
    max_web_results: int = 2
    web_min_relevance: float = 0.25
    dialogue_turns: int = 3
    heartbeats_per_run: int = 5
    reflect_every: int = 5
    random_seed: int | None = None
    seed_questions: list[str] = Field(default_factory=lambda: [
        "Plato says philosophy begins in wonder: what exactly is wonder, and is it the same thing as curiosity?",
        "Meno asks how anyone can search for what they do not know. How does curiosity get started if we cannot recognise what we lack?",
        "Dewey says curiosity becomes intellectual when a question is held open in one's own mind. What keeps a question alive, and what kills it?",
        "When does curiosity turn into a vice, a restless hunger for novelty rather than a love of understanding?",
        "Is doubt the engine of inquiry or its enemy?",
    ])
    newborn_confidence: float = 0.25
    max_initial_confidence: float = 0.5
    max_confidence_step: float = 0.25
    # You cannot be more confident than your evidence allows:
    # ceiling = base + per_support * (verified confirmations gathered for the question)
    evidence_ceiling_base: float = 0.5
    evidence_ceiling_per_support: float = 0.1
    settle_confidence: float = 0.8
    settle_max_error: float = 0.2
    max_open_questions: int = 40
    max_new_questions_per_heartbeat: int = 3
    dedupe_similarity: float = 0.55
    lp_window: int = 4
    lp_prior: float = 0.6
    surprise_decay: float = 0.85
    refractory: float = 0.5
    boredom_lp_floor: float = 0.08
    boredom_rate: float = 0.35
    incubation_beliefs: int = 2
    homeostasis: bool = True
    export_experience: bool = True
    trace_llm: bool = False  # write every prompt and raw model reply to llm_trace.jsonl (for tuning prompts)
    max_tokens_text: int = 260
    max_tokens_json: int = 700
    temperament: TemperamentConfig = Field(default_factory=TemperamentConfig)
    body: BodyConfig = Field(default_factory=BodyConfig)


class ProjectConfig(BaseModel):
    name: str = "Curiosity Epistemic Loop"
    version: str = "7.0.0"
    description: str = "Aggressive-depth philosophical curiosity system with argument integrity gates, balance stress testing, hybrid web/academic grounding, and stricter revision control."


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    web_grounding: WebGroundingConfig = Field(default_factory=WebGroundingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    aggressive_mode: AggressiveModeConfig = Field(default_factory=AggressiveModeConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    revision: RevisionConfig = Field(default_factory=RevisionConfig)
    curiosity_state: CuriosityStateConfig = Field(default_factory=CuriosityStateConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)
    publishing: PublishingConfig = Field(default_factory=PublishingConfig)
    organism: OrganismConfig = Field(default_factory=OrganismConfig)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        return AppConfig()
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)
