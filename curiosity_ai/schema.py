from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_str(value: Any) -> str:
    """Coerce imperfect local-LLM outputs into strings."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_list(value: Any) -> list[Any]:
    """Coerce common local-LLM schema mistakes into lists.

    Mistral/Ollama can sometimes return a string where the schema asked for
    ["string"]. This keeps the loop running instead of crashing.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        # If the dict looks like one object, wrap it; otherwise keep values.
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Split bullet-like multiline strings, but keep normal sentences intact.
        lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
        return [text]
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [_as_str(v) for v in _as_list(value) if _as_str(v).strip()]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("%", "")
        try:
            number = float(text)
            if "%" in value:
                number /= 100.0
            return number
        except ValueError:
            return default
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "publish", "continue"}
    return default


class Source(BaseModel):
    source_id: str
    source_type: Literal["local", "web", "memory", "synthetic"] = "local"
    title: str = "Unknown"
    author: str | None = None
    year: str | None = None
    file_path: str | None = None
    url: str | None = None
    license: str | None = None
    tradition: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Passage(BaseModel):
    passage_id: str
    text: str
    source: Source
    citation: str
    line_start: int | None = None
    line_end: int | None = None
    chunk_index: int | None = None
    score: float | None = None


class WebResult(BaseModel):
    web_id: str
    title: str
    url: str
    content: str
    citation: str
    score: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class QuestionCandidate(BaseModel):
    question: str
    assumptions: list[str] = Field(default_factory=list)
    why_interesting: str = ""
    philosophical_tension: str = ""
    likely_sources: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_question_object(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"question": data}
        return data

    @field_validator("question", "why_interesting", "philosophical_tension", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("assumptions", "likely_sources", "tags", mode="before")
    @classmethod
    def coerce_str_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)


class Critique(BaseModel):
    accepted_questions: list[QuestionCandidate] = Field(default_factory=list)
    rejected_questions: list[dict[str, str]] = Field(default_factory=list)
    main_objections: list[str] = Field(default_factory=list)
    strongest_question: str = ""
    refinement: str = ""

    @field_validator("accepted_questions", mode="before")
    @classmethod
    def coerce_accepted_questions(cls, value: Any) -> list[Any]:
        return _as_list(value)

    @field_validator("rejected_questions", mode="before")
    @classmethod
    def coerce_rejected_questions(cls, value: Any) -> list[dict[str, str]]:
        items = _as_list(value)
        out: list[dict[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                out.append({str(k): _as_str(v) for k, v in item.items()})
            else:
                out.append({"question": "", "reason": _as_str(item)})
        return out

    @field_validator("main_objections", mode="before")
    @classmethod
    def coerce_objections(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("strongest_question", "refinement", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)


class CuriosityScore(BaseModel):
    novelty: float = 0.0
    depth: float = 0.0
    epistemic_tension: float = 0.0
    researchability: float = 0.0
    generativity: float = 0.0
    citation_grounding: float = 0.0
    redundancy: float = 0.0
    total: float = 0.0
    explanation: str = ""

    @field_validator(
        "novelty",
        "depth",
        "epistemic_tension",
        "researchability",
        "generativity",
        "citation_grounding",
        "redundancy",
        "total",
        mode="before",
    )
    @classmethod
    def coerce_float_fields(cls, value: Any) -> float:
        return _as_float(value)

    @field_validator("explanation", mode="before")
    @classmethod
    def coerce_explanation(cls, value: Any) -> str:
        return _as_str(value)


class Synthesis(BaseModel):
    title: str
    central_question: str
    thesis: str
    argument: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    unresolved_tensions: list[str] = Field(default_factory=list)
    next_directions: list[str] = Field(default_factory=list)
    cited_passages: list[str] = Field(default_factory=list)
    cited_web: list[str] = Field(default_factory=list)
    report_markdown: str = ""

    @field_validator("title", "central_question", "thesis", "report_markdown", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator(
        "argument",
        "counterarguments",
        "unresolved_tensions",
        "next_directions",
        "cited_passages",
        "cited_web",
        mode="before",
    )
    @classmethod
    def coerce_str_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)


class Evaluation(BaseModel):
    score: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    citation_issues: list[str] = Field(default_factory=list)
    should_publish: bool = False
    should_continue: bool = True
    recommended_revision: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score(cls, value: Any) -> float:
        return _as_float(value)

    @field_validator("strengths", "weaknesses", "citation_issues", mode="before")
    @classmethod
    def coerce_str_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("should_publish", mode="before")
    @classmethod
    def coerce_should_publish(cls, value: Any) -> bool:
        return _as_bool(value, default=False)

    @field_validator("should_continue", mode="before")
    @classmethod
    def coerce_should_continue(cls, value: Any) -> bool:
        return _as_bool(value, default=True)

    @field_validator("recommended_revision", mode="before")
    @classmethod
    def coerce_revision(cls, value: Any) -> str:
        return _as_str(value)


class PlanningDecision(BaseModel):
    chosen_next_topic: str
    alternative_topics: list[str] = Field(default_factory=list)
    reason: str = ""
    exploit_or_explore: Literal["exploit", "explore", "bridge", "stop"] = "bridge"
    expected_value: float = 0.0

    @field_validator("chosen_next_topic", "reason", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("alternative_topics", mode="before")
    @classmethod
    def coerce_alternatives(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("expected_value", mode="before")
    @classmethod
    def coerce_expected_value(cls, value: Any) -> float:
        return _as_float(value)

    @field_validator("exploit_or_explore", mode="before")
    @classmethod
    def coerce_mode(cls, value: Any) -> str:
        text = _as_str(value).strip().lower()
        if text not in {"exploit", "explore", "bridge", "stop"}:
            return "bridge"
        return text


class SourceGateReport(BaseModel):
    primary_source_count: int = 0
    memory_source_count: int = 0
    web_source_count: int = 0
    required_primary_sources: int = 2
    decision: Literal["pass", "limited", "block"] = "limited"
    reason: str = ""
    missing_sources: list[str] = Field(default_factory=list)
    usable_citations: list[str] = Field(default_factory=list)

    @field_validator("reason", mode="before")
    @classmethod
    def coerce_reason(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("missing_sources", "usable_citations", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)


class TextualAnalysis(BaseModel):
    source_label: str = ""
    source_role: str = ""
    literal_claim: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    interpretive_analysis: str = ""
    relevance_to_question: str = ""
    anachronism_risk: str = ""
    limitations: str = ""

    @field_validator(
        "source_label", "source_role", "literal_claim", "interpretive_analysis",
        "relevance_to_question", "anachronism_risk", "limitations", mode="before"
    )
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("key_concepts", mode="before")
    @classmethod
    def coerce_concepts(cls, value: Any) -> list[str]:
        return _as_str_list(value)


class ArgumentReconstruction(BaseModel):
    central_claim: str = ""
    premises: list[str] = Field(default_factory=list)
    conclusion: str = ""
    hidden_assumptions: list[str] = Field(default_factory=list)
    inferential_gaps: list[str] = Field(default_factory=list)
    strongest_objection: str = ""
    possible_reply: str = ""

    @field_validator("central_claim", "conclusion", "strongest_objection", "possible_reply", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("premises", "hidden_assumptions", "inferential_gaps", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @model_validator(mode="after")
    def require_philosophical_slots(self):
        # v7: local LLMs sometimes return empty conclusion / hidden assumptions / gaps.
        # These fields are philosophically mandatory, so create conservative placeholders
        # rather than silently letting the report display blanks.
        if not self.central_claim.strip():
            self.central_claim = "A substantive philosophical claim must be reconstructed from the selected question and sources."
        if not self.premises:
            self.premises = ["P1: The available source analyses must support at least one explicit premise; otherwise synthesis should be blocked or revised."]
        if not self.conclusion.strip():
            self.conclusion = self.central_claim
        if not self.hidden_assumptions:
            self.hidden_assumptions = [
                "The selected source passages can be legitimately connected to the central curiosity question.",
                "The interpretive bridge from textual evidence to philosophical thesis is acceptable but must be defended."
            ]
        if not self.inferential_gaps:
            self.inferential_gaps = [
                "The model did not explicitly identify an inferential gap; verify whether the conclusion follows from the premises rather than from thematic similarity."
            ]
        if not self.strongest_objection.strip():
            self.strongest_objection = "The reconstructed argument may rely on a thematic connection rather than a demonstrated logical or textual connection."
        if not self.possible_reply.strip():
            self.possible_reply = "Defend the interpretive bridge using a closer reading of the cited passages, or narrow the claim."
        return self


class BalanceStressTest(BaseModel):
    balance_claim_detected: bool = False
    what_is_balanced: str = ""
    who_or_what_decides_balance: str = ""
    criterion_for_virtue: str = ""
    criterion_for_danger: str = ""
    failure_modes: list[str] = Field(default_factory=list)
    hard_question_for_synthesis: str = ""
    verdict: Literal["pass", "revise", "block"] = "revise"

    @field_validator("balance_claim_detected", mode="before")
    @classmethod
    def coerce_bool_field(cls, value: Any) -> bool:
        return _as_bool(value)

    @field_validator("what_is_balanced", "who_or_what_decides_balance", "criterion_for_virtue", "criterion_for_danger", "hard_question_for_synthesis", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("failure_modes", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("verdict", mode="before")
    @classmethod
    def coerce_verdict(cls, value: Any) -> str:
        text = _as_str(value).strip().lower()
        if text not in {"pass", "revise", "block"}:
            return "revise"
        return text

    @model_validator(mode="after")
    def require_balance_clarity(self):
        if self.balance_claim_detected:
            if not self.what_is_balanced.strip():
                self.what_is_balanced = "The synthesis uses balance/tempering language but does not define what elements are being balanced."
            if not self.who_or_what_decides_balance.strip():
                self.who_or_what_decides_balance = "The synthesis must specify the criterion, agent, method, or practice that decides the balance."
            if not self.criterion_for_virtue.strip():
                self.criterion_for_virtue = "Curiosity becomes virtuous only if it improves inquiry without collapsing into skeptical paralysis or dogmatic closure."
            if not self.criterion_for_danger.strip():
                self.criterion_for_danger = "Curiosity becomes dangerous when it generates endless questioning without criteria for evidence, action, or closure."
            if not self.failure_modes:
                self.failure_modes = ["skeptical paralysis", "dogmatic closure disguised as balance", "educational compliance instead of inquiry"]
            if not self.hard_question_for_synthesis.strip():
                self.hard_question_for_synthesis = "Balance exactly means what, who decides it, and what fails when balance fails?"
        return self


class ArgumentCritique(BaseModel):
    verdict: Literal["accept", "revise", "block"] = "revise"
    major_objections: list[str] = Field(default_factory=list)
    citation_problems: list[str] = Field(default_factory=list)
    logical_gaps: list[str] = Field(default_factory=list)
    anachronism_warnings: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)

    @field_validator("major_objections", "citation_problems", "logical_gaps", "anachronism_warnings", "revision_instructions", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("verdict", mode="before")
    @classmethod
    def coerce_verdict(cls, value: Any) -> str:
        text = _as_str(value).strip().lower()
        if text not in {"accept", "revise", "block"}:
            return "revise"
        return text


class PhilosophicalDepthJudgment(BaseModel):
    """Post-synthesis philosophical quality judgment.

    This is intentionally separate from the CuriosityScore. CuriosityScore asks
    whether a question is promising; this judge asks whether the produced
    philosophical discussion is actually deep, careful, and on-theme.
    """
    concept_definition: float = 0.0
    anachronism_control: float = 0.0
    genuine_tension: float = 0.0
    counterargument_strength: float = 0.0
    source_interpretation: float = 0.0
    curiosity_theme_progress: float = 0.0
    final_score: float = 0.0
    verdict: Literal["pass", "revise", "block"] = "revise"
    main_findings: list[str] = Field(default_factory=list)
    depth_failures: list[str] = Field(default_factory=list)
    required_revisions: list[str] = Field(default_factory=list)
    curiosity_progress_note: str = ""

    @field_validator(
        "concept_definition",
        "anachronism_control",
        "genuine_tension",
        "counterargument_strength",
        "source_interpretation",
        "curiosity_theme_progress",
        "final_score",
        mode="before",
    )
    @classmethod
    def coerce_float_fields(cls, value: Any) -> float:
        return _as_float(value)

    @field_validator("main_findings", "depth_failures", "required_revisions", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("curiosity_progress_note", mode="before")
    @classmethod
    def coerce_note(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("verdict", mode="before")
    @classmethod
    def coerce_verdict(cls, value: Any) -> str:
        text = _as_str(value).strip().lower()
        if text not in {"pass", "revise", "block"}:
            return "revise"
        return text


class RadicalCandidate(BaseModel):
    radical_question: str = ""
    radical_thesis: str = ""
    why_dangerous: str = ""
    required_sources: list[str] = Field(default_factory=list)
    possible_objection: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("radical_question", "radical_thesis", "why_dangerous", "possible_objection", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("required_sources", "tags", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)


class StrongPosition(BaseModel):
    position_type: Literal["virtue", "method", "danger", "conditional", "revisionary"] = "conditional"
    thesis: str = ""
    decisive_condition: str = ""
    defended_claim: str = ""
    concession: str = ""
    source_labels: list[str] = Field(default_factory=list)
    forbidden_hedges_to_remove: list[str] = Field(default_factory=list)
    reasoning_strategy: str = ""

    @field_validator("thesis", "decisive_condition", "defended_claim", "concession", "reasoning_strategy", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("source_labels", "forbidden_hedges_to_remove", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("position_type", mode="before")
    @classmethod
    def coerce_position_type(cls, value: Any) -> str:
        text = _as_str(value).strip().lower()
        if text not in {"virtue", "method", "danger", "conditional", "revisionary"}:
            return "conditional"
        return text


class PhilosophicalMove(BaseModel):
    move_type: Literal["distinction", "paradox", "contradiction", "dilemma", "reversal", "genealogy", "hidden_assumption", "objection_reply", "criterion", "none"] = "none"
    move_statement: str = ""
    concept_definition: str = ""
    paradox_or_tension: str = ""
    source_support: list[str] = Field(default_factory=list)
    why_it_advances_curiosity: str = ""
    hedge_count: int = 0
    hedge_penalty: float = 0.0
    unsupported_inference_penalty: float = 0.0
    final_quality_score: float = 0.0
    verdict: Literal["pass", "revise", "block"] = "revise"
    required_revision: str = ""

    @field_validator("move_statement", "concept_definition", "paradox_or_tension", "why_it_advances_curiosity", "required_revision", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("source_support", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("hedge_penalty", "unsupported_inference_penalty", "final_quality_score", mode="before")
    @classmethod
    def coerce_float_fields(cls, value: Any) -> float:
        return _as_float(value)

    @field_validator("hedge_count", mode="before")
    @classmethod
    def coerce_int_fields(cls, value: Any) -> int:
        return int(_as_float(value))

    @field_validator("move_type", mode="before")
    @classmethod
    def coerce_move_type(cls, value: Any) -> str:
        text = _as_str(value).strip().lower()
        allowed = {"distinction", "paradox", "contradiction", "dilemma", "reversal", "genealogy", "hidden_assumption", "objection_reply", "criterion", "none"}
        return text if text in allowed else "none"

    @field_validator("verdict", mode="before")
    @classmethod
    def coerce_verdict(cls, value: Any) -> str:
        text = _as_str(value).strip().lower()
        if text not in {"pass", "revise", "block"}:
            return "revise"
        return text


class CuriosityTheoryState(BaseModel):
    current_definition: str = "Curiosity is an unresolved object of inquiry."
    properties_discovered: list[str] = Field(default_factory=list)
    unresolved_tensions: list[str] = Field(default_factory=list)
    latest_update: str = ""
    new_property_this_iteration: str = ""
    next_required_move: str = ""
    confidence: float = 0.0
    history: list[str] = Field(default_factory=list)

    @field_validator("current_definition", "latest_update", "new_property_this_iteration", "next_required_move", mode="before")
    @classmethod
    def coerce_text_fields(cls, value: Any) -> str:
        return _as_str(value)

    @field_validator("properties_discovered", "unresolved_tensions", "history", mode="before")
    @classmethod
    def coerce_lists(cls, value: Any) -> list[str]:
        return _as_str_list(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_float_fields(cls, value: Any) -> float:
        return _as_float(value)


class IterationRecord(BaseModel):
    run_id: str
    iteration: int
    created_at: str = Field(default_factory=utc_now_iso)
    input_topic: str
    retrieved_passages: list[Passage] = Field(default_factory=list)
    web_results: list[WebResult] = Field(default_factory=list)
    source_gate: SourceGateReport | None = None
    questions: list[QuestionCandidate] = Field(default_factory=list)
    radical_candidates: list[RadicalCandidate] = Field(default_factory=list)
    critique: Critique | None = None
    curiosity_score: CuriosityScore | None = None
    textual_analyses: list[TextualAnalysis] = Field(default_factory=list)
    argument_reconstruction: ArgumentReconstruction | None = None
    strong_position: StrongPosition | None = None
    balance_stress_test: BalanceStressTest | None = None
    synthesis: Synthesis | None = None
    argument_critique: ArgumentCritique | None = None
    philosophical_depth: PhilosophicalDepthJudgment | None = None
    philosophical_move: PhilosophicalMove | None = None
    curiosity_theory_state: CuriosityTheoryState | None = None
    evaluation: Evaluation | None = None
    planning: PlanningDecision | None = None
    next_topic: str | None = None
    raw_agent_outputs: dict[str, Any] = Field(default_factory=dict)


class PublicationItem(BaseModel):
    publication_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    run_id: str
    iteration: int
    title: str
    body_markdown: str
    score: float
    approved: bool = False
    platform: str = "local_markdown"
    metadata: dict[str, Any] = Field(default_factory=dict)
