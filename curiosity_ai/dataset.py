from __future__ import annotations

from pathlib import Path
from .config import AppConfig
from .schema import IterationRecord
from .utils import ensure_dir, write_jsonl


class DatasetExporter:
    """Exports iteration traces into JSONL examples for later SFT/QLoRA.

    This does not train the model. It creates a high-quality dataset from your own loop traces.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.path = Path(config.dataset.output_path)
        ensure_dir(self.path.parent)

    def maybe_export(self, record: IterationRecord) -> bool:
        if not self.config.dataset.enabled or not record.synthesis or not record.critique:
            return False
        curiosity_score = record.curiosity_score.total if record.curiosity_score else 0.0
        evaluation_score = record.evaluation.score if record.evaluation else curiosity_score
        depth_score = record.philosophical_depth.final_score if record.philosophical_depth else evaluation_score
        score = min(curiosity_score, evaluation_score, depth_score) if self.config.dataset.use_evaluation_score else curiosity_score
        if self.config.dataset.require_philosophical_depth_pass:
            if not record.philosophical_depth or record.philosophical_depth.verdict != "pass":
                return False
        if getattr(self.config.dataset, "require_real_move_pass", False):
            if not record.philosophical_move or record.philosophical_move.verdict != "pass":
                return False
        if getattr(self.config.dataset, "require_strong_position", False):
            if not record.strong_position or not record.strong_position.thesis.strip():
                return False
        if record.evaluation and not record.evaluation.should_publish:
            return False
        if not self.config.dataset.include_low_score_items and score < self.config.dataset.min_score_for_dataset:
            return False
        if record.web_results and not self.config.dataset.include_web_sources:
            # Avoid accidentally mixing web snippets into training data unless explicitly enabled.
            web_note = "Web sources were used but are excluded from the SFT example."
        else:
            web_note = ""

        local_context = "\n\n".join(
            f"{p.citation} {p.text[:1000]}" for p in record.retrieved_passages if p.source.source_type != "web"
        )
        instruction = (
            "Given a philosophical topic and retrieved context, generate rigorous curiosity-driven "
            "questions, critique weak questions, synthesize a source-grounded insight, and propose next directions."
        )
        user = f"Topic: {record.input_topic}\n\nContext:\n{local_context}\n\n{web_note}"
        assistant = {
            "strongest_question": record.critique.strongest_question,
            "accepted_questions": [q.model_dump() for q in record.critique.accepted_questions],
            "source_gate": record.source_gate.model_dump() if record.source_gate else None,
            "curiosity_score": record.curiosity_score.model_dump() if record.curiosity_score else None,
            "textual_analyses": [a.model_dump() for a in record.textual_analyses],
            "argument_reconstruction": record.argument_reconstruction.model_dump() if record.argument_reconstruction else None,
            "radical_candidates": [r.model_dump() for r in record.radical_candidates],
            "strong_position": record.strong_position.model_dump() if record.strong_position else None,
            "synthesis": record.synthesis.model_dump(),
            "argument_critique": record.argument_critique.model_dump() if record.argument_critique else None,
            "philosophical_depth": record.philosophical_depth.model_dump() if record.philosophical_depth else None,
            "philosophical_move": record.philosophical_move.model_dump() if record.philosophical_move else None,
            "curiosity_theory_state": record.curiosity_theory_state.model_dump() if record.curiosity_theory_state else None,
            "evaluation": record.evaluation.model_dump() if record.evaluation else None,
            "next_topic": record.next_topic,
        }
        example = {
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user},
                {"role": "assistant", "content": str(assistant)},
            ],
            "metadata": {
                "run_id": record.run_id,
                "iteration": record.iteration,
                "score": score,
                "curiosity_score": curiosity_score,
                "evaluation_score": evaluation_score,
                "philosophical_depth_score": depth_score,
                "philosophical_move_score": record.philosophical_move.final_quality_score if record.philosophical_move else None,
                "source_gate": record.source_gate.decision if record.source_gate else None,
                "created_at": record.created_at,
            },
        }
        write_jsonl(self.path, example)
        return True
