from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4
from typing import Any

from .config import AppConfig
from .llm import OllamaClient
from .memory import ChromaMemory
from .grounding import (
    LocalGroundingAgent,
    WebGroundingAgent,
    format_passages,
    format_web_results,
    build_source_gate_report,
)
from .agents import (
    ReaderAgent,
    ExplorerAgent,
    RadicalQuestionAgent,
    CriticAgent,
    ResearchAgent,
    TextualAnalystAgent,
    ArgumentReconstructorAgent,
    PositionEnforcerAgent,
    BalanceStressTesterAgent,
    SynthesizerAgent,
    ArgumentCriticAgent,
    PhilosophicalDepthJudgeAgent,
    PhilosophicalMoveDetectorAgent,
    CuriosityTheoryTrackerAgent,
    EvaluatorAgent,
    PlanningAgent,
    PublisherAgent,
    compact_recent_memory,
)
from .scoring import CuriosityScorer
from .schema import (
    IterationRecord,
    QuestionCandidate,
    Synthesis,
    ArgumentReconstruction,
    CuriosityTheoryState,
    BalanceStressTest,
    ArgumentCritique,
    PhilosophicalDepthJudgment,
    PhilosophicalMove,
)
from .reporter import ReportWriter
from .dataset import DatasetExporter
from .publishing import PublishingManager
from .utils import ensure_dir


class LoopController:
    def __init__(self, config: AppConfig):
        self.config = config
        self.run_id = f"run_{int(time.time())}_{uuid4().hex[:8]}"
        self.llm = OllamaClient(config.llm)
        self.memory = ChromaMemory(config)
        self.local_grounder = LocalGroundingAgent(config, self.memory)
        self.web_grounder = WebGroundingAgent(config)
        self.reader = ReaderAgent(self.llm)
        self.explorer = ExplorerAgent(config, self.llm)
        self.radicalizer = RadicalQuestionAgent(config, self.llm)
        self.critic = CriticAgent(self.llm)
        self.research = ResearchAgent(self.llm)
        self.textual_analyst = TextualAnalystAgent(config, self.llm)
        self.argument_reconstructor = ArgumentReconstructorAgent(self.llm)
        self.position_enforcer = PositionEnforcerAgent(config, self.llm)
        self.balance_stress_tester = BalanceStressTesterAgent(config, self.llm)
        self.synthesizer = SynthesizerAgent(config, self.llm)
        self.argument_critic = ArgumentCriticAgent(self.llm)
        self.philosophical_depth_judge = PhilosophicalDepthJudgeAgent(self.llm)
        self.philosophical_move_detector = PhilosophicalMoveDetectorAgent(config, self.llm)
        self.curiosity_theory_tracker = CuriosityTheoryTrackerAgent(self.llm)
        self.evaluator = EvaluatorAgent(self.llm)
        self.planner = PlanningAgent(config, self.llm)
        self.publisher_agent = PublisherAgent(self.llm)
        self.scorer = CuriosityScorer(config)
        self.reporter = ReportWriter(config, self.run_id)
        self.dataset = DatasetExporter(config)
        self.publisher = PublishingManager(config, self.publisher_agent)
        self.theory_state_path = Path(config.curiosity_state.path)
        ensure_dir(self.theory_state_path.parent)

    def bootstrap(self) -> None:
        if self.config.corpus.auto_ingest_on_start:
            count = self.memory.ingest_corpus()
            print(f"[bootstrap] Ingested/checked {count} corpus chunks.")

    def run(self, initial_topic: str, iterations: int | None = None) -> list[IterationRecord]:
        self.bootstrap()
        max_iter = iterations or self.config.loop.max_iterations
        topic = initial_topic
        records: list[IterationRecord] = []
        no_progress = 0

        for i in range(1, max_iter + 1):
            print(f"\n[loop] Iteration {i}/{max_iter}: {topic}")
            record = self.run_iteration(topic, i)
            records.append(record)
            self.memory.store_iteration(record)
            if record.curiosity_theory_state:
                self._save_theory_state(record.curiosity_theory_state)
            self.reporter.write_iteration(record)
            exported = self.dataset.maybe_export(record)
            if exported:
                print(f"[dataset] Exported high-quality SFT example for iteration {i}.")
            queued = self.publisher.maybe_queue(record)
            if queued:
                print(f"[publish] Queued candidate for human approval: {queued}")

            # v6 progress depends on the weakest major quality signal, not novelty alone.
            curiosity_score = record.curiosity_score.total if record.curiosity_score else 0.0
            depth_score = record.philosophical_depth.final_score if record.philosophical_depth else 0.0
            move_score = record.philosophical_move.final_quality_score if record.philosophical_move else 0.0
            eval_score = record.evaluation.score if record.evaluation else 0.0
            progress_score = min(curiosity_score, depth_score or curiosity_score, move_score or curiosity_score, eval_score or curiosity_score)
            redundancy = record.curiosity_score.redundancy if record.curiosity_score else 1.0

            if progress_score < self.config.loop.min_curiosity_score or redundancy > self.config.loop.stop_if_redundancy_above:
                no_progress += 1
            else:
                no_progress = 0

            if record.evaluation and not record.evaluation.should_continue:
                print("[loop] Evaluator recommended stopping.")
                break
            if record.planning and record.planning.exploit_or_explore == "stop":
                print("[loop] Planning agent recommended stopping.")
                break
            if no_progress >= self.config.loop.stop_if_no_progress_for:
                print("[loop] Stopping because progress score stayed too low or redundancy too high.")
                break
            if not record.next_topic:
                print("[loop] No next topic produced; stopping.")
                break
            topic = record.next_topic

        self.reporter.write_final(records)
        return records

    def run_iteration(self, topic: str, iteration: int) -> IterationRecord:
        recent_records = self.memory.recent_records(limit=self.config.loop.short_term_memory_items)
        recent_memory = compact_recent_memory(recent_records)
        theory_state = self._load_theory_state()

        passages = self.local_grounder.retrieve(topic)
        web_results = self.web_grounder.search(topic)
        source_gate = build_source_gate_report(self.config, passages, web_results)
        print(f"[source-gate] {source_gate.decision}: {source_gate.reason}")

        passages_text = format_passages(passages)
        web_text = format_web_results(web_results)

        raw: dict[str, Any] = {}
        raw["source_gate"] = source_gate.model_dump()
        raw["curiosity_theory_state_before"] = theory_state.model_dump() if theory_state else None

        grounding_notes = self.reader.distill(topic, passages_text, web_text, source_gate)
        raw["grounding_notes"] = grounding_notes

        questions = self.explorer.generate_questions(topic, grounding_notes, recent_memory, source_gate, theory_state)
        radical_candidates = []
        if self.config.aggressive_mode.enabled:
            radical_candidates = self.radicalizer.generate_radicals(topic, grounding_notes, questions, source_gate, theory_state)
            questions = questions + self.radicalizer.as_questions(radical_candidates)
        raw["questions"] = [q.model_dump() for q in questions]
        raw["radical_candidates"] = [r.model_dump() for r in radical_candidates]

        critique = self.critic.critique(topic, questions, grounding_notes, source_gate)
        raw["critique"] = critique.model_dump()

        strongest = self._select_strongest_question(critique, questions)
        score = self.scorer.score(strongest, [str(r) for r in recent_records], passages, web_results)
        print(f"[score] {score.total:.3f} — {score.explanation}")

        research_brief = self.research.connect(strongest.question, passages_text, web_text, recent_memory, source_gate)
        raw["research_brief"] = research_brief

        primary_passages = [p for p in passages if p.source.source_type == "local"]
        passages_for_analysis = primary_passages if primary_passages else passages
        analysis_text = format_passages(passages_for_analysis, max_chars_per_passage=2200)
        textual_analyses = self.textual_analyst.analyze(strongest.question, analysis_text, source_gate)
        raw["textual_analyses"] = [a.model_dump() for a in textual_analyses]

        if textual_analyses:
            reconstruction = self.argument_reconstructor.reconstruct(strongest.question, research_brief, textual_analyses)
            reconstruction = self._repair_argument_reconstruction(reconstruction, strongest.question, research_brief, textual_analyses)
        else:
            reconstruction = ArgumentReconstruction(
                central_claim="Insufficient textual analysis was generated.",
                premises=[],
                conclusion="The system should retrieve and analyze primary passages before making a substantive claim.",
                inferential_gaps=["No textual analyses available."],
                strongest_objection="The answer would be memory-only or speculative.",
                possible_reply="Narrow the next topic toward a primary source passage.",
            )
        raw["argument_reconstruction"] = reconstruction.model_dump()

        strong_position = None
        if self.config.aggressive_mode.require_strong_position:
            strong_position = self.position_enforcer.enforce(
                topic, strongest.question, research_brief, textual_analyses, reconstruction, radical_candidates
            )
        raw["strong_position"] = strong_position.model_dump() if strong_position else None

        balance_stress_test = None
        if self.config.balance.enabled:
            balance_stress_test = self.balance_stress_tester.stress_test(
                strongest.question, strong_position, reconstruction, textual_analyses
            )
        raw["balance_stress_test"] = balance_stress_test.model_dump() if balance_stress_test else None

        synthesis = self.synthesizer.synthesize(
            topic,
            strongest.question,
            grounding_notes,
            research_brief,
            textual_analyses,
            reconstruction,
            score,
            source_gate,
            strong_position,
            theory_state,
            balance_stress_test,
        )
        raw["synthesis_draft"] = synthesis.model_dump()

        argument_critique: ArgumentCritique | None = None
        philosophical_depth: PhilosophicalDepthJudgment | None = None
        philosophical_move: PhilosophicalMove | None = None

        max_rounds = max(1, self.config.revision.max_revision_rounds if self.config.revision.enabled else 1)
        for round_i in range(max_rounds):
            argument_critique = self.argument_critic.critique(synthesis, source_gate, textual_analyses, reconstruction, strong_position)
            raw[f"argument_critique_round_{round_i+1}"] = argument_critique.model_dump()

            if argument_critique.verdict == "block":
                synthesis = self._blocked_synthesis(topic, strongest.question, source_gate, argument_critique)
                raw[f"synthesis_blocked_round_{round_i+1}"] = synthesis.model_dump()
                break
            if argument_critique.verdict == "revise":
                synthesis = self.synthesizer.revise(synthesis, argument_critique, source_gate, strong_position, balance_stress=balance_stress_test)
                raw[f"synthesis_revised_after_argument_critique_round_{round_i+1}"] = synthesis.model_dump()

            philosophical_depth = self.philosophical_depth_judge.judge(
                topic,
                strongest.question,
                synthesis,
                source_gate,
                textual_analyses,
                reconstruction,
                argument_critique,
                strong_position,
            )
            raw[f"philosophical_depth_round_{round_i+1}"] = philosophical_depth.model_dump()

            philosophical_move = self.philosophical_move_detector.detect(
                topic, strongest.question, synthesis, strong_position, philosophical_depth
            )
            raw[f"philosophical_move_round_{round_i+1}"] = philosophical_move.model_dump()

            print(
                f"[depth:r{round_i+1}] {philosophical_depth.final_score:.3f} — "
                f"concept={philosophical_depth.concept_definition:.2f}, "
                f"tension={philosophical_depth.genuine_tension:.2f}, "
                f"counter={philosophical_depth.counterargument_strength:.2f}, "
                f"source={philosophical_depth.source_interpretation:.2f}, "
                f"curiosity={philosophical_depth.curiosity_theme_progress:.2f}; "
                f"move={philosophical_move.final_quality_score:.3f}/{philosophical_move.verdict}"
            )

            passes_depth = (
                philosophical_depth.final_score >= self.config.depth.min_philosophical_depth_score
                and philosophical_depth.curiosity_theme_progress >= self.config.depth.min_curiosity_progress_score
                and philosophical_depth.verdict == "pass"
            )
            passes_move = (
                philosophical_move.final_quality_score >= self.config.aggressive_mode.min_real_move_score
                and philosophical_move.verdict == "pass"
            )
            if passes_depth and passes_move and argument_critique.verdict == "accept":
                break

            # If there is another round, force a deeper revision before moving on.
            if round_i < max_rounds - 1:
                synthesis = self.synthesizer.revise(
                    synthesis,
                    argument_critique,
                    source_gate,
                    strong_position,
                    philosophical_depth,
                    philosophical_move,
                    balance_stress_test,
                )
                raw[f"synthesis_revised_after_depth_move_round_{round_i+1}"] = synthesis.model_dump()

        # Ensure we always have final judgments.
        if argument_critique is None:
            argument_critique = self.argument_critic.critique(synthesis, source_gate, textual_analyses, reconstruction, strong_position)
        if philosophical_depth is None:
            philosophical_depth = self.philosophical_depth_judge.judge(topic, strongest.question, synthesis, source_gate, textual_analyses, reconstruction, argument_critique, strong_position)
        if philosophical_move is None:
            philosophical_move = self.philosophical_move_detector.detect(topic, strongest.question, synthesis, strong_position, philosophical_depth)

        raw["argument_critique"] = argument_critique.model_dump()
        raw["philosophical_depth"] = philosophical_depth.model_dump()
        raw["philosophical_move"] = philosophical_move.model_dump()

        updated_theory_state = theory_state
        if self.config.curiosity_state.enabled:
            updated_theory_state = self.curiosity_theory_tracker.update(topic, synthesis, theory_state, philosophical_move, philosophical_depth)
            raw["curiosity_theory_state_after"] = updated_theory_state.model_dump()

        evaluation = self.evaluator.evaluate(
            synthesis,
            score,
            strict_citations=self.config.web_grounding.strict_citations,
            source_gate=source_gate,
            argument_critique=argument_critique,
            philosophical_depth=philosophical_depth,
            philosophical_move=philosophical_move,
            position=strong_position,
        )
        evaluation.score = self._recalibrate_evaluation_score(evaluation.score, score, philosophical_depth, philosophical_move, argument_critique)
        if philosophical_depth.verdict in {"revise", "block"} or philosophical_move.verdict in {"revise", "block"}:
            evaluation.should_publish = False
        if philosophical_depth.final_score < self.config.depth.min_philosophical_depth_score:
            evaluation.should_publish = False
        if philosophical_move.final_quality_score < self.config.aggressive_mode.min_real_move_score:
            evaluation.should_publish = False
        if argument_critique.verdict != "accept":
            evaluation.should_publish = False
        if balance_stress_test and balance_stress_test.balance_claim_detected and balance_stress_test.verdict != "pass":
            evaluation.should_publish = False

        planning = self.planner.plan_next(
            topic, synthesis, critique, score, recent_memory, source_gate,
            argument_critique, philosophical_depth, philosophical_move, updated_theory_state
        )

        next_topic = planning.chosen_next_topic if planning.exploit_or_explore != "stop" else None
        if self._must_force_revision_topic(philosophical_depth, philosophical_move, evaluation, argument_critique, balance_stress_test):
            next_topic = f"{self.config.revision.same_topic_prefix}{strongest.question}"
            planning.chosen_next_topic = next_topic
            planning.exploit_or_explore = "exploit"
            planning.reason = (
                "Forced by v6 revision gate: philosophical depth, curiosity progress, or real move quality "
                "did not meet the threshold, so the next iteration must deepen the same question rather than drift. "
                + planning.reason
            )

        record = IterationRecord(
            run_id=self.run_id,
            iteration=iteration,
            input_topic=topic,
            retrieved_passages=passages,
            web_results=web_results,
            source_gate=source_gate,
            questions=questions,
            radical_candidates=radical_candidates,
            critique=critique,
            curiosity_score=score,
            textual_analyses=textual_analyses,
            argument_reconstruction=reconstruction,
            strong_position=strong_position,
            balance_stress_test=balance_stress_test,
            synthesis=synthesis,
            argument_critique=argument_critique,
            philosophical_depth=philosophical_depth,
            philosophical_move=philosophical_move,
            curiosity_theory_state=updated_theory_state,
            evaluation=evaluation,
            planning=planning,
            next_topic=next_topic,
            raw_agent_outputs=raw,
        )
        return record

    def _repair_argument_reconstruction(self, reconstruction: ArgumentReconstruction, question: str, research_brief: str, analyses) -> ArgumentReconstruction:
        # v7 deterministic safety net: the argument slots are mandatory for philosophy.
        # Pydantic fills many blanks, but this method also adds source-aware defaults.
        if not reconstruction.central_claim.strip():
            reconstruction.central_claim = f"The selected question requires a defensible claim about curiosity: {question}"
        if not reconstruction.conclusion.strip():
            reconstruction.conclusion = reconstruction.central_claim
        if not reconstruction.premises:
            reconstruction.premises = [
                "P1: At least one cited passage must support the interpretation; otherwise the synthesis must mark the claim as provisional."
            ]
        if not reconstruction.hidden_assumptions:
            reconstruction.hidden_assumptions = [
                "The retrieved passages are relevant to the selected question rather than merely thematically adjacent.",
                "The synthesis can move from source interpretation to a general claim about curiosity without overgeneralizing."
            ]
        if not reconstruction.inferential_gaps:
            reconstruction.inferential_gaps = [
                "The report must show how the cited passages entail, or at least strongly motivate, the conclusion."
            ]
        if not reconstruction.strongest_objection.strip():
            reconstruction.strongest_objection = "The argument may be converting a cautious textual point into a stronger thesis about curiosity than the sources justify."
        if not reconstruction.possible_reply.strip():
            reconstruction.possible_reply = "Limit the thesis to an argued interpretation and support it with direct passage analysis."
        return reconstruction

    def _recalibrate_evaluation_score(self, eval_score, curiosity_score, depth, move, critique) -> float:
        base = float(eval_score or 0.0)
        q = curiosity_score.total if curiosity_score else base
        d = depth.final_score if depth else base
        m = move.final_quality_score if move else base
        # Argument critique accept/revise/block influences score.
        crit_factor = {"accept": 1.0, "revise": 0.82, "block": 0.45}.get(critique.verdict if critique else "revise", 0.82)
        recalibrated = min(base, 0.18 * q + 0.34 * d + 0.28 * m + 0.20 * base) * crit_factor
        return max(0.0, min(1.0, recalibrated))

    def _must_force_revision_topic(self, depth, move, evaluation, argument_critique=None, balance_stress_test=None) -> bool:
        if not self.config.revision.enabled or not self.config.revision.force_same_topic_on_failure:
            return False
        if depth.final_score < self.config.revision.revise_if_depth_below:
            return True
        if depth.curiosity_theme_progress < self.config.revision.revise_if_curiosity_progress_below:
            return True
        if move.final_quality_score < self.config.revision.revise_if_move_below:
            return True
        if evaluation.score < self.config.dataset.min_score_for_dataset:
            return True
        if argument_critique is not None and argument_critique.verdict != "accept":
            return True
        if balance_stress_test is not None and balance_stress_test.balance_claim_detected and balance_stress_test.verdict != "pass":
            return True
        return False

    def _load_theory_state(self) -> CuriosityTheoryState | None:
        if not self.config.curiosity_state.enabled or not self.theory_state_path.exists():
            return None
        try:
            return CuriosityTheoryState.model_validate(json.loads(self.theory_state_path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def _save_theory_state(self, state: CuriosityTheoryState) -> None:
        if not self.config.curiosity_state.enabled:
            return
        self.theory_state_path.write_text(json.dumps(state.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _blocked_synthesis(self, topic: str, question: str, source_gate, argument_critique) -> Synthesis:
        body = "\n".join([
            "# Synthesis Blocked by Argument Critic",
            "",
            f"**Topic:** {topic}",
            f"**Question:** {question}",
            "",
            "The system blocked substantive synthesis because the argument critic judged the available grounding or reasoning insufficient.",
            "",
            "## Source Gate",
            source_gate.reason,
            "",
            "## Critique",
            *[f"- {x}" for x in argument_critique.major_objections],
            *[f"- Citation problem: {x}" for x in argument_critique.citation_problems],
            *[f"- Logical gap: {x}" for x in argument_critique.logical_gaps],
            "",
            "## Next Step",
            "Narrow the topic and retrieve primary passages before continuing.",
        ])
        return Synthesis(
            title="Synthesis Blocked: Insufficient Grounding",
            central_question=question,
            thesis="The system should not make a substantive philosophical claim without adequate primary-source grounding.",
            argument=[],
            counterarguments=[],
            unresolved_tensions=argument_critique.major_objections,
            next_directions=["Retrieve primary passages", "Narrow the next topic to a specific source or passage"],
            report_markdown=body,
        )

    def _select_strongest_question(self, critique, original_questions: list[QuestionCandidate]) -> QuestionCandidate:
        if critique.accepted_questions:
            if critique.strongest_question:
                for q in critique.accepted_questions:
                    if q.question.strip().lower() == critique.strongest_question.strip().lower():
                        return q
            return critique.accepted_questions[0]
        if original_questions:
            return original_questions[0]
        return QuestionCandidate(
            question="What hidden assumption is preventing this topic from becoming a deeper inquiry?",
            assumptions=["The current topic may be too broad or poorly grounded."],
            why_interesting="This fallback question protects the loop from empty generations.",
            philosophical_tension="The system needs a tension to continue, but none was generated.",
            tags=["fallback"],
        )
