"""Autonomous experiment subsystem using direct, domain-specific programs.

There is no universal experiment blueprint.  The subsystem chooses a question and protocol, emits a
small self-contained program, and executes that exact program twice.  The resulting record says only
what execution observed; it does not self-certify methodology.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field

from ...agents.experiment import ExperimentAgent
from ...artifact_store import ArtifactStore
from ...experimenter.validation import validate_experiment_program
from ...llm import LLMRouter
from ...schemas import ArtifactRef, ExperimentProgram, StrictModel
from ..models import (
    ActionOutcome,
    ActionProposal,
    EvidenceReceipt,
    EvidenceType,
    RecordDraft,
    RecordKind,
    RecordRelation,
    ResearchView,
    content_fingerprint,
)


class ExperimentChoice(StrictModel):
    action: Literal["run", "idle"]
    question: str = Field(default="", max_length=1200)
    hypothesis: str = Field(default="", max_length=1200)
    protocol: str = Field(default="", max_length=5000)
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=30)
    rationale: str = Field(default="", max_length=2000)
    parent_ids: list[str] = Field(default_factory=list, max_length=12)


class ExperimentSubsystem:
    name = "experiment"
    description = "Chooses and exactly replicates bounded domain-specific computational studies."
    model_call_budget = 3

    def __init__(self, store: ArtifactStore, router: LLMRouter):
        self.store = store
        self.router = router

    def propose(self, view: ResearchView) -> ActionProposal | None:
        if self.router.dry_run or self.router.experimenter is None:
            return None
        choice = self.router.complete_structured(
            task_type="experiment_design",
            schema=ExperimentChoice,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the autonomous experiment subsystem. Select one small computation "
                        "that discriminates a concrete uncertainty in the task or shared records. "
                        "Design domain-specific measurements; never borrow metric names from an "
                        "unrelated field. Include fixed seeds, explicit units, a reference or invariant "
                        "when possible, all cost accounting relevant to the claim, and a null outcome. "
                        "Choose idle rather than repeat an executed design."
                    ),
                },
                {"role": "user", "content": view.model_dump_json()},
            ],
        )
        if choice.action == "idle" or not choice.question.strip():
            return None
        return ActionProposal(
            subsystem=self.name,
            action_type="run",
            title=choice.question[:300],
            rationale=choice.rationale,
            payload={
                "question": choice.question,
                "hypothesis": choice.hypothesis,
                "protocol": choice.protocol,
                "seeds": choice.seeds,
            },
            parent_ids=[
                record_id
                for record_id in choice.parent_ids
                if record_id in {record.record_id for record in view.records}
            ],
        )

    def execute(
        self, proposal: ActionProposal, view: ResearchView, *, run_dir: str
    ) -> ActionOutcome:
        program = self.router.complete_structured(
            task_type="experiment_implementation",
            schema=ExperimentProgram,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Implement the supplied protocol as one self-contained Python module. Define "
                        "run_experiment(mode: str) -> dict. Return ExperimentOutput schema_version 1 "
                        "with protocol, raw observations [{unit_id, condition, values}], summaries, "
                        "interpretation, and limitations. Do not emit pass/fail checks or an expected-"
                        "direction assertion. Compute every value from executed code. Use only the "
                        "Python standard library, fixed seeds, and bounded loops."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(proposal.payload, ensure_ascii=False),
                },
            ],
        )
        expected_seeds = list(proposal.payload.get("seeds") or [])
        if expected_seeds and program.seeds != expected_seeds:
            raise ValueError("implementation seeds do not match the persisted protocol")
        validate_experiment_program(program)
        source_ref = self.store.write_text(
            f"{run_dir}/implementation.py", program.python_code.rstrip() + "\n"
        )
        self.store.write_json(f"{run_dir}/program.json", program)
        agent = ExperimentAgent(self.store, self.router.experimenter)
        first = agent.run_program(program=program, name=proposal.title, mode="full")
        second = agent.run_program(program=program, name=proposal.title, mode="full")
        self.store.write_json(f"{run_dir}/replications.json", {"first": first, "second": second})
        refs = _fresh_refs(
            self.store, [source_ref, *first.artifact_refs, *second.artifact_refs]
        )
        assert source_ref.sha256 is not None
        source_sha = source_ref.sha256
        attempted = list(view.subsystem_state.get("attempted", []))
        attempted.append(proposal.fingerprint)
        state_patch = {"attempted": attempted[-100:]}
        if not first.success or not second.success:
            return ActionOutcome(
                summary="The exact-replication experiment did not complete twice.",
                records=[
                    RecordDraft(
                        kind=RecordKind.obstruction,
                        title=f"Experiment blocked: {proposal.title}",
                        summary=f"First: {first.summary} Second: {second.summary}",
                        body=str(proposal.payload.get("protocol") or ""),
                        relation=(
                            RecordRelation.challenges
                            if proposal.parent_ids
                            else RecordRelation.none
                        ),
                        parent_ids=proposal.parent_ids,
                        evidence=EvidenceReceipt(
                            evidence_type=EvidenceType.system,
                            details={"first_success": first.success, "second_success": second.success},
                            artifact_refs=refs,
                        ),
                    )
                ],
                state_patch=state_patch,
            )
        assert first.validated_output is not None and second.validated_output is not None
        first_payload = first.validated_output.model_dump(mode="json")
        second_payload = second.validated_output.model_dump(mode="json")
        replicated = first_payload == second_payload
        result_refs = [ref for ref in refs if ref.path.endswith("/results.json")]
        result_hashes = {ref.sha256 for ref in result_refs if ref.sha256}
        replicated = replicated and len(result_refs) >= 2 and len(result_hashes) == 1
        output_sha = next(iter(result_hashes), content_fingerprint(first_payload))
        if not replicated:
            return ActionOutcome(
                summary="The same program and seeds produced non-identical structured outputs.",
                records=[
                    RecordDraft(
                        kind=RecordKind.obstruction,
                        title=f"Non-reproducible experiment: {proposal.title}",
                        summary="Exact replication failed; neither output was admitted as observed evidence.",
                        body=json.dumps(
                            {"first": first_payload, "second": second_payload},
                            ensure_ascii=False,
                            indent=2,
                        )[:30_000],
                        relation=(
                            RecordRelation.challenges
                            if proposal.parent_ids
                            else RecordRelation.none
                        ),
                        parent_ids=proposal.parent_ids,
                        evidence=EvidenceReceipt(
                            evidence_type=EvidenceType.system,
                            details={"program_sha256": source_sha, "replicated": False},
                            artifact_refs=refs,
                        ),
                    )
                ],
                state_patch=state_patch,
            )
        output = first.validated_output
        body = (
            f"Question: {proposal.payload.get('question', '')}\n\n"
            f"Hypothesis: {proposal.payload.get('hypothesis', '')}\n\n"
            f"Protocol: {output.protocol}\n\n"
            f"Program interpretation (not independently verified): {output.interpretation}\n\n"
            f"Summaries: {json.dumps(output.summaries, ensure_ascii=False, sort_keys=True)}"
        )
        return ActionOutcome(
            summary=f"Exactly replicated {len(output.observations)} raw observation(s).",
            records=[
                RecordDraft(
                    kind=RecordKind.experiment,
                    title=output.experiment,
                    summary=(
                        f"The same hashed program and seeds produced identical structured output "
                        f"twice ({len(output.observations)} observations)."
                    ),
                    body=body,
                    relation=(
                        RecordRelation.answers
                        if proposal.parent_ids
                        else RecordRelation.none
                    ),
                    parent_ids=proposal.parent_ids,
                    evidence=EvidenceReceipt(
                        evidence_type=EvidenceType.execution,
                        details={
                            "success": True,
                            "replicated": True,
                            "program_sha256": source_sha,
                            "output_sha256": output_sha,
                            "seeds": program.seeds,
                            "interpretation_unverified": True,
                        },
                        artifact_refs=refs,
                    ),
                )
            ],
            state_patch=state_patch,
        )


def _fresh_refs(store: ArtifactStore, refs: list[ArtifactRef]) -> list[ArtifactRef]:
    fresh: list[ArtifactRef] = []
    for ref in refs:
        if ref.path and store.exists(ref.path):
            current = store.artifact_ref(ref.path)
            if current.path not in {item.path for item in fresh}:
                fresh.append(current)
    return fresh
