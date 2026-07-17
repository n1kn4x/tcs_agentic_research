"""Fresh, typed LLM calls used by the LEAP controller."""

from __future__ import annotations

import json
from typing import Sequence

from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from .models import (
    BlueprintCandidate,
    BlueprintChild,
    DecompositionReview,
    FormalProofCandidate,
    InformalProof,
    LeanDiagnostic,
    ProofContext,
    SketchCandidate,
)
from .state import compact_context


class LEAPAgents:
    def __init__(self, router: LLMRouter, *, prompt_dir: str | None = None):
        self.router = router
        self.prompt_dir = prompt_dir

    def informal_proof(self, context: ProofContext) -> InformalProof:
        mock = InformalProof(strategy="Dry-run: no mathematical proof is claimed.")
        return self.router.complete_structured(
            task_type="proof_planning",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt("leap_informal_prover", override_dir=self.prompt_dir),
                },
                {"role": "user", "content": compact_context(context)},
            ],
            schema=InformalProof,
            mock_output=mock if self.router.dry_run else None,
        )

    def formal_proof(
        self,
        context: ProofContext,
        informal: InformalProof,
    ) -> FormalProofCandidate:
        mock = FormalProofCandidate(
            informal_proof=informal.strategy,
            proof="by\n  sorry",
            notes=["Dry-run placeholder; it cannot be accepted."],
        )
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt("leap_direct_prover", override_dir=self.prompt_dir),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "state": compact_context(context, max_chars=12000),
                            "informal_plan": informal.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=FormalProofCandidate,
            mock_output=mock if self.router.dry_run else None,
            allow_repair=False,
        )

    def revise_proof(
        self,
        context: ProofContext,
        informal: InformalProof,
        candidate: FormalProofCandidate,
        diagnostics: Sequence[LeanDiagnostic],
    ) -> FormalProofCandidate:
        return self.router.complete_structured(
            task_type="proof_revision",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt("leap_reviser", override_dir=self.prompt_dir),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "state": compact_context(context, max_chars=5000),
                            "informal_plan": informal.model_dump(mode="json"),
                            "candidate": candidate.model_dump(mode="json"),
                            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=FormalProofCandidate,
            mock_output=candidate if self.router.dry_run else None,
            allow_repair=False,
        )

    def blueprint(self, context: ProofContext) -> BlueprintCandidate:
        mock = BlueprintCandidate(
            overview="Dry-run decomposition repeats the goal and must be rejected.",
            parent_strategy="No parent proof is claimed.",
            children=[
                BlueprintChild(
                    label="same_goal",
                    statement=context.goal.statement,
                    rationale="Dry-run sentinel.",
                )
            ],
        )
        return self.router.complete_structured(
            task_type="blueprint_generation",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt("leap_blueprint", override_dir=self.prompt_dir),
                },
                {"role": "user", "content": compact_context(context)},
            ],
            schema=BlueprintCandidate,
            mock_output=mock if self.router.dry_run else None,
        )

    def formal_sketch(
        self,
        context: ProofContext,
        blueprint: BlueprintCandidate,
        child_declarations: Sequence[dict[str, object]],
    ) -> SketchCandidate:
        mock = SketchCandidate(
            parent_proof="by\n  sorry",
            notes=["Dry-run placeholder; it cannot be accepted."],
        )
        return self.router.complete_structured(
            task_type="sketch_generation",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt("leap_sketch", override_dir=self.prompt_dir),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "state": compact_context(context, max_chars=7000),
                            "blueprint": _blueprint_payload(blueprint),
                            "application_owned_child_declarations": list(child_declarations),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=SketchCandidate,
            mock_output=mock if self.router.dry_run else None,
            allow_repair=False,
        )

    def revise_sketch(
        self,
        context: ProofContext,
        blueprint: BlueprintCandidate,
        child_declarations: Sequence[dict[str, object]],
        candidate: SketchCandidate,
        diagnostics: Sequence[LeanDiagnostic],
    ) -> SketchCandidate:
        return self.router.complete_structured(
            task_type="sketch_revision",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt("leap_sketch_reviser", override_dir=self.prompt_dir),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "state": compact_context(context, max_chars=5000),
                            "blueprint": _blueprint_payload(blueprint),
                            "application_owned_child_declarations": list(child_declarations),
                            "candidate": candidate.model_dump(mode="json"),
                            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=SketchCandidate,
            mock_output=candidate if self.router.dry_run else None,
            allow_repair=False,
        )

    def review_decomposition(
        self,
        context: ProofContext,
        blueprint: BlueprintCandidate,
        child_declarations: Sequence[dict[str, object]],
        sketch: SketchCandidate,
    ) -> DecompositionReview:
        mock = DecompositionReview(
            accept=False,
            score=0.0,
            reasons=["Dry-run reviewer cannot establish useful simplification."],
        )
        return self.router.complete_structured(
            task_type="decomposition_review",
            messages=[
                {
                    "role": "system",
                    "content": render_prompt(
                        "leap_decomposition_reviewer", override_dir=self.prompt_dir
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "state": compact_context(context, max_chars=7000),
                            "blueprint": _blueprint_payload(blueprint),
                            "child_declarations": list(child_declarations),
                            "verified_parent_proof": sketch.parent_proof,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=DecompositionReview,
            mock_output=mock if self.router.dry_run else None,
        )


def _blueprint_payload(blueprint: BlueprintCandidate) -> dict[str, object]:
    """Keep exact child types while bounding prose duplicated into sketch/review prompts."""
    return {
        "overview": blueprint.overview[:1500],
        "parent_strategy": blueprint.parent_strategy[:1500],
        "children": [
            {
                "label": child.label,
                "statement": child.statement,
                "required": child.required,
                "rationale": child.rationale[:500],
            }
            for child in blueprint.children
        ],
        "library_notes": [note[:500] for note in blueprint.library_notes[:8]],
    }
