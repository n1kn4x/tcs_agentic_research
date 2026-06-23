"""Named internal LEAP agents.

These small classes make the LEAP architecture explicit and reusable. The high-level
``LEAPHarness`` wires equivalent behavior into one control loop; advanced deployments can swap
these classes for richer implementations.
"""

from __future__ import annotations

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter
from ..prompt_loader import render_prompt
from ..schemas import LeanStatement
from .dag import ProofDAG
from .harness import BlueprintCandidate, DecompositionReview, FormalProofCandidate
from .lean import LeanVerifier
from .search import LeanSearchIndex


class StateReader:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.search = LeanSearchIndex(store)

    def read(self, goal: LeanStatement) -> dict[str, object]:
        return {
            "goal": goal.model_dump(mode="json"),
            "nomenclature": self.store.read_text(ArtifactStore.NOMENCLATURE)
            if self.store.exists(ArtifactStore.NOMENCLATURE)
            else "",
            "local_lean_hits": [h.model_dump(mode="json") for h in self.search.search(goal.name + " " + goal.statement)],
        }


class InformalProofGenerator:
    def __init__(self, router: LLMRouter, *, prompt_dir: str | None = None):
        self.router = router
        self.prompt_dir = prompt_dir

    def generate(self, goal: LeanStatement, context: str) -> str:
        return self.router.complete_text(
            task_type="theorem_proving",
            messages=[
                {"role": "system", "content": "Give a concise informal proof strategy."},
                {"role": "user", "content": f"Context:\n{context}\nGoal:\n{goal.model_dump_json()}"},
            ],
        )


class FormalProofGenerator:
    def __init__(self, router: LLMRouter, *, prompt_dir: str | None = None):
        self.router = router
        self.prompt_dir = prompt_dir

    def generate(self, goal: LeanStatement, context: str) -> FormalProofCandidate:
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=[
                {"role": "system", "content": render_prompt("leap_direct_prover", override_dir=self.prompt_dir)},
                {"role": "user", "content": f"Context:\n{context}\nGoal:\n{goal.model_dump_json()}"},
            ],
            schema=FormalProofCandidate,
        )


class Reviser(FormalProofGenerator):
    def revise(self, goal: LeanStatement, candidate: FormalProofCandidate, compiler_error: str, context: str) -> FormalProofCandidate:
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=[
                {"role": "system", "content": render_prompt("leap_reviser", override_dir=self.prompt_dir)},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\nGoal:\n{goal.model_dump_json()}\nCompiler error:\n{compiler_error}\nCandidate:\n{candidate.model_dump_json()}",
                },
            ],
            schema=FormalProofCandidate,
        )


class InformalBlueprintGenerator:
    def __init__(self, router: LLMRouter, *, prompt_dir: str | None = None):
        self.router = router
        self.prompt_dir = prompt_dir

    def generate(self, goal: LeanStatement, context: str) -> str:
        return self.router.complete_text(
            task_type="theorem_proving",
            messages=[
                {"role": "system", "content": "Propose a non-circular informal lemma decomposition."},
                {"role": "user", "content": f"Context:\n{context}\nGoal:\n{goal.model_dump_json()}"},
            ],
        )


class FormalSketchGenerator:
    def __init__(self, router: LLMRouter, *, prompt_dir: str | None = None):
        self.router = router
        self.prompt_dir = prompt_dir

    def generate(self, goal: LeanStatement, context: str) -> BlueprintCandidate:
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=[
                {"role": "system", "content": render_prompt("leap_blueprint", override_dir=self.prompt_dir)},
                {"role": "user", "content": f"Context:\n{context}\nGoal:\n{goal.model_dump_json()}"},
            ],
            schema=BlueprintCandidate,
        )


class DecompositionReviewer:
    def __init__(self, router: LLMRouter, *, prompt_dir: str | None = None):
        self.router = router
        self.prompt_dir = prompt_dir

    def review(self, goal: LeanStatement, blueprint: BlueprintCandidate, context: str) -> DecompositionReview:
        return self.router.complete_structured(
            task_type="theorem_proving",
            messages=[
                {"role": "system", "content": render_prompt("leap_decomposition_reviewer", override_dir=self.prompt_dir)},
                {"role": "user", "content": f"Context:\n{context}\nGoal:\n{goal.model_dump_json()}\nBlueprint:\n{blueprint.model_dump_json()}"},
            ],
            schema=DecompositionReview,
        )


class LeanVerifierAgent(LeanVerifier):
    pass


class StateWriter:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def commit_dag(self, dag: ProofDAG):
        return self.store.write_json(f"LeanProject/ProofDAGs/{dag.dag_id}.json", dag)
