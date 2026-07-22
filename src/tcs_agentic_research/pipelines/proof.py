"""Lean/LEAP proof evidence pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

from ..agents.theorem_prover import TheoremProverAgent
from ..artifact_store import ArtifactStore
from ..leap.graph import ProofGraph
from ..leap.lean import LeanVerifier
from ..llm import LLMRouter
from ..schemas import (
    ArtifactRef,
    CriterionResult,
    EvidenceStrength,
    Finding,
    FindingPolarity,
    FindingStatus,
    LeanGoalDraft,
    LeanStatement,
    ProofGoalReview,
    WorkItem,
    WorkKind,
    WorkResult,
)


class ProofPipeline:
    def __init__(
        self,
        store: ArtifactStore,
        router: LLMRouter,
        *,
        prompt_dir: str | None,
    ):
        self.store = store
        self.router = router
        self.prompt_dir = prompt_dir

    def run(
        self, item: WorkItem, run_dir: str, *, prior_context: dict[str, Any]
    ) -> WorkResult:
        verifier = LeanVerifier(
            self.store,
            timeout_seconds=self.router.leap.compiler_timeout_seconds,
            memory_mb=self.router.leap.compiler_memory_mb,
        )
        verifier.ensure_project()
        if not verifier.available():
            return WorkResult(
                work_id=item.work_id,
                outcome="blocked",
                failure_class="operational",
                summary="Lean is unavailable; no formulation call was spent.",
                errors=["Neither lake nor lean is available on PATH."],
                next_steps=["Install the configured Lean toolchain and retry."],
            )
        mock = LeanGoalDraft(name="dry_run_goal", statement="∀ n : Nat, n = n")
        required_names = _required_lean_names(item)
        prior_statements = " ".join(
            str(row.get("statement") or "")
            for row in prior_context.get("accepted_prior_evidence", [])
            if isinstance(row, dict)
        )
        already_verified = [
            name for name in required_names if _mentions_lean_name(prior_statements, name)
        ]
        missing_names = [name for name in required_names if name not in already_verified]
        messages = [
            {
                "role": "system",
                "content": (
                    "Formulate the smallest nontrivial Lean proposition directly advancing the "
                    "evidence requirement. Build on the supplied verified prior evidence and prefer "
                    "the smallest still-missing proposition. Return only a proposition type with "
                    "explicit binders. Use TCSResearch.Basic and Lean core. Do not invent unavailable "
                    "APIs or retreat to reflexivity, True, or an unrelated library fact."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "work_item": item.model_dump(mode="json"),
                        "prior_context": prior_context,
                        "required_lean_identifiers": required_names,
                        "already_verified_identifiers": already_verified,
                        "missing_identifiers": missing_names,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        refs: list[ArtifactRef] = [
            self.store.write_json(f"{run_dir}/proof_input.json", {"messages": messages})
        ]
        goal: LeanStatement | None = None
        last_error = ""
        for attempt in range(2):
            draft = self.router.complete_structured(
                task_type="proof_formulation",
                messages=messages,
                schema=LeanGoalDraft,
                mock_output=mock if self.router.dry_run else None,
            )
            refs.append(
                self.store.write_json(
                    f"{run_dir}/lean_goal_attempt_{attempt + 1}.json", draft
                )
            )
            candidate = LeanStatement(
                name=draft.name,
                statement=draft.statement,
                imports=["TCSResearch.Basic"],
                namespace=draft.namespace,
            )
            _, check = verifier.elaborate_statement(candidate)
            if check.log_path and self.store.exists(check.log_path):
                refs.append(self.store.artifact_ref(check.log_path))
            if check.accepted and not _obviously_trivial(candidate.statement):
                goal = candidate
                break
            last_error = (
                check.reason
                if not check.accepted
                else "Goal is a trivial or irrelevant tautology."
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Repair the proposition after elaboration or relevance rejection. Preserve "
                        "the research dependency, simplify syntax, and do not use reflexivity or True."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_item": item.model_dump(mode="json"),
                            "rejected_goal": draft.model_dump(mode="json"),
                            "error": last_error,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        if goal is None:
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="invalid",
                summary="No nontrivial elaborated Lean proposition was produced.",
                artifact_refs=refs,
                errors=[last_error or "formulation failed"],
                next_steps=["Use a smaller combinatorial proposition."],
            )
        refs.append(self.store.write_json(f"{run_dir}/lean_goal.json", goal))
        if self.router.dry_run:
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                summary="Dry run elaborated a goal but created no proof evidence.",
                artifact_refs=refs,
            )
        review = self.router.complete_structured(
            task_type="proof_formulation_review",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Review whether this exact proposition is nontrivial and lies on a concrete "
                        "dependency path to the evidence requirement. Reject generic library facts, "
                        "reflexivity in disguise, omitted assumptions, and unusably weak facts. Set "
                        "closes_requirement true only when proving this proposition, together with the "
                        "listed prior verified evidence, covers the entire atomic requirement. A useful "
                        "supporting lemma may be accepted while closes_requirement remains false."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "work_item": item.model_dump(mode="json"),
                            "goal": goal.model_dump(mode="json"),
                            "accepted_prior_evidence": prior_context.get(
                                "accepted_prior_evidence", []
                            ),
                            "required_lean_identifiers": required_names,
                            "already_verified_identifiers": already_verified,
                            "missing_identifiers": missing_names,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=ProofGoalReview,
            allow_repair=False,
        )
        refs.append(self.store.write_json(f"{run_dir}/lean_goal_review.json", review))
        if not review.accepted:
            return WorkResult(
                work_id=item.work_id,
                outcome="partial",
                failure_class="method",
                summary="The Lean goal failed dependency/relevance review.",
                artifact_refs=refs,
                errors=review.issues,
                next_steps=[review.route_to_requirement],
            )
        before: set[str] = set()
        if self.store.exists(ProofGraph.DB_PATH):
            before = {
                node.fingerprint
                for node in ProofGraph(self.store).proved_nodes(limit=10_000)
            }
        proof_result = TheoremProverAgent(
            self.store, self.router, prompt_dir=self.prompt_dir
        ).prove(
            goal,
            context=(
                f"{item.instruction}\nDependency route: {review.route_to_requirement}"
            ),
        )
        refs.append(self.store.write_json(f"{run_dir}/lean_result.json", proof_result))
        findings: list[Finding] = []
        if proof_result.status == "proved":
            findings.append(
                Finding(
                    work_id=item.work_id,
                    question_id=item.question_id,
                    requirement_id=item.requirement_id,
                    kind=WorkKind.proof,
                    statement=f"Lean verified `{goal.name} : {goal.statement}`.",
                    status=FindingStatus.verified,
                    polarity=FindingPolarity.supports,
                    strength=EvidenceStrength.conclusive,
                    scope=review.route_to_requirement,
                    evidence_refs=proof_result.proved_artifacts,
                    source_ids=[proof_result.result_id],
                )
            )
        elif self.store.exists(ProofGraph.DB_PATH):
            graph = ProofGraph(self.store)
            for node in graph.proved_nodes(limit=10_000):
                if node.fingerprint in before:
                    continue
                evidence: list[ArtifactRef] = []
                if node.proof_artifact_path and self.store.exists(node.proof_artifact_path):
                    evidence.append(self.store.artifact_ref(node.proof_artifact_path))
                findings.append(
                    Finding(
                        work_id=item.work_id,
                        question_id=item.question_id,
                        requirement_id=item.requirement_id,
                        kind=WorkKind.proof,
                        statement=(
                            "Lean verified new supporting lemma "
                            f"`{node.goal.name} : {node.goal.statement}`."
                        ),
                        status=FindingStatus.verified,
                        polarity=FindingPolarity.characterizes,
                        strength=EvidenceStrength.preliminary,
                        scope=review.route_to_requirement,
                        evidence_refs=evidence,
                        source_ids=[node.node_id],
                        caveats=["This supporting lemma does not close the parent requirement."],
                    )
                )
        root_proved = proof_result.status == "proved"
        cumulatively_verified = [
            name
            for name in required_names
            if _mentions_lean_name(prior_statements, name)
            or _mentions_lean_name(f"{goal.name} {goal.statement}", name)
        ]
        identifier_coverage = not required_names or set(cumulatively_verified) == set(
            required_names
        )
        closes_requirement = (
            root_proved and review.closes_requirement and identifier_coverage
        )
        criteria = [
            CriterionResult(
                criterion=criterion,
                satisfied=closes_requirement,
                detail=(
                    "The root is kernel checked and the relevance review says it closes the full gap."
                    if closes_requirement
                    else "The root is kernel checked but is only one supporting proposition for the gap."
                    if root_proved
                    else f"Produced {len(findings)} new verified supporting lemma(s)."
                    if findings
                    else proof_result.proof_dag_summary[:1200] or "No verified node."
                ),
            )
            for criterion in item.success_criteria
        ]
        next_steps = list(proof_result.recommended_next_steps)
        if root_proved and not closes_requirement:
            still_missing = [
                name for name in required_names if name not in cumulatively_verified
            ]
            next_steps.append(
                "Formulate the smallest proposition still missing from the full evidence requirement"
                + (f": {', '.join(still_missing)}." if still_missing else ".")
            )
        return WorkResult(
            work_id=item.work_id,
            outcome="done" if root_proved else "partial",
            failure_class="none" if findings else "method",
            evidence_level=(
                "conclusive"
                if root_proved
                else "preliminary"
                if findings
                else "none"
            ),
            requirement_satisfied=closes_requirement,
            criteria=criteria,
            summary=f"Persistent LEAP search ended with `{proof_result.status}`.",
            findings=findings,
            artifact_refs=[*refs, *proof_result.artifact_refs],
            errors=[] if findings else [proof_result.proof_dag_summary],
            next_steps=next_steps,
        )


def _required_lean_names(item: WorkItem) -> list[str]:
    text = " ".join([item.instruction, *item.success_criteria])
    names: list[str] = []
    for value in re.findall(r"`([^`]+)`", text):
        candidate = value.strip()
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*",
            candidate,
        ):
            continue
        name = candidate.rsplit(".", 1)[-1]
        if name not in names and name.lower() not in {"lean", "true", "false"}:
            names.append(name)
    return names


def _mentions_lean_name(text: str, name: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_']){re.escape(name)}(?![A-Za-z0-9_'])", text))


def _obviously_trivial(statement: str) -> bool:
    normalized = re.sub(r"\s+", " ", statement.strip())
    if normalized in {"True", "∀ _ : Unit, True"}:
        return True
    body = normalized.rsplit(",", 1)[-1].strip().strip("()")
    match = re.fullmatch(r"(.+?)\s*=\s*(.+)", body)
    return bool(match and match.group(1).strip("() ") == match.group(2).strip("() "))
