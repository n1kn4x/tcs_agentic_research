"""Resumable depth-first LEAP search controller."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..artifact_store import ArtifactStore
from ..llm import LLMRouter, ModelBudgetExceeded, StructuredLLMError
from ..schemas import (
    ArtifactRef,
    LeanCompilerLog,
    LeanStatement,
    LeapSettings,
    ProofGoal,
    TheoremProverResult,
    new_id,
)
from .agents import LEAPAgents
from .graph import (
    GraphInvariantError,
    ProofGraph,
    canonical_lemma_name,
    normalize_lean_source,
    statement_fingerprint,
)
from .lean import LeanVerifier
from .models import (
    AndNode,
    AndStatus,
    AttemptRecord,
    BlueprintCandidate,
    LeanDiagnostic,
    OrNode,
    OrStatus,
    ProofContext,
    RetrievalHit,
    VerificationResult,
)
from .render import (
    MaterializationError,
    ProofMaterializer,
    referenced_support_ids,
    render_direct_module,
    render_sketch_module,
)
from .retrieval import LeanRetriever, proved_support_nodes
from .state import StateReader


@dataclass
class _RunBudget:
    settings: LeapSettings
    started: float = field(default_factory=time.monotonic)
    stop_reason: str = ""

    def seconds_left(self) -> int:
        return max(0, int(self.settings.max_wall_seconds - (time.monotonic() - self.started)))

    def allows(self, *, depth: int, graph_nodes: int) -> bool:
        if depth > self.settings.max_depth:
            self.stop_reason = f"maximum proof depth {self.settings.max_depth} reached"
            return False
        # Reaching the node cap prevents graph expansion, not direct work on existing nodes.
        # `_attempt_decomposition` enforces the expansion limit transactionally.
        if self.seconds_left() <= 0:
            self.stop_reason = f"wall-clock budget {self.settings.max_wall_seconds}s exhausted"
            return False
        return True


class SearchController:
    """Orchestrate direct proving, verified decomposition, DFS, and final assembly."""

    def __init__(
        self,
        store: ArtifactStore,
        router: LLMRouter,
        *,
        prompt_dir: str | None = None,
        settings: LeapSettings | None = None,
    ):
        self.store = store
        self.router = router
        self.settings = settings or router.leap
        self.verifier = LeanVerifier(
            store,
            timeout_seconds=self.settings.compiler_timeout_seconds,
            memory_mb=self.settings.compiler_memory_mb,
        )
        self.graph: ProofGraph | None = None
        self.retriever: LeanRetriever | None = None
        self.reader: StateReader | None = None
        self.materializer: ProofMaterializer | None = None
        self.agents = LEAPAgents(router, prompt_dir=prompt_dir)
        self._run_id = ""
        self._worker_id = new_id("leap_worker")
        self._user_context = ""
        self._budget = _RunBudget(self.settings)
        self._verifications: list[VerificationResult] = []
        self._artifact_paths: set[str] = set()
        self._unavailable = False
        self._operational_error = ""
        self._active_stack: list[str] = []

    def prove(self, target: LeanStatement, *, context: str = "") -> TheoremProverResult:
        self.verifier.ensure_project()
        environment = self.verifier.environment_fingerprint()
        elaborated, statement_check = self.verifier.elaborate_statement(target)
        self._remember_verification(statement_check)
        if not statement_check.accepted:
            status = "unavailable" if statement_check.exit_code == 127 else "exhausted"
            return self._early_result(
                target,
                status=status,
                summary=f"Lean rejected the root statement before search: {statement_check.reason}",
            )

        self.graph = ProofGraph(self.store)
        self.retriever = LeanRetriever(self.store, self.graph)
        self.reader = StateReader(self.graph, self.retriever)
        self.materializer = ProofMaterializer(self.graph)
        root = self.graph.register_goal(
            target,
            environment_fingerprint=environment,
            elaborated_statement=elaborated,
        )
        run = self.graph.create_or_resume_run(root, target, user_context=context)
        self._run_id = run.run_id
        self._user_context = context
        self._artifact_paths.add(ProofGraph.DB_PATH)

        try:
            if root.status != OrStatus.proved:
                self._solve_or(root.node_id, depth=0)
        except ModelBudgetExceeded as exc:
            self._budget.stop_reason = str(exc)
        except StructuredLLMError as exc:
            # Model transport/schema failures are operational, not mathematical refutations.  The
            # graph remains resumable and every completed attempt is already durable.
            self._operational_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # preserve graph progress at the subsystem boundary
            self._operational_error = f"{type(exc).__name__}: {exc}"

        root = self.graph.get_or(root.node_id)
        if root.status == OrStatus.proved:
            final = self._materialize_and_verify(root, target)
            if final is not None:
                final_ref, verification = final
                self.graph.update_run(
                    self._run_id, status="proved", final_artifact_path=final_ref.path
                )
                return self._result(
                    target,
                    root,
                    status="proved",
                    proved_artifacts=[
                        final_ref,
                        *(
                            [self.store.artifact_ref(verification.log_path)]
                            if verification.log_path and self.store.exists(verification.log_path)
                            else []
                        ),
                    ],
                )
            self._operational_error = self._operational_error or (
                "The graph root is proved, but final no-placeholder assembly did not compile."
            )

        status = "unavailable" if self._unavailable else self._partial_status(root)
        self.graph.update_run(self._run_id, status=status)
        return self._result(target, root, status=status)

    # ------------------------------------------------------------------
    # Depth-first AND-OR search
    # ------------------------------------------------------------------
    def _solve_or(self, node_id: str, *, depth: int) -> bool:
        graph = self._graph
        node = graph.get_or(node_id)
        if node.status == OrStatus.proved:
            return True
        if node_id in self._active_stack:
            raise GraphInvariantError(f"runtime recursion cycle at {node_id}")
        if not self._budget.allows(depth=depth, graph_nodes=graph.node_count()):
            return False
        if not graph.acquire_lease(
            node_id,
            owner=self._worker_id,
            ttl_seconds=max(
                self.settings.max_wall_seconds + 300,
                self.settings.compiler_timeout_seconds * 2,
                3600,
            ),
        ):
            self._budget.stop_reason = f"OR node {node_id} is leased by another LEAP worker"
            return False

        self._active_stack.append(node_id)
        try:
            # Cheap, compiler-checked tactics solve many definitional, simplification, and finite
            # decidable goals without model uncertainty.  Run them even before resuming a branch:
            # an existing decomposition may have been created by an older invocation that lacked
            # this deterministic pre-pass.
            if not self.router.dry_run and self._attempt_deterministic(node, depth=depth):
                return True

            # Continue active branches and paused branches that already proved at least one child.
            # A wholly stalled paused branch is deferred until after alternatives are generated;
            # otherwise every resumed invocation could spend its entire budget repeating the first
            # failed DFS branch and never genuinely backtrack.
            decompositions = graph.decompositions(node_id)
            priority, deferred = self._partition_decompositions(decompositions)
            for decomposition in priority:
                if decomposition.status == AndStatus.proved:
                    graph.propagate()
                    return graph.get_or(node_id).status == OrStatus.proved
                if self._solve_and(decomposition, depth=depth):
                    return True

            if self._attempt_direct(node, depth=depth):
                return True

            for _ in range(self.settings.blueprint_attempts_per_node):
                if not self._budget.allows(depth=depth, graph_nodes=graph.node_count()):
                    break
                new_decomposition = self._attempt_decomposition(node, depth=depth)
                if new_decomposition is None:
                    continue
                if self._solve_and(new_decomposition, depth=depth):
                    return True
            for decomposition in deferred:
                if self._solve_and(decomposition, depth=depth):
                    return True
            return graph.get_or(node_id).status == OrStatus.proved
        finally:
            self._active_stack.pop()
            graph.release_lease(node_id, owner=self._worker_id)

    def _partition_decompositions(
        self, decompositions: Sequence[AndNode]
    ) -> tuple[list[AndNode], list[AndNode]]:
        priority: list[AndNode] = []
        deferred: list[AndNode] = []
        for decomposition in decompositions:
            has_proved_child = any(
                child.required
                and self._graph.get_or(child.child_or_id).status == OrStatus.proved
                for child in decomposition.children
            )
            if decomposition.status != AndStatus.paused or has_proved_child:
                priority.append(decomposition)
            else:
                deferred.append(decomposition)
        return priority, deferred

    def _solve_and(self, decomposition: AndNode, *, depth: int) -> bool:
        graph = self._graph
        graph.activate_and(decomposition.node_id)
        for child in decomposition.children:
            if not child.required:
                continue
            if not self._solve_or(child.child_or_id, depth=depth + 1):
                graph.pause_and(decomposition.node_id)
                return False
        graph.propagate()
        return graph.get_or(decomposition.parent_or_id).status == OrStatus.proved

    def _attempt_deterministic(self, node: OrNode, *, depth: int) -> bool:
        """Try a tiny tactic portfolio before spending model calls.

        These proof bodies are intentionally generic rather than inferred from goal syntax.  Lean
        elaboration and kernel checking remain the only acceptance criteria, while candidate
        digests prevent failed tactics from being recompiled on resumed invocations.
        """
        for label, proof in deterministic_proof_candidates():
            if not self._budget.allows(depth=depth, graph_nodes=self._graph.node_count()):
                return False
            digest = _sha256(proof)
            if self._graph.candidate_seen(node.node_id, "deterministic", digest):
                continue
            attempt_id = new_id("leap_attempt")
            code = render_direct_module(node.goal, proof)
            rel_file = f"TCSResearch/Generated/{attempt_id}_{label}.lean"
            started = time.monotonic()
            verification = self.verifier.check_direct(
                code, rel_file=rel_file, target_name=node.goal.name
            )
            duration = time.monotonic() - started
            self._remember_verification(verification)
            self._record_attempt(
                node.node_id,
                attempt_id=attempt_id,
                mode="deterministic",
                outcome="verified" if verification.accepted else "compiler_rejected",
                digest=digest,
                artifact_path=verification.source_path,
                diagnostics=verification.diagnostics,
                duration=duration,
                note=verification.reason or f"Lean accepted deterministic tactic `{label}`.",
            )
            if verification.accepted:
                self._graph.commit_direct_proof(
                    node.node_id,
                    proof=proof,
                    artifact_path=verification.source_path,
                )
                return True
            if verification.exit_code == 127:
                self._unavailable = True
                return False
        return False

    def _attempt_direct(self, node: OrNode, *, depth: int) -> bool:
        diagnostics: list[LeanDiagnostic] = []
        for _ in range(self.settings.direct_attempts_per_node):
            if not self._budget.allows(depth=depth, graph_nodes=self._graph.node_count()):
                return False
            context = self._context(node.node_id, diagnostics=diagnostics)
            try:
                informal = self.agents.informal_proof(context)
                self._write_json_artifact(node.node_id, "informal_plan", informal)
                context = self._context(
                    node.node_id,
                    informal_queries=[informal.strategy, *informal.search_queries],
                    diagnostics=diagnostics,
                )
                candidate = self.agents.formal_proof(context, informal)
            except ModelBudgetExceeded:
                raise
            except StructuredLLMError as exc:
                self._record_control_failure(node.node_id, "direct_generation", str(exc))
                continue

            parent_attempt_id: str | None = None
            support_pool: dict[str, OrNode] = {}
            for revision in range(self.settings.direct_revisions + 1):
                attempt_id = new_id("leap_attempt")
                candidate_ref = self._write_json_artifact(
                    node.node_id, f"{attempt_id}_candidate", candidate
                )
                digest = _sha256(candidate.proof)
                if self._graph.candidate_seen(node.node_id, "direct", digest):
                    duplicate = LeanDiagnostic(
                        severity="error",
                        message="This exact proof body was already attempted; make a substantive change.",
                    )
                    self._record_attempt(
                        node.node_id,
                        attempt_id=attempt_id,
                        mode="direct",
                        outcome="duplicate",
                        digest=digest,
                        artifact_path=candidate_ref.path,
                        diagnostics=[duplicate],
                        retrieval=[*context.proved_lemmas, *context.library_results],
                        parent_attempt_id=parent_attempt_id,
                    )
                    diagnostics = [duplicate]
                    if revision < self.settings.direct_revisions:
                        candidate = self.agents.revise_proof(
                            context, informal, candidate, diagnostics
                        )
                        parent_attempt_id = attempt_id
                        continue
                    break

                current_supports = proved_support_nodes(self._graph, context.proved_lemmas)
                support_pool.update({support.node_id: support for support in current_supports})
                supports = list(support_pool.values())
                support_code = self._materializer.support_declarations(
                    [support.node_id for support in supports]
                )
                code = render_direct_module(
                    node.goal, candidate.proof, support_declarations=support_code
                )
                rel_file = f"TCSResearch/Generated/{attempt_id}_direct.lean"
                started = time.monotonic()
                verification = self.verifier.check_direct(
                    code, rel_file=rel_file, target_name=node.goal.name
                )
                duration = time.monotonic() - started
                self._remember_verification(verification)
                self._record_attempt(
                    node.node_id,
                    attempt_id=attempt_id,
                    mode="direct",
                    outcome="verified" if verification.accepted else "compiler_rejected",
                    digest=digest,
                    artifact_path=verification.source_path or candidate_ref.path,
                    diagnostics=verification.diagnostics,
                    retrieval=[*context.proved_lemmas, *context.library_results],
                    parent_attempt_id=parent_attempt_id,
                    duration=duration,
                    note=verification.reason,
                )
                if verification.accepted:
                    dependency_ids = referenced_support_ids(candidate.proof, supports)
                    self._graph.commit_direct_proof(
                        node.node_id,
                        proof=candidate.proof,
                        artifact_path=verification.source_path,
                        dependency_or_ids=dependency_ids,
                    )
                    return True
                if verification.exit_code == 127:
                    self._unavailable = True
                    return False
                diagnostics = verification.diagnostics or [
                    LeanDiagnostic(severity="error", message=verification.reason)
                ]
                if revision >= self.settings.direct_revisions:
                    break
                context = self._context(
                    node.node_id,
                    informal_queries=[informal.strategy, *informal.search_queries],
                    diagnostics=diagnostics,
                )
                try:
                    candidate = self.agents.revise_proof(
                        context, informal, candidate, diagnostics
                    )
                except ModelBudgetExceeded:
                    raise
                except StructuredLLMError as exc:
                    self._record_control_failure(node.node_id, "direct_revision", str(exc))
                    break
                parent_attempt_id = attempt_id
        return False

    def _attempt_decomposition(self, node: OrNode, *, depth: int) -> AndNode | None:
        context = self._context(node.node_id)
        try:
            blueprint = self.agents.blueprint(context)
        except ModelBudgetExceeded:
            raise
        except StructuredLLMError as exc:
            self._record_control_failure(node.node_id, "blueprint_generation", str(exc))
            return None
        blueprint_ref = self._write_json_artifact(node.node_id, "blueprint", blueprint)
        blueprint_digest = _sha256(blueprint.model_dump_json())
        if self._graph.candidate_seen(node.node_id, "blueprint", blueprint_digest):
            self._record_attempt(
                node.node_id,
                attempt_id=new_id("leap_attempt"),
                mode="blueprint",
                outcome="duplicate",
                digest=blueprint_digest,
                artifact_path=blueprint_ref.path,
                note="The same decomposition blueprint was previously considered.",
            )
            return None

        structural_errors, proposed, child_declarations = self._prepare_blueprint(node, blueprint)
        if structural_errors:
            self._record_attempt(
                node.node_id,
                attempt_id=new_id("leap_attempt"),
                mode="blueprint",
                outcome="structurally_rejected",
                digest=blueprint_digest,
                artifact_path=blueprint_ref.path,
                diagnostics=[
                    LeanDiagnostic(severity="error", message=error)
                    for error in structural_errors
                ],
            )
            return None
        self._record_attempt(
            node.node_id,
            attempt_id=new_id("leap_attempt"),
            mode="blueprint",
            outcome="accepted_for_sketching",
            digest=blueprint_digest,
            artifact_path=blueprint_ref.path,
        )

        try:
            sketch = self.agents.formal_sketch(context, blueprint, child_declarations)
        except ModelBudgetExceeded:
            raise
        except StructuredLLMError as exc:
            self._record_control_failure(node.node_id, "sketch_generation", str(exc))
            return None

        parent_attempt_id: str | None = None
        verification: VerificationResult | None = None
        supports: list[OrNode] = []
        support_pool: dict[str, OrNode] = {}
        for revision in range(self.settings.sketch_revisions + 1):
            attempt_id = new_id("leap_attempt")
            sketch_ref = self._write_json_artifact(
                node.node_id, f"{attempt_id}_sketch", sketch
            )
            digest = _sha256(sketch.parent_proof)
            if self._graph.candidate_seen(node.node_id, "sketch", digest):
                diagnostics = [
                    LeanDiagnostic(
                        severity="error",
                        message="This exact parent sketch was already attempted; change it.",
                    )
                ]
                self._record_attempt(
                    node.node_id,
                    attempt_id=attempt_id,
                    mode="sketch",
                    outcome="duplicate",
                    digest=digest,
                    artifact_path=sketch_ref.path,
                    diagnostics=diagnostics,
                    parent_attempt_id=parent_attempt_id,
                )
            else:
                usage_errors = _sketch_usage_errors(sketch.parent_proof, child_declarations)
                if usage_errors:
                    verification = VerificationResult(
                        accepted=False,
                        reason="; ".join(usage_errors),
                        exit_code=2,
                        diagnostics=[
                            LeanDiagnostic(severity="error", message=error)
                            for error in usage_errors
                        ],
                    )
                else:
                    context = self._context(node.node_id)
                    current_supports = proved_support_nodes(
                        self._graph, context.proved_lemmas
                    )
                    support_pool.update(
                        {support.node_id: support for support in current_supports}
                    )
                    supports = list(support_pool.values())
                    support_code = self._materializer.support_declarations(
                        [support.node_id for support in supports]
                    )
                    child_goals = [item[1] for item in proposed]
                    code = render_sketch_module(
                        node.goal,
                        child_goals,
                        sketch.parent_proof,
                        support_declarations=support_code,
                    )
                    rel_file = f"TCSResearch/Generated/{attempt_id}_sketch.lean"
                    verification = self.verifier.check_sketch(
                        code,
                        rel_file=rel_file,
                        parent_name=node.goal.name,
                        child_names=[child.name for child in child_goals],
                    )
                    self._remember_verification(verification)
                self._record_attempt(
                    node.node_id,
                    attempt_id=attempt_id,
                    mode="sketch",
                    outcome="verified" if verification.accepted else "compiler_rejected",
                    digest=digest,
                    artifact_path=verification.source_path or sketch_ref.path,
                    diagnostics=verification.diagnostics,
                    retrieval=[*context.proved_lemmas, *context.library_results],
                    parent_attempt_id=parent_attempt_id,
                    note=verification.reason,
                )
                if verification.accepted:
                    break
                if verification.exit_code == 127:
                    self._unavailable = True
                    return None
                diagnostics = verification.diagnostics
            if revision >= self.settings.sketch_revisions:
                return None
            context = self._context(node.node_id, diagnostics=diagnostics)
            try:
                sketch = self.agents.revise_sketch(
                    context, blueprint, child_declarations, sketch, diagnostics
                )
            except ModelBudgetExceeded:
                raise
            except StructuredLLMError as exc:
                self._record_control_failure(node.node_id, "sketch_revision", str(exc))
                return None
            parent_attempt_id = attempt_id

        assert verification is not None and verification.accepted
        try:
            review = self.agents.review_decomposition(
                context, blueprint, child_declarations, sketch
            )
        except ModelBudgetExceeded:
            raise
        except StructuredLLMError as exc:
            self._record_control_failure(node.node_id, "decomposition_review", str(exc))
            return None
        review_ref = self._write_json_artifact(node.node_id, "decomposition_review", review)
        accepted = review.accept and review.score >= self.settings.reviewer_min_score
        effective_review = review.model_copy(update={"accept": accepted})
        self._record_attempt(
            node.node_id,
            attempt_id=new_id("leap_attempt"),
            mode="review",
            outcome="accepted" if accepted else "rejected",
            digest=_sha256(review.model_dump_json()),
            artifact_path=review_ref.path,
            note="; ".join(review.reasons) or review.suggested_direction,
        )
        if not accepted:
            return None
        new_child_count = sum(
            self._graph.find_or(
                source,
                environment_fingerprint=node.environment_fingerprint,
                elaborated_statement=elaborated,
            )
            is None
            for source, _, elaborated in proposed
        )
        if self._graph.node_count() + new_child_count > self.settings.max_nodes:
            self._budget.stop_reason = "decomposition would exceed the graph node budget"
            return None

        external_dependencies = referenced_support_ids(sketch.parent_proof, supports)
        try:
            return self._graph.commit_decomposition(
                node.node_id,
                blueprint=blueprint,
                children=[
                    (source, blueprint.children[index].required, canonical.name)
                    for index, (source, canonical, _) in enumerate(proposed)
                ],
                parent_proof=sketch.parent_proof,
                sketch_artifact_path=verification.source_path,
                review=effective_review,
                environment_fingerprint=node.environment_fingerprint,
                dependency_or_ids=external_dependencies,
                child_elaborations=[elaborated for _, _, elaborated in proposed],
            )
        except GraphInvariantError as exc:
            self._record_control_failure(node.node_id, "graph_commit", str(exc))
            return None

    # ------------------------------------------------------------------
    # Deterministic review and rendering helpers
    # ------------------------------------------------------------------
    def _prepare_blueprint(
        self, node: OrNode, blueprint: BlueprintCandidate
    ) -> tuple[
        list[str],
        list[tuple[LeanStatement, LeanStatement, str]],
        list[dict[str, object]],
    ]:
        errors: list[str] = []
        if len(blueprint.children) > self.settings.max_children:
            errors.append(
                f"blueprint has {len(blueprint.children)} children; limit is {self.settings.max_children}"
            )
        forbidden = {node.fingerprint}
        forbidden.update(ancestor.fingerprint for ancestor in self._graph.ancestors(node.node_id))
        seen: set[str] = set()
        proposed: list[tuple[LeanStatement, LeanStatement, str]] = []
        declarations: list[dict[str, object]] = []
        for child in blueprint.children[: self.settings.max_children]:
            source = LeanStatement(
                name=child.label,
                statement=child.statement,
                imports=node.goal.imports,
                namespace=node.goal.namespace,
            )
            elaborated, statement_check = self.verifier.elaborate_statement(source)
            self._remember_verification(statement_check)
            if not statement_check.accepted:
                errors.append(
                    f"child `{child.label}` does not elaborate: {statement_check.reason}"
                )
                if statement_check.exit_code == 127:
                    self._unavailable = True
            fingerprint = statement_fingerprint(
                source,
                node.environment_fingerprint,
                elaborated_statement=elaborated,
            )
            canonical = source.model_copy(
                update={"name": canonical_lemma_name(child.label, fingerprint)}
            )
            if fingerprint in forbidden:
                errors.append(f"child `{child.label}` repeats the parent or an ancestor proposition")
            if fingerprint in seen:
                errors.append(f"child `{child.label}` duplicates another proposed child")
            seen.add(fingerprint)
            proposed.append((source, canonical, elaborated))
            declarations.append(
                {
                    "name": canonical.name,
                    "statement": canonical.statement,
                    "required": child.required,
                    "rationale": child.rationale,
                }
            )
        if not any(bool(item["required"]) for item in declarations):
            errors.append("decomposition has no required child")
        return errors, proposed, declarations

    def _context(
        self,
        node_id: str,
        *,
        informal_queries: Sequence[str] = (),
        diagnostics: Sequence[LeanDiagnostic] = (),
    ) -> ProofContext:
        return self._reader.build(
            node_id,
            user_context=self._user_context,
            informal_queries=informal_queries,
            diagnostics=diagnostics,
            remaining_nodes=self.settings.max_nodes - self._graph.node_count(),
            remaining_seconds=self._budget.seconds_left(),
        )

    def _materialize_and_verify(
        self, root: OrNode, target: LeanStatement
    ) -> tuple[ArtifactRef, VerificationResult] | None:
        try:
            rendered = self._materializer.final_module(root.node_id, target)
        except MaterializationError as exc:
            self._operational_error = str(exc)
            return None
        rel_file = f"TCSResearch/Generated/Final_{self._run_id}.lean"
        verification = self.verifier.check_direct(
            rendered.code, rel_file=rel_file, target_name=target.name
        )
        self._remember_verification(verification)
        if not verification.accepted:
            self._operational_error = verification.reason
            return None
        final_ref = self.store.artifact_ref(verification.source_path)
        self._artifact_paths.add(final_ref.path)
        return final_ref, verification

    # ------------------------------------------------------------------
    # Durable attempt/artifact bookkeeping and result conversion
    # ------------------------------------------------------------------
    def _write_json_artifact(self, node_id: str, label: str, value: object) -> ArtifactRef:
        artifact_id = new_id("artifact")
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:100]
        path = f"LeanProject/LEAP/artifacts/{node_id}/{artifact_id}_{safe_label}.json"
        ref = self.store.write_json(path, value)
        self._artifact_paths.add(ref.path)
        return ref

    def _record_attempt(
        self,
        or_id: str,
        *,
        attempt_id: str,
        mode: str,
        outcome: str,
        digest: str = "",
        artifact_path: str = "",
        diagnostics: Sequence[LeanDiagnostic] = (),
        retrieval: Sequence[RetrievalHit] = (),
        parent_attempt_id: str | None = None,
        duration: float = 0.0,
        note: str = "",
    ) -> None:
        self._graph.record_attempt(
            AttemptRecord(
                attempt_id=attempt_id,
                or_id=or_id,
                mode=mode,
                # ProofGraph allocates the authoritative ordinal inside its write transaction.
                ordinal=0,
                outcome=outcome,
                candidate_sha256=digest,
                candidate_artifact_path=artifact_path,
                diagnostics=list(diagnostics),
                retrieval=list(retrieval),
                parent_attempt_id=parent_attempt_id,
                duration_seconds=round(duration, 4),
                note=note[:3000],
            )
        )
        if artifact_path:
            self._artifact_paths.add(artifact_path)

    def _record_control_failure(self, or_id: str, mode: str, note: str) -> None:
        self._record_attempt(
            or_id,
            attempt_id=new_id("leap_attempt"),
            mode=mode,
            outcome="operational_failure",
            note=note,
        )

    def _remember_verification(self, verification: VerificationResult) -> None:
        self._verifications.append(verification)
        for path in [verification.source_path, verification.log_path]:
            if path:
                self._artifact_paths.add(path)

    def _partial_status(self, root: OrNode) -> str:
        assert self.graph is not None
        if self.graph.decompositions(root.node_id):
            return "partial"
        return "exhausted"

    def _result(
        self,
        target: LeanStatement,
        root: OrNode,
        *,
        status: str,
        proved_artifacts: Sequence[ArtifactRef] = (),
    ) -> TheoremProverResult:
        assert self.graph is not None
        open_nodes = self.graph.open_nodes(reachable_from=root.node_id)[:100]
        summary = {
            "run_id": self._run_id,
            "root_or_id": root.node_id,
            "root_status": self.graph.get_or(root.node_id).status.value,
            "or_node_count": self.graph.node_count(),
            "proved_or_count": self.graph.proved_count(),
            "open_reachable_count": len(open_nodes),
            "stop_reason": self._budget.stop_reason,
            "operational_error": self._operational_error,
            "persistent_graph": ProofGraph.DB_PATH,
        }
        next_steps: list[str] = []
        if status == "unavailable":
            next_steps.append("Install/configure the pinned Lean/Lake toolchain and resume this run.")
        elif status == "partial":
            next_steps.append("Resume the same theorem; LEAP will continue from the persistent DAG.")
        elif status == "exhausted":
            next_steps.append(
                "Inspect recent LEAP attempts/reviewer feedback, improve retrieval or budgets, and resume."
            )
        elif status == "proved":
            next_steps.append("Use the final batch-compiled no-placeholder Lean module as evidence.")
        self.graph.checkpoint()
        artifact_refs = [
            self.store.artifact_ref(path)
            for path in sorted(self._artifact_paths)
            if self.store.exists(path)
        ]
        return TheoremProverResult(
            status=status,
            root_goal=target,
            proved_artifacts=list(proved_artifacts),
            artifact_refs=artifact_refs,
            open_goals=[
                ProofGoal(
                    goal_id=node.node_id,
                    lean_statement=node.goal,
                    status="open" if node.status != OrStatus.abandoned else "failed",
                )
                for node in open_nodes
            ],
            proof_dag_summary=json.dumps(summary, indent=2, sort_keys=True),
            compiler_logs=[self._compiler_log(item) for item in self._verifications],
            recommended_next_steps=next_steps,
        )

    def _early_result(
        self, target: LeanStatement, *, status: str, summary: str
    ) -> TheoremProverResult:
        return TheoremProverResult(
            status=status,
            root_goal=target,
            artifact_refs=[
                self.store.artifact_ref(path)
                for path in sorted(self._artifact_paths)
                if self.store.exists(path)
            ],
            proof_dag_summary=summary,
            compiler_logs=[self._compiler_log(item) for item in self._verifications],
            recommended_next_steps=[
                "Fix the Lean project/toolchain or root proposition, then retry registration."
            ],
        )

    def _compiler_log(self, result: VerificationResult) -> LeanCompilerLog:
        source = result.source_path
        rel = source.removeprefix("LeanProject/") if source else "<generated>"
        return LeanCompilerLog(
            command=["lake", "env", "lean", rel],
            cwd=str(self.verifier.project_root),
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr or result.reason,
            success=result.accepted,
            artifact_ref=(
                self.store.artifact_ref(result.log_path)
                if result.log_path and self.store.exists(result.log_path)
                else None
            ),
        )

    @property
    def _graph(self) -> ProofGraph:
        assert self.graph is not None
        return self.graph

    @property
    def _reader(self) -> StateReader:
        assert self.reader is not None
        return self.reader

    @property
    def _materializer(self) -> ProofMaterializer:
        assert self.materializer is not None
        return self.materializer


def _sketch_usage_errors(
    parent_proof: str, child_declarations: Sequence[dict[str, object]]
) -> list[str]:
    errors: list[str] = []
    for child in child_declarations:
        name = str(child["name"])
        used = re.search(
            rf"(?<![A-Za-z0-9_']){re.escape(name)}(?![A-Za-z0-9_'])", parent_proof
        )
        if bool(child["required"]) and used is None:
            errors.append(f"required child `{name}` is not referenced by the parent proof")
        if not bool(child["required"]) and used is not None:
            errors.append(f"anticipatory child `{name}` is used and therefore must be required")
    return errors


def deterministic_proof_candidates() -> tuple[tuple[str, str], ...]:
    """Small, stable proof portfolio tried without model calls.

    ``rfl`` covers definitional equalities, ``simp`` uses only simplification lemmas available from
    the goal's pinned imports, and ``decide`` handles closed decidable propositions (including
    quantification over finite types when Lean can synthesize the instance).
    """
    return (
        ("rfl", "by\n  rfl"),
        ("simp", "by\n  simp"),
        ("decide", "by\n  decide"),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(normalize_lean_source(value).encode("utf-8")).hexdigest()
