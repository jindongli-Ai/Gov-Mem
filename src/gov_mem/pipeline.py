from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import os

from gov_mem.answering.answer_agent import AnsweringAgent
from gov_mem.data.adapters import DatasetBundle, build_dataset_adapter
from gov_mem.data.schema import CaseResult, MemoryInstance, QueryPlan, ReasoningState, RetrievedEvidence
from gov_mem.evaluation.evaluator import Evaluator
from gov_mem.eval.benchmark_official import run_official_scorer
from gov_mem.experience.experience_bank import ExperienceBank
from gov_mem.governance_runtime.action_predictor import GovernedActionPredictor
from gov_mem.governance_runtime.leakage_guard import contains_hidden_eval_fields
from gov_mem.governance_runtime.leakage_guard import runtime_instance_view
from gov_mem.llm.client import LLMClient, LLMConfig, YUNWU_PROVIDER_NAMES, is_real_llm_enabled
from gov_mem.llm.model_registry import build_resolved_llm_settings, resolve_llm_model
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.memory.ingestion import MemoryIngestionAgent
from gov_mem.memory.symbolic_store import SymbolicMemoryStore
from gov_mem.planning.query_planner import QueryUnderstandingAgent
from gov_mem.reasoning.reasoner import SymbolicReasoner
from gov_mem.retrieval.dense_retriever import DenseRetriever
from gov_mem.retrieval.hybrid_retriever import HybridRetriever
from gov_mem.retrieval.symbolic_retriever import SymbolicRetriever
from gov_mem.skills.registry import SkillRegistry
from gov_mem.skills.updater import SkillUpdater
from gov_mem.utils.io import append_jsonl, ensure_dir, read_jsonl, write_json, write_jsonl
from gov_mem.utils.logging import setup_logger


FINAL_STAGES = {"all", "evaluate"}


class GovMemRunner:
    def __init__(
        self,
        *,
        dataset_name: str,
        data_path: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
        stage: str,
        max_instances: int | None = None,
        start_index: int = 0,
        checkpoint_ids: list[str] | None = None,
        resume: bool = False,
        run_official_benchmark_eval: bool = True,
        experiment_mode: str = "govmem_structured_old",
    ):
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.config = config
        self.stage = stage
        self.max_instances = max_instances
        self.start_index = start_index
        self.checkpoint_ids = checkpoint_ids or []
        self.resume = resume
        self.run_official_benchmark_eval = run_official_benchmark_eval
        self.experiment_mode = experiment_mode

        log_dir = ensure_dir(self.output_dir / "logs")
        self.logger = setup_logger("govmem", log_dir / "run.log")
        self.skill_registry = SkillRegistry(self.output_dir / "skills_runtime")
        if self.resume:
            self._validate_resume_metadata()
        self._configure_provider_env()

        llm_cfg = LLMConfig(
            provider=str(self.config["llm"]["provider"]),
            temperature=float(self.config["llm"]["temperature"]),
            max_retries=int(self.config["llm"]["max_retries"]),
            api_base=self.config["llm"].get("api_base"),
            api_key_env=self.config["llm"].get("api_key_env"),
            allow_fallback=bool(self.config["llm"].get("allow_fallback", True)),
            request_timeout=int(self.config["llm"].get("request_timeout", 120)),
            retry_backoff_seconds=float(self.config["llm"].get("retry_backoff_seconds", 3.0)),
            max_retry_backoff_seconds=float(self.config["llm"].get("max_retry_backoff_seconds", 30.0)),
        )
        embed_cfg = LLMConfig(
            provider=str(self.config["embedding"]["provider"]),
            temperature=0.0,
            max_retries=int(self.config["llm"]["max_retries"]),
            api_base=self.config["embedding"].get("api_base") or self.config["llm"].get("api_base"),
            api_key_env=self.config["embedding"].get("api_key_env") or self.config["llm"].get("api_key_env"),
            allow_fallback=bool(self.config["embedding"].get("allow_fallback", True)),
            request_timeout=int(self.config["embedding"].get("request_timeout", self.config["llm"].get("request_timeout", 120))),
            retry_backoff_seconds=float(self.config["embedding"].get("retry_backoff_seconds", self.config["llm"].get("retry_backoff_seconds", 3.0))),
            max_retry_backoff_seconds=float(self.config["embedding"].get("max_retry_backoff_seconds", self.config["llm"].get("max_retry_backoff_seconds", 30.0))),
            cache_dir=str(self.config["embedding"].get("cache_dir")) if self.config["embedding"].get("cache_dir") else None,
        )
        self.llm_client = LLMClient(llm_cfg)
        self.embedding_client = LLMClient(embed_cfg)
        self.resolved_llm_settings = build_resolved_llm_settings(self.config)
        self._log_llm_mode()
        self._save_run_metadata()

        experience_path = self.output_dir / "experience" / self.dataset_name / "experience_bank.jsonl"
        self.experience_bank = (
            ExperienceBank(experience_path)
            if self.config["self_evolving"]["enable_experience"]
            else None
        )
        self.skill_updater = SkillUpdater(
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "query_planning"),
            skill_registry=self.skill_registry,
            enable_skill_update=bool(self.config["self_evolving"]["enable_skill_update"]),
            update_every_n_failures=int(self.config["self_evolving"]["update_every_n_failures"]),
        )
        self._backbone = None

    def _configure_provider_env(self) -> None:
        provider = str(self.config["llm"]["provider"]).strip().lower()
        if provider not in YUNWU_PROVIDER_NAMES:
            return
        # An explicit pool is used for controlled parallel/resume runs. Only
        # fall back to the local README when the caller did not provide one.
        configured_pool = [
            value.strip()
            for value in os.environ.get("YUNWU_API_KEYS", "").split(",")
            if value.strip()
        ]
        if configured_pool:
            keys = list(dict.fromkeys(configured_pool))
        else:
            readme = self.output_dir.parents[0] / "README_API_Yunwu.md"
            if not readme.exists():
                readme = Path("README_API_Yunwu.md")
            if not readme.exists():
                return
            content = readme.read_text(encoding="utf-8")
            keys = []
            for line in content.splitlines():
                line = line.strip()
                if "sk-" in line:
                    key = line.split("`")[1] if "`" in line else line
                    if key.startswith("sk-"):
                        keys.append(key)
        if keys:
            # Expose the same pool to the client so transient failures can
            # rotate keys. Keep an explicitly supplied key authoritative;
            # direct domain runs otherwise receive a stable per-output-dir
            # starting key to avoid concentrating parallel jobs on index 0.
            os.environ["YUNWU_API_KEYS"] = ",".join(dict.fromkeys(keys))
            try:
                requested_index = int(os.environ.get("YUNWU_API_KEY_INDEX", "0"))
            except ValueError:
                requested_index = 0
            current_key = os.environ.get("YUNWU_API_KEY")
            if current_key in keys:
                index = keys.index(current_key)
            elif "YUNWU_API_KEY_INDEX" in os.environ:
                index = requested_index % len(keys)
            else:
                digest = sha256(str(self.output_dir.resolve()).encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % len(keys)
            # Resolve the selected pool entry into the env var consumed by
            # LLMClient and by the official scorer subprocess.
            os.environ["YUNWU_API_KEY"] = keys[index]
            os.environ.setdefault("YUNWU_BASE_URL", str(self.config["llm"].get("api_base") or "https://yunwu.ai/v1"))
            self.logger.info("[Gov-Mem] Yunwu key pool selected index=%d pool_size=%d", index, len(keys))

    def _log_llm_mode(self) -> None:
        if is_real_llm_enabled(self.llm_client.config):
            self.logger.info(
                "[Gov-Mem] Real LLM mode enabled: provider=%s, model=%s",
                self.config["llm"]["provider"],
                resolve_llm_model(self.config, "answering"),
            )
        else:
            self.logger.warning(
                "[Gov-Mem Warning] No valid LLM API key detected. Falling back to heuristic mode. Accuracy may be invalid."
            )

    def _save_run_metadata(self) -> None:
        write_json(
            self.output_dir / "run_metadata.json",
            {
                "dataset_name": self.dataset_name,
                "data_path": str(self.data_path),
                "output_dir": str(self.output_dir),
                "stage": self.stage,
                "max_instances": self.max_instances,
                "start_index": self.start_index,
                "checkpoint_ids": self.checkpoint_ids,
                "resume": self.resume,
                "experiment_mode": self.experiment_mode,
                "runtime_source_fingerprint": self._runtime_source_fingerprint(),
                "runtime_capabilities": {
                    "llm_available": self.llm_client.is_available(),
                    "embedding_available": self.embedding_client.is_available(),
                    "semantic_contract_required": True,
                },
                "resolved_llm_settings": self.resolved_llm_settings,
                "llm_telemetry": self.llm_client.telemetry_snapshot(),
                "embedding_telemetry": self.embedding_client.telemetry_snapshot(),
                "config_snapshot": self.config,
            },
        )

    def load_dataset(self) -> DatasetBundle:
        adapter = build_dataset_adapter(
            self.dataset_name,
            use_asking_user_id=bool(self.config["pipeline"]["use_asking_user_id"]),
        )
        bundle = adapter.load(
            self.data_path,
            max_instances=self.max_instances,
            start_index=self.start_index,
            checkpoint_ids=set(self.checkpoint_ids) if self.checkpoint_ids else None,
        )
        for instance in bundle.instances:
            if contains_hidden_eval_fields(instance.metadata.get("observable_metadata", {})):
                raise ValueError(
                    f"Observable metadata for instance {instance.instance_id} contains hidden evaluation fields."
                )
        self.logger.info(
            "Loaded dataset=%s instances=%d metadata=%s",
            bundle.dataset_name,
            len(bundle.instances),
            bundle.metadata,
        )
        return bundle

    def run(self) -> dict[str, Any]:
        bundle = self.load_dataset()
        official_predictions_path = self.output_dir / "predictions" / bundle.dataset_name / "predictions.jsonl"
        completed_ids = self._load_completed_prediction_ids(official_predictions_path) if self.resume else set()
        expected_ids = {str(instance.instance_id) for instance in bundle.instances}
        unexpected = completed_ids - expected_ids
        if unexpected:
            raise ValueError(f"Resume predictions contain checkpoints outside the requested manifest: {sorted(unexpected)[:5]}")
        if official_predictions_path.exists() and not self.resume:
            official_predictions_path.unlink()

        evaluator = Evaluator(
            output_dir=self.output_dir,
            dataset_name=bundle.dataset_name,
            experience_bank=self.experience_bank,
            initial_case_results=self._load_resume_case_results(
                dataset_name=bundle.dataset_name,
                completed_ids=completed_ids,
            ) if self.resume else None,
        )

        for instance in bundle.instances:
            if str(instance.instance_id) in completed_ids:
                self.logger.info("Skipping completed checkpoint=%s under strict resume", instance.instance_id)
                continue
            try:
                if self.experiment_mode == "govmem_structured_old":
                    self._process_instance(instance, bundle.dataset_name, official_predictions_path, evaluator)
                else:
                    self._process_instance_backbone(instance, bundle.dataset_name, official_predictions_path, evaluator)
            except Exception as exc:
                self.logger.exception("Instance %s failed: %s", instance.instance_id, exc)

        expected_checkpoint_ids = {str(instance.instance_id) for instance in bundle.instances}
        completed_checkpoint_ids = self._load_completed_prediction_ids(official_predictions_path)
        missing_checkpoint_ids = sorted(expected_checkpoint_ids - completed_checkpoint_ids)
        if missing_checkpoint_ids:
            # A partial prediction set must never be silently scored as a
            # completed suite.  The caller can rerun with --resume, which
            # processes only these IDs after validating run identity.
            raise RuntimeError(
                "Run incomplete; use strict --resume after transient failures. Missing checkpoints: "
                + ", ".join(missing_checkpoint_ids[:10])
            )

        metrics = {}
        if self.stage in FINAL_STAGES:
            metrics = evaluator.finalize()
            self.logger.info("Local evaluation metrics: %s", metrics)
            if (
                bool(bundle.metadata.get("supports_official_benchmark", False))
                and self.run_official_benchmark_eval
                and bundle.metadata.get("domain")
            ):
                self._bundle_metadata = dict(bundle.metadata)
                self._run_official_benchmark_eval(
                    domain=str(bundle.metadata["domain"]),
                    predictions_path=official_predictions_path,
                )
                self._attach_official_scores_to_debug_cases(domain=str(bundle.metadata["domain"]))
        # Persist endpoint timing/retry totals after all model calls complete.
        self._save_run_metadata()
        return metrics

    def _validate_resume_metadata(self) -> None:
        """Reject resume when the old output cannot prove execution equivalence."""
        path = self.output_dir / "run_metadata.json"
        if not path.exists():
            raise ValueError("--resume requires an existing run_metadata.json")
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError("--resume could not read existing run_metadata.json") from exc
        required = {
            "dataset_name": self.dataset_name,
            "experiment_mode": self.experiment_mode,
            "config_snapshot": self.config,
            "runtime_source_fingerprint": self._runtime_source_fingerprint(),
        }
        for key, expected in required.items():
            if previous.get(key) != expected:
                raise ValueError(f"--resume refused: existing run metadata differs for {key}")
        previous_ids = [str(value) for value in list(previous.get("checkpoint_ids") or [])]
        if previous_ids != [str(value) for value in self.checkpoint_ids]:
            raise ValueError("--resume refused: checkpoint manifest differs from existing run")

    @staticmethod
    def _runtime_source_fingerprint() -> str:
        """Fingerprint runtime code so interrupted outputs cannot mix revisions."""
        root = Path(__file__).resolve().parents[2]
        paths = [root / "run_govmem.py", *sorted((root / "src" / "gov_mem").rglob("*.py"))]
        digest = sha256()
        for path in paths:
            if not path.exists():
                continue
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _load_completed_prediction_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        rows = read_jsonl(path)
        ids = [str(row.get("checkpoint_id") or "").strip() for row in rows]
        if not ids or any(not value for value in ids):
            raise ValueError("--resume refused: predictions.jsonl contains a missing checkpoint_id")
        duplicates = {value for value in ids if ids.count(value) > 1}
        if duplicates:
            raise ValueError(f"--resume refused: duplicate checkpoint predictions: {sorted(duplicates)[:5]}")
        return set(ids)

    def _load_resume_case_results(self, *, dataset_name: str, completed_ids: set[str]) -> list[CaseResult]:
        path = self.output_dir / "eval" / dataset_name / "case_results.jsonl"
        if not path.exists():
            return []
        rows = read_jsonl(path)
        results: list[CaseResult] = []
        seen: set[str] = set()
        for row in rows:
            instance_id = str(row.get("instance_id") or "")
            if not instance_id or instance_id not in completed_ids or instance_id in seen:
                continue
            seen.add(instance_id)
            results.append(CaseResult(
                instance_id=instance_id,
                question=str(row.get("question") or ""),
                gold_answer=row.get("gold_answer"),
                prediction=row.get("prediction"),
                correct=bool(row.get("correct")),
                query_type=row.get("query_type"),
                used_memory_ids=list(row.get("used_memory_ids") or []),
                failure_type=row.get("failure_type"),
                domain=row.get("domain"),
                metadata=dict(row.get("metadata") or {}),
            ))
        missing = completed_ids - seen
        if missing:
            self.logger.warning(
                "resume_case_results_missing=%d; official scorer remains complete but local metrics omit those prior cases",
                len(missing),
            )
        return results

    def _get_backbone(self):
        if self._backbone is not None:
            return self._backbone
        kwargs = {
            "llm_client": self.llm_client,
            "embedding_client": self.embedding_client,
            "config": self.config,
            "output_dir": self.output_dir,
            "dataset_name": self.dataset_name,
        }
        if self.experiment_mode == "stateful_policy_reasoning":
            from gov_mem.backbones.stateful_policy import StatefulPolicyBackbone

            self._backbone = StatefulPolicyBackbone(**kwargs)
        elif self.experiment_mode in {"rag_naive", "rag_naive_v3_typed_rerank"}:
            from gov_mem.backbones.rag_naive import RAGNaiveBackbone

            self._backbone = RAGNaiveBackbone(**kwargs)
        elif self.experiment_mode == "rag_policy":
            from gov_mem.backbones.rag_policy import RAGPolicyBackbone

            self._backbone = RAGPolicyBackbone(**kwargs)
        elif self.experiment_mode == "rag_policy_amem":
            from gov_mem.backbones.rag_policy_amem import RAGPolicyAMemBackbone

            self._backbone = RAGPolicyAMemBackbone(**kwargs)
        elif self.experiment_mode == "govmem_rag_policy_incremental":
            from gov_mem.backbones.govmem_incremental import GovMemIncrementalBackbone

            self._backbone = GovMemIncrementalBackbone(**kwargs)
        else:
            raise ValueError(f"Unsupported experiment_mode: {self.experiment_mode}")
        return self._backbone

    def _process_instance(
        self,
        instance: MemoryInstance,
        dataset_name: str,
        official_predictions_path: Path,
        evaluator: Evaluator,
    ) -> None:
        self.logger.info("Processing instance=%s", instance.instance_id)

        ingestion_agent = MemoryIngestionAgent(
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "memory_ingestion"),
            skill_text=self.skill_registry.get_stage_text("ingestion"),
        )
        memory_items = ingestion_agent.ingest(instance)
        self._save_memory_items(dataset_name, instance.instance_id, memory_items)
        if self.stage == "ingest":
            return

        dense_index = DenseMemoryIndex.build(
            items=memory_items,
            llm_client=self.embedding_client,
            embedding_model=self.config["embedding"]["model"],
        )
        symbolic_store = SymbolicMemoryStore(memory_items)

        planner = QueryUnderstandingAgent(
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "query_planning"),
            skill_text=self.skill_registry.get_stage_text("query_planning"),
            use_asking_user_id=bool(self.config["pipeline"]["use_asking_user_id"]),
            experience_bank=self.experience_bank,
        )
        if bool(self.config.get("ablation", {}).get("use_query_planner", True)):
            plan = planner.plan(instance)
        else:
            plan = planner._heuristic_plan(
                instance,
                asking_user_id=instance.asking_user_id if bool(self.config["pipeline"]["use_asking_user_id"]) else None,
            )
        self._save_query_plan(dataset_name, instance.instance_id, plan)

        hybrid_retriever = HybridRetriever(
            dense_retriever=DenseRetriever(
                llm_client=self.embedding_client,
                embedding_model=self.config["embedding"]["model"],
                top_k=int(self.config["retrieval"]["dense_top_k"]),
            ),
            symbolic_retriever=SymbolicRetriever(
                top_k=int(self.config["retrieval"]["symbolic_top_k"])
            ),
            final_top_k=int(self.config["retrieval"]["final_top_k"]),
        )
        retrieval_result = hybrid_retriever.retrieve(
            plan=plan,
            dense_index=dense_index,
            symbolic_store=symbolic_store,
            memory_by_id={item.memory_id: item for item in memory_items},
            requester=instance.asking_user_id,
            config={
                **self.config,
                "_runtime_instance_metadata": {
                    "requester": instance.metadata.get("requester"),
                    "observable": instance.metadata.get("observable"),
                },
            },
        )
        evidence = retrieval_result["retrieved_after_privacy_filter"]
        self._save_retrieval(dataset_name, instance.instance_id, retrieval_result)
        if self.stage == "retrieve":
            return

        if bool(self.config.get("ablation", {}).get("use_logical_reasoning", True)):
            reasoning_state = SymbolicReasoner().reason(
                plan=plan,
                evidence=evidence,
                config={
                    **self.config,
                    "_runtime_instance_metadata": {
                        "question": instance.question,
                        "requester": instance.metadata.get("requester"),
                        "observable": instance.metadata.get("observable"),
                    },
                },
            )
        else:
            reasoning_state = ReasoningState(
                selected_evidence=evidence,
                reasoning_trace=["Logical reasoning disabled by ablation."],
                conflicts=[],
                conclusion_hint="Logical reasoning ablation path.",
                selected_frames=[],
                current_state_ledger={},
                required_slot_plan={},
                slot_coverage={},
            )
        self._save_reasoning(dataset_name, instance.instance_id, reasoning_state)

        action_decision = None
        runtime_profile = dict(instance.metadata.get("runtime_profile") or {})
        if bool(runtime_profile.get("use_action_decision", False)) and bool(self.config.get("ablation", {}).get("use_action_predictor", True)):
            action_predictor = GovernedActionPredictor(
                llm_client=self.llm_client,
                model_name=resolve_llm_model(self.config, "action_decision"),
                experience_bank=self.experience_bank,
            )
            action_decision = action_predictor.decide(
                instance=instance,
                plan=plan,
                evidence=evidence,
            )

        answer_agent = AnsweringAgent(
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "answering"),
            skill_text=self.skill_registry.get_stage_text("answering"),
            use_asking_user_id=bool(self.config["pipeline"]["use_asking_user_id"]),
            experience_bank=self.experience_bank,
            config=self.config,
        )
        answer_result = (
            answer_agent.answer_with_action(
                instance=instance,
                reasoning_state=reasoning_state,
                action_decision=action_decision,
                query_type=plan.query_type,
            )
            if action_decision is not None
            else answer_agent.answer(instance=instance, reasoning_state=reasoning_state, query_type=plan.query_type)
        )
        self._save_prediction(dataset_name, instance.instance_id, answer_result)
        self._append_official_prediction(official_predictions_path, instance, answer_result)
        if self.stage == "answer":
            return

        case_result = evaluator.evaluate_case(
            instance=instance,
            answer_result=answer_result,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
        )
        self._save_debug_case(
            dataset_name=dataset_name,
            instance=instance,
            query_plan=plan,
            memory_items=memory_items,
            retrieval_result=retrieval_result,
            action_decision=action_decision,
            answer_result=answer_result,
            case_result=case_result,
        )
        if self.experience_bank is not None:
            for stage_name in ["query_planning", "action_decision", "answering"]:
                self.skill_updater.maybe_update(stage=stage_name, experience_bank=self.experience_bank)

    def _process_instance_backbone(
        self,
        instance: MemoryInstance,
        dataset_name: str,
        official_predictions_path: Path,
        evaluator: Evaluator,
    ) -> None:
        self.logger.info("Processing instance=%s mode=%s", instance.instance_id, self.experiment_mode)
        runtime_instance = runtime_instance_view(instance)
        result = self._get_backbone().run_instance(runtime_instance)
        self._save_query_plan(dataset_name, instance.instance_id, result.query_plan)
        self._save_retrieval(dataset_name, instance.instance_id, result.retrieval_result)
        self._save_reasoning(dataset_name, instance.instance_id, result.reasoning_state)
        self._save_prediction(dataset_name, instance.instance_id, result.answer_result)
        self._append_official_prediction(official_predictions_path, instance, result.answer_result)
        case_result = evaluator.evaluate_case(
            instance=instance,
            answer_result=result.answer_result,
            reasoning_state=result.reasoning_state,
            action_decision=result.action_decision,
        )
        self._save_debug_case(
            dataset_name=dataset_name,
            instance=instance,
            query_plan=result.query_plan,
            memory_items=[],
            retrieval_result=result.retrieval_result,
            action_decision=result.action_decision,
            answer_result=result.answer_result,
            case_result=case_result,
            extra_debug=result.debug_payload,
        )

    def _save_memory_items(self, dataset_name: str, instance_id: str, memory_items) -> None:
        path = self.output_dir / "memory" / dataset_name / instance_id / "memory_items.jsonl"
        write_jsonl(path, memory_items)

    def _save_query_plan(self, dataset_name: str, instance_id: str, plan: QueryPlan) -> None:
        path = self.output_dir / "query_plan" / dataset_name / f"{instance_id}.json"
        write_json(path, plan)

    def _save_retrieval(self, dataset_name: str, instance_id: str, evidence) -> None:
        path = self.output_dir / "retrieval" / dataset_name / f"{instance_id}.json"
        write_json(path, evidence)

    def _save_reasoning(self, dataset_name: str, instance_id: str, reasoning_state: ReasoningState) -> None:
        path = self.output_dir / "reasoning" / dataset_name / f"{instance_id}.json"
        write_json(path, reasoning_state)

    def _save_prediction(self, dataset_name: str, instance_id: str, answer_result) -> None:
        path = self.output_dir / "predictions" / dataset_name / f"{instance_id}.json"
        write_json(path, answer_result)

    def _append_official_prediction(self, path: Path, instance: MemoryInstance, answer_result) -> None:
        # Official benchmark scoring is text-based. Exporting debug-heavy structured
        # payloads can create false privacy leaks because the scorer scans both
        # `answer` and `answer_structured` as flat text.
        row = {
            "checkpoint_id": instance.instance_id,
            "action": answer_result.action,
            "answer": answer_result.answer_text,
            "answer_structured": {},
            "used_record_ids": answer_result.used_memory_ids,
        }
        append_jsonl(path, row)

    def _save_debug_case(
        self,
        *,
        dataset_name: str,
        instance: MemoryInstance,
        query_plan,
        memory_items,
        retrieval_result,
        action_decision,
        answer_result,
        case_result,
        extra_debug: dict[str, Any] | None = None,
    ) -> None:
        debug_dir = ensure_dir(self.output_dir / "debug_cases" / dataset_name)
        path = debug_dir / f"{instance.instance_id}.json"
        extra_debug = dict(extra_debug or {})
        write_json(
            path,
            {
                "instance_id": instance.instance_id,
                "experiment_mode": self.experiment_mode,
                "resolved_llm_settings": self.resolved_llm_settings,
                "question": instance.question,
                "asking_user_id": instance.asking_user_id,
                "requester": instance.metadata.get("requester"),
                "query_plan": query_plan,
                "memory_items": memory_items,
                "retrieved_before_privacy_filter": retrieval_result.get("retrieved_before_privacy_filter", []),
                "retrieved_after_privacy_filter": retrieval_result.get("retrieved_after_privacy_filter", []),
                "filtered_evidence": retrieval_result.get("filtered_evidence", []),
                "action_decision": action_decision,
                "final_prediction": answer_result,
                "slot_audit": (answer_result.raw_response or {}).get("slot_audit"),
                "rendered_answer_verifier": (answer_result.raw_response or {}).get("rendered_answer_verifier"),
                "action_correction_trace": (answer_result.raw_response or {}).get("action_correction_trace", []),
                "selected_frame_typed_slots": (answer_result.raw_response or {}).get("selected_frame_typed_slots", []),
                "event_ledger_summary": (answer_result.raw_response or {}).get("event_ledger_summary", {}),
                "answer_need_spec": (answer_result.raw_response or {}).get("answer_need_spec"),
                "utility_records": (answer_result.raw_response or {}).get("utility_records", []),
                "packed_utility_evidence": (answer_result.raw_response or {}).get("packed_utility_evidence"),
                "canonical_answer": (answer_result.raw_response or {}).get("canonical_answer"),
                "coverage_verification": (answer_result.raw_response or {}).get("coverage_verification"),
                "used_renderer": (answer_result.raw_response or {}).get("used_renderer"),
                "renderer_repair_trace": (answer_result.raw_response or {}).get("renderer_repair_trace", []),
                "renderer_arbitration": (answer_result.raw_response or {}).get("renderer_arbitration"),
                "pcur_preview": (answer_result.raw_response or {}).get("pcur_preview"),
                "v11_fallback_preview": (answer_result.raw_response or {}).get("v11_fallback_preview"),
                "correct": case_result.correct,
                "failure_type": case_result.failure_type,
                **extra_debug,
            },
        )

    def _run_official_benchmark_eval(self, *, domain: str, predictions_path: Path) -> None:
        try:
            out_dir = self.output_dir / "official_eval" / self.dataset_name / domain
            judge_config = dict((self.config.get("evaluation") or {}).get("official_judge") or {})
            # GateMem paper metrics are defined by this judge, independently
            # from the model used inside the evaluated memory backbone.
            judge_settings = {
                "use_llm_judge": bool(judge_config.get("enabled", True)),
                "provider": str(judge_config.get("provider") or "yunwu"),
                "model": str(judge_config.get("model") or "gpt-4o"),
                "temperature": float(judge_config.get("temperature", 0.0)),
                "max_output_tokens": int(judge_config.get("max_output_tokens", 4096)),
                "api_base": str(judge_config.get("api_base") or self.config["llm"].get("api_base") or ""),
                "api_key_env": str(judge_config.get("api_key_env") or self.config["llm"].get("api_key_env") or ""),
                "concurrency": int(judge_config.get("concurrency", 4)),
            }
            write_json(out_dir / "judge_config.json", judge_settings)
            run_official_scorer(
                domain=domain,
                data_dir=Path(str((getattr(self, "_bundle_metadata", {}) or {}).get("data_dir"))),
                predictions_path=predictions_path,
                out_dir=out_dir,
                use_llm_judge=judge_settings["use_llm_judge"],
                judge_provider=judge_settings["provider"],
                judge_model=judge_settings["model"],
                judge_temperature=judge_settings["temperature"],
                judge_max_output_tokens=judge_settings["max_output_tokens"],
                judge_api_base=judge_settings["api_base"] or None,
                judge_api_key_env=judge_settings["api_key_env"] or None,
                judge_concurrency=judge_settings["concurrency"],
            )
            self._write_benchmark_paper_metrics(out_dir=out_dir)
            self.logger.info("Official benchmark scorer completed: %s", out_dir)
        except Exception as exc:
            self.logger.warning("Official benchmark scorer failed: %s", exc)

    def _write_benchmark_paper_metrics(self, *, out_dir: Path) -> None:
        summary_path = out_dir / "summary.json"
        if not summary_path.exists():
            return
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        utility = float(summary.get("utility_accuracy", 0.0))
        access = float(summary.get("privacy_leakage_rate", 0.0))
        forgetting = float(summary.get("deletion_leakage_rate", 0.0))
        paper_metrics = {
            "U": utility,
            "A": access,
            "F": forgetting,
            "OR": float(summary.get("over_refusal_rate", 0.0)),
            "MGS": utility * (1.0 - access) * (1.0 - forgetting),
            "source_summary": str(summary_path),
        }
        write_json(out_dir / "paper_metrics.json", paper_metrics)

    def _attach_official_scores_to_debug_cases(self, *, domain: str) -> None:
        import json

        scores_path = self.output_dir / "official_eval" / self.dataset_name / domain / "scores.jsonl"
        if not scores_path.exists():
            return
        scores = {}
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scores[str(row.get("checkpoint_id"))] = row
        debug_dir = self.output_dir / "debug_cases" / self.dataset_name
        if not debug_dir.exists():
            return
        for path in debug_dir.glob("*.json"):
            obj = json.loads(path.read_text(encoding="utf-8"))
            score = scores.get(str(obj.get("instance_id")))
            if score is None:
                continue
            obj["official_score"] = score
            write_json(path, obj)
