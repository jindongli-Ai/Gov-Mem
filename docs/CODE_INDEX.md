# Repository file index

The original runtime implementation is retained. `run_govmem.py` and `src/gov_mem/pipeline.py` select the current `govmem_v8_late_governance` backbone. Historical modules below remain only where imported by the original runner or compatibility paths; they must not be confused with the current V8 method.

The publication excludes experiment outputs, caches, API-key files, benchmark data, copied third-party packages, PDFs, internal handoffs and machine-specific launch scripts. Benchmark data/scoring dependencies are obtained separately.

## Root and configuration

| File | Purpose |
|---|---|
| `README.md` | Short setup and run instructions; under-review status. |
| `run_govmem.py` | Original CLI entry point. |
| `pyproject.toml` | Installation metadata, core dependencies and test settings. |
| `.gitignore` | Exclude credentials, caches, generated files and local environments. |
| `configs/govmem.yaml` | Default GPT-5.4-nano configuration; relative embedding cache, K=20. |
| `docs/METHOD.md` | Current method/code correspondence and evaluation definitions. |
| `docs/CODE_INDEX.md` | This complete file index. |

## Source folders

| Folder | Purpose |
|---|---|
| `src/gov_mem/answering/` | Answer generation helpers used by compatibility paths. |
| `src/gov_mem/backbones/` | Current V8 backbone and imported historical backbones/support routines. Start with govmem_v8_late_governance.py. |
| `src/gov_mem/data/` | Dataset adapters, checkpoint schemas and timestamp handling. |
| `src/gov_mem/eval/` | Official GateMem evaluation integration. |
| `src/gov_mem/evaluation/` | Local case evaluation and prompt-context audits. |
| `src/gov_mem/experience/` | Optional experience storage used by the runner; disabled in the default configuration. |
| `src/gov_mem/extraction/` | Source-grounded event extraction, schemas, validation and protocol parsers. |
| `src/gov_mem/governance_runtime/` | Claim reasoning, state projection, symbolic review, evidence release and compatibility governance helpers. |
| `src/gov_mem/graph/` | Graph structures used by imported compatibility helpers. |
| `src/gov_mem/llm/` | API client, JSON parsing and model-role configuration. |
| `src/gov_mem/memory/` | Dense memory index, governed slot graph and memory helpers. |
| `src/gov_mem/planning/` | Query planning and intent helpers. |
| `src/gov_mem/reasoning/` | Symbolic reasoning helpers imported by historical modes. |
| `src/gov_mem/retrieval/` | Dense, symbolic and hybrid retrieval helpers. |
| `src/gov_mem/skills/` | Runtime skill registry and optional updates; updates disabled by default. |
| `src/gov_mem/utils/` | Configuration, storage, serialization and logging. |

## Source files

Every runtime Python file below matches its original workspace source byte-for-byte. `__init__.py` files provide Python package boundaries.

| File | Description |
|---|---|
| `src/gov_mem/__init__.py` | Package initializer. |
| `src/gov_mem/answer_projection.py` | Field-level answer planning and evidence projection. |
| `src/gov_mem/answering/__init__.py` | Package initializer. |
| `src/gov_mem/answering/answer_agent.py` | Runtime module: answer agent. |
| `src/gov_mem/backbones/__init__.py` | Package initializer. |
| `src/gov_mem/backbones/action_arbitrator.py` | Runtime module: action arbitrator. |
| `src/gov_mem/backbones/action_constrained_realizer.py` | Runtime module: action constrained realizer. |
| `src/gov_mem/backbones/canonical_renderer.py` | Runtime module: canonical renderer. |
| `src/gov_mem/backbones/common.py` | Runtime module: common. |
| `src/gov_mem/backbones/coverage_packer.py` | Runtime module: coverage packer. |
| `src/gov_mem/backbones/coverage_verifier.py` | Runtime module: coverage verifier. |
| `src/gov_mem/backbones/govmem_incremental.py` | Runtime module: govmem incremental. |
| `src/gov_mem/backbones/govmem_v8_dev1_late_governance.py` | Compatibility alias for the active V8 implementation. |
| `src/gov_mem/backbones/govmem_v8_late_governance.py` | Gov-Mem v8: RAG-first retrieval with late claim-level governance. |
| `src/gov_mem/backbones/need_spec.py` | Runtime module: need spec. |
| `src/gov_mem/backbones/rag_naive.py` | Runtime module: rag naive. |
| `src/gov_mem/backbones/rag_naive_v3_no_symbolic.py` | RAG-Naive v3 (non-symbolic baseline). Excluded from Gov-Mem results. |
| `src/gov_mem/backbones/rag_policy.py` | Runtime module: rag policy. |
| `src/gov_mem/backbones/rag_policy_amem.py` | Runtime module: rag policy amem. |
| `src/gov_mem/backbones/relation_state.py` | Runtime module: relation state. |
| `src/gov_mem/backbones/semantic_compiler.py` | Lexicon-free, query-conditioned semantic compilation. |
| `src/gov_mem/backbones/slot_authorization.py` | Runtime module: slot authorization. |
| `src/gov_mem/backbones/stage2_typed_rerank.py` | Small, conservative Stage 2 pilot for typed scalar questions. |
| `src/gov_mem/backbones/stateful_policy.py` | Stateful Policy Reasoning backbone. |
| `src/gov_mem/backbones/symbolic_evidence.py` | Lightweight typed relation graph for Gov-Mem v4. |
| `src/gov_mem/backbones/utility_records.py` | Runtime module: utility records. |
| `src/gov_mem/backbones/v4_support.py` | Small support surface used by the paper-facing Gov-Mem v4 backbone. |
| `src/gov_mem/controlled_retrieval.py` | Content retrieval constrained by an already-issued PolicyDecision. |
| `src/gov_mem/data/__init__.py` | Package initializer. |
| `src/gov_mem/data/adapters.py` | Runtime module: adapters. |
| `src/gov_mem/data/checkpoint_benchmark.py` | Runtime module: checkpoint benchmark. |
| `src/gov_mem/data/schema.py` | Runtime module: schema. |
| `src/gov_mem/data/timestamps.py` | Helpers for preserving second-resolution source timestamps. |
| `src/gov_mem/entity_resolution.py` | Deterministic, provenance-free entity resolution for policy routing. |
| `src/gov_mem/eval/benchmark_official.py` | Runtime module: benchmark official. |
| `src/gov_mem/evaluation/__init__.py` | Package initializer. |
| `src/gov_mem/evaluation/evaluator.py` | Runtime module: evaluator. |
| `src/gov_mem/evaluation/metrics.py` | Runtime module: metrics. |
| `src/gov_mem/evaluation/prompt_context_audit.py` | Runtime module: prompt context audit. |
| `src/gov_mem/evolution/dev_guard.py` | Explicit provenance gate for artifacts that may influence future runtime. |
| `src/gov_mem/execution_planner.py` | Turn a PolicyDecision into an explicit execution plan. |
| `src/gov_mem/experience/__init__.py` | Package initializer. |
| `src/gov_mem/experience/adaptation_audit.py` | Runtime module: adaptation audit. |
| `src/gov_mem/experience/experience_bank.py` | Runtime module: experience bank. |
| `src/gov_mem/experience/failure_analyzer.py` | Runtime module: failure analyzer. |
| `src/gov_mem/experience/failure_case.py` | Runtime module: failure case. |
| `src/gov_mem/experience/pattern_inducer.py` | Runtime module: pattern inducer. |
| `src/gov_mem/experience/runtime_experience.py` | Runtime module: runtime experience. |
| `src/gov_mem/experience/structural_diagnosis.py` | Post-hoc, dev-only diagnosis from runtime traces rather than case wording. |
| `src/gov_mem/extraction/__init__.py` | Versioned extraction components for Gov-Mem. |
| `src/gov_mem/extraction/v8_event_extractor.py` | Scene-conditioned, query-independent event extraction for Gov-Mem v8. |
| `src/gov_mem/extraction/v8_event_schema.py` | Typed contracts for Gov-Mem v8 scene and governance extraction. |
| `src/gov_mem/extraction/v8_event_validator.py` | Source and roster validation for Gov-Mem v8 extraction output. |
| `src/gov_mem/extraction/v8_line_protocol.py` | Flat, unescaped record lines for model output; typed storage stays internal. |
| `src/gov_mem/extraction/v8_principal_registry.py` | Build a closed principal registry from checkpoint-observable episode data. |
| `src/gov_mem/extraction/v8_scene_schema.py` | Published, domain-independent structural vocabulary for Gov-Mem V8. |
| `src/gov_mem/field_state_projection.py` | Field-scoped state reasoning for the Stateful Policy pipeline. |
| `src/gov_mem/general_lexicon.py` | Compatibility surface after removing deterministic semantic vocabularies. |
| `src/gov_mem/governance_executor.py` | Execute structured governance decisions and update state explicitly. |
| `src/gov_mem/governance_runtime/__init__.py` | Package initializer. |
| `src/gov_mem/governance_runtime/access.py` | Runtime module: access. |
| `src/gov_mem/governance_runtime/action_predictor.py` | Runtime module: action predictor. |
| `src/gov_mem/governance_runtime/current_state.py` | Runtime module: current state. |
| `src/gov_mem/governance_runtime/evidence_frames.py` | Source-grounded evidence-frame adapter. |
| `src/gov_mem/governance_runtime/factual_claim_quality.py` | Request-aware quality checks for source-grounded factual values. |
| `src/gov_mem/governance_runtime/information_owner_ledger.py` | Episode-local attribution of a fact's protected-information owner. |
| `src/gov_mem/governance_runtime/leakage_guard.py` | Runtime module: leakage guard. |
| `src/gov_mem/governance_runtime/policy_frames.py` | LLM-derived, provenance-preserving policy bindings for governed atoms. |
| `src/gov_mem/governance_runtime/principal_relation_ledger.py` | Episode-local, source-grounded principal relationship resolution. |
| `src/gov_mem/governance_runtime/provenance_authorization.py` | Slot-level authorization certificates for versioned governed evidence. |
| `src/gov_mem/governance_runtime/query_policy_authorization.py` | Query-scoped policy authorization over closed graph evidence. |
| `src/gov_mem/governance_runtime/source_grounding.py` | Source-local text views for typed evidence validation. |
| `src/gov_mem/governance_runtime/source_role_ledger.py` | Episode-local fact/policy source separation for governed graph evidence. |
| `src/gov_mem/governance_runtime/state_ledger.py` | Runtime module: state ledger. |
| `src/gov_mem/governance_runtime/utility_answering.py` | Runtime module: utility answering. |
| `src/gov_mem/governance_runtime/utility_source_locator.py` | LLM-mediated, closed-set utility source selection for one episode. |
| `src/gov_mem/governance_runtime/v8_claim_reasoner.py` | One language pass for incremental graph updates and late governance. |
| `src/gov_mem/governance_runtime/v8_event_store.py` | Append-only, checkpoint-safe event storage for Gov-Mem v8. |
| `src/gov_mem/governance_runtime/v8_prompt_input.py` | Lossless prompt transport and explicit application-supplied access policy. |
| `src/gov_mem/governance_runtime/v8_prompt_sources.py` | Exact excerpts actually shown to Stage 2, without hidden history expansion. |
| `src/gov_mem/governance_runtime/v8_safe_evidence.py` | Only explicitly released excerpts cross the final-answer boundary. |
| `src/gov_mem/governance_runtime/v8_shallow_memory.py` | V8's shallow policy ledger: raw memory is not converted into a world graph. |
| `src/gov_mem/governance_runtime/v8_state_projector.py` | Deterministic, prefix-safe projection of the V8 permission/event graph. |
| `src/gov_mem/governance_runtime/v8_symbolic_critic.py` | Late symbolic verification after LLM scope matching, without lexical vetoes. |
| `src/gov_mem/graph/__init__.py` | Package initializer. |
| `src/gov_mem/graph/governed_graph.py` | Runtime module: governed graph. |
| `src/gov_mem/graph/graph_builder.py` | Runtime module: graph builder. |
| `src/gov_mem/graph/graph_retriever.py` | Runtime module: graph retriever. |
| `src/gov_mem/graph/graph_store.py` | Runtime module: graph store. |
| `src/gov_mem/legacy/__init__.py` | Legacy Gov-Mem implementations retained for comparison only. |
| `src/gov_mem/legacy/claim_adjudicator.py` | LEGACY: batch claim/state adjudication over a closed Stage-2 evidence set. |
| `src/gov_mem/legacy/graph_slot_renderer.py` | LEGACY: graph-certified, source-grounded typed slot realization. |
| `src/gov_mem/legacy/semantic_alignment.py` | LEGACY: LLM-mediated alignment from query attributes to observed graph slots. |
| `src/gov_mem/legacy/semantic_reranker.py` | LEGACY: closed-set LLM reranking from the pre-Stateful-Policy runtime. |
| `src/gov_mem/legacy/typed_realization_audit.py` | LEGACY: closed-set semantic realization for typed memory answers. |
| `src/gov_mem/llm/__init__.py` | Package initializer. |
| `src/gov_mem/llm/client.py` | Runtime module: client. |
| `src/gov_mem/llm/json_parser.py` | Runtime module: json parser. |
| `src/gov_mem/llm/model_registry.py` | Runtime module: model registry. |
| `src/gov_mem/llm/prompts.py` | Runtime module: prompts. |
| `src/gov_mem/memory/__init__.py` | Package initializer. |
| `src/gov_mem/memory/amem_memory.py` | Runtime module: amem memory. |
| `src/gov_mem/memory/dense_index.py` | Runtime module: dense index. |
| `src/gov_mem/memory/governed_atom.py` | Runtime module: governed atom. |
| `src/gov_mem/memory/governed_slot_graph.py` | Optional ingestion-time, source-grounded governed slot graph. |
| `src/gov_mem/memory/ingestion.py` | Runtime module: ingestion. |
| `src/gov_mem/memory/symbolic_store.py` | Runtime module: symbolic store. |
| `src/gov_mem/memory/v8_slot_graph.py` | Transactional V8 adapter for the source-grounded slot graph. |
| `src/gov_mem/pipeline.py` | Runtime module: pipeline. |
| `src/gov_mem/planning/__init__.py` | Package initializer. |
| `src/gov_mem/planning/query_planner.py` | Runtime module: query planner. |
| `src/gov_mem/policy_conflict_resolver.py` | Configurable, deterministic policy conflict resolution. |
| `src/gov_mem/policy_reasoner.py` | Stateful Policy Reasoning: the single governance decision boundary. |
| `src/gov_mem/policy_schema.py` | Structured schemas for Stateful Policy Reasoning. |
| `src/gov_mem/policy_selector.py` | Applicable-policy selection over explicit PolicyState. |
| `src/gov_mem/policy_state_builder.py` | Build an observable PolicyState from an episode prefix. |
| `src/gov_mem/policy_verifier.py` | Final permission/privacy verification before a response is delivered. |
| `src/gov_mem/query_intent_parser.py` | Natural-language query intent parsing for policy reasoning. |
| `src/gov_mem/query_semantics.py` | Compatibility boundary for query semantics. |
| `src/gov_mem/reasoning/__init__.py` | Package initializer. |
| `src/gov_mem/reasoning/governance_decision.py` | Runtime module: governance decision. |
| `src/gov_mem/reasoning/operators.py` | Lexicon-free structural operators for Gov-Mem. |
| `src/gov_mem/reasoning/predicates.py` | Runtime module: predicates. |
| `src/gov_mem/reasoning/proof_trace.py` | Runtime module: proof trace. |
| `src/gov_mem/reasoning/reasoner.py` | Runtime module: reasoner. |
| `src/gov_mem/reasoning/symbolic_reasoner.py` | Runtime module: symbolic reasoner. |
| `src/gov_mem/retrieval/__init__.py` | Package initializer. |
| `src/gov_mem/retrieval/dense_retriever.py` | Runtime module: dense retriever. |
| `src/gov_mem/retrieval/dual_channel_retriever.py` | Runtime module: dual channel retriever. |
| `src/gov_mem/retrieval/governance_retriever.py` | Runtime module: governance retriever. |
| `src/gov_mem/retrieval/hybrid_retriever.py` | Runtime module: hybrid retriever. |
| `src/gov_mem/retrieval/retrieval_bundle.py` | Runtime module: retrieval bundle. |
| `src/gov_mem/retrieval/symbolic_retriever.py` | Runtime module: symbolic retriever. |
| `src/gov_mem/retrieval/utility_retriever.py` | Runtime module: utility retriever. |
| `src/gov_mem/skills/__init__.py` | Package initializer. |
| `src/gov_mem/skills/governance_skill.py` | Runtime module: governance skill. |
| `src/gov_mem/skills/registry.py` | Runtime module: registry. |
| `src/gov_mem/skills/runtime_skill_router.py` | Runtime module: runtime skill router. |
| `src/gov_mem/skills/skill_context.py` | Runtime module: skill context. |
| `src/gov_mem/skills/skill_executor.py` | Runtime module: skill executor. |
| `src/gov_mem/skills/skill_library.py` | Runtime module: skill library. |
| `src/gov_mem/skills/skill_retriever.py` | Runtime module: skill retriever. |
| `src/gov_mem/skills/updater.py` | Runtime module: updater. |
| `src/gov_mem/state_transition_engine.py` | Pure temporal replay of PolicyState for one query. |
| `src/gov_mem/utils/__init__.py` | Package initializer. |
| `src/gov_mem/utils/config.py` | Runtime module: config. |
| `src/gov_mem/utils/io.py` | Runtime module: io. |
| `src/gov_mem/utils/logging.py` | Runtime module: logging. |
| `src/gov_mem/utils/storage.py` | Storage controls for high-concurrency experiment runs. |

## Tests

| File | Coverage |
|---|---|
| `tests/test_v8_event_extractor.py` | `test_registry_keeps_identity_relationship_but_drops_future_scope_summary`, `test_permission_can_use_cross_turn_sources_and_closed_principal_id`, `test_lifecycle_effect_without_grantee_is_not_rejected_as_missing_grantee`, `test_scene_is_explicitly_extracted_and_grounded`, `test_completed_deletion_backstop_adds_tombstone_when_llm_omits_it`, `test_deletion_request_does_not_create_completed_tombstone`, `test_direct_deletion_request_creates_immediate_tombstone`, `test_ontology_is_small_and_domain_independent`, `test_exact_source_grounding_rejects_changed_punctuation`, `test_compressed_fact_restores_exact_source_including_qualifiers`, `test_text_prefill_extracts_source_grounded_graph_without_json` |
| `tests/test_v8_late_governance.py` | `test_missing_permission_or_scene_member_does_not_veto`, `test_lexical_overlap_is_not_resource_scope_matching`, `test_bound_unconditional_deny_is_enforced_after_language_matching`, `test_ambiguous_scope_and_different_action_are_not_hard_vetoes`, `test_subject_and_record_neighbors_do_not_propagate_veto`, `test_shared_value_does_not_mean_same_deleted_resource`, `test_scoped_allow_does_not_relabel_complete_release_as_summary`, `test_requester_authored_sensitive_value_is_not_automatically_authorized`, `test_block_requires_actual_visible_restriction`, `test_claim_cannot_normalize_away_negative_sign_or_invent_quote`, `test_safe_view_omits_denied_value_even_in_free_text_reason_and_slot`, `test_background_does_not_count_as_partial_answer`, `test_overlapping_protected_source_span_is_masked_but_other_occurrence_survives`, `test_duplicate_values_require_an_unambiguous_source_binding`, `test_wrong_candidate_index_is_rebound_only_to_unique_exact_visible_quote`, `test_blocked_paraphrase_uses_exact_protected_quote_without_releasing_it`, `test_compact_claim_value_is_its_exact_quote`, `test_ungrounded_release_is_omitted_but_malformed_denial_is_fatal`, `test_allow_in_other_scene_does_not_disable_explicit_bound_deny`, `test_identical_denied_values_keep_both_source_spans_for_late_redaction`, `test_invalid_ambiguous_duplicate_cannot_suppress_later_grounded_claim`, `test_grounded_deleted_classification_is_symbolically_blocked_without_repair`, `test_malformed_deleted_release_with_grounded_span_is_dropped`, `test_nonapplicable_lifecycle_event_does_not_create_a_deletion` |
| `tests/test_v8_late_governance_backbone.py` | `test_v8_backbone_keeps_rag_first_and_answers_from_safe_claims`, `test_failed_extraction_does_not_advance_cache_watermark`, `test_same_visible_prefix_reuses_event_cache_without_extraction_call`, `test_deleted_only_safe_view_returns_no_memory_action`, `test_refusal_takes_precedence_over_confirming_deleted_state`, `test_explicit_deleted_history_request_returns_no_memory_even_if_unauthorized`, `test_joint_delta_governs_current_query_and_never_reaches_answer_prompt`, `test_full_scoped_answer_is_not_redacted`, `test_missing_events_does_not_mark_new_turns_processed`, `test_cold_prefix_extra_call_is_bounded_counted_and_cached`, `test_cold_prefix_budget_failure_makes_no_paid_request`, `test_runner_stops_paid_work_after_first_failure_and_saves_telemetry`, `test_invalid_advisory_fact_does_not_discard_valid_answer_or_relax_grounding`, `test_invalid_lifecycle_fact_remains_a_critical_extraction_failure`, `test_contract_repair_is_bounded_and_metered`, `test_evaluation_retains_errors_and_attempts_entire_episode`, `test_line_protocol_end_to_end_activates_bound_symbolic_veto`, `test_valid_graph_delta_survives_bad_claim_and_is_not_reextracted`, `test_public_policy_mixed_request_keeps_utility_and_hides_only_blocked_value`, `test_joint_extraction_validates_old_retrieved_source_outside_ingestion_context`, `test_enabled_slot_graph_is_observer_and_does_not_supply_stage2_evidence`, `test_slot_graph_observer_reports_explicit_findings_without_evidence_effect`, `test_graph_audit_is_passed_to_stage2_as_advisory` |
| `tests/test_v8_line_protocol.py` | `test_lines_preserve_quotes_backslashes_equals_and_literal_none`, `test_malformed_or_incomplete_records_fail_explicitly`, `test_bind_uses_exact_event_identity_and_rejects_unknown_or_incompatible_events`, `test_text_api_omits_json_mode_preserves_literals_and_does_not_retry_truncation` |
| `tests/test_v8_nested_denial.py` | `test_nested_denial_preserves_decision_sources_and_never_becomes_keep_release`, `test_conflicting_nested_denial_fails`, `test_incomplete_or_ambiguous_denial_fails`, `test_legacy_keep_without_bind_does_not_create_permission`, `test_nested_denial_cannot_bypass_exact_source_validation` |
| `tests/test_v8_prompt_sources.py` | `test_sources_are_exactly_exposed_and_mixed_chunks_have_no_false_attribution`, `test_old_retrieved_lifecycle_is_context_not_a_new_graph_event`, `test_new_lifecycle_can_cite_old_retrieved_source_and_new_change`, `test_lifecycle_uses_cited_words_instead_of_generated_boolean_or_deleted_value` |
| `tests/test_v8_shallow_contract_recovery.py` | `test_literal_controls_are_preserved_without_structural_repair`, `test_quote_only_release_still_requires_exact_source`, `test_explicit_keep_false_omission_is_ignored_without_contract_failure`, `test_incomplete_claim_is_ignored_and_audited_without_release`, `test_candidate_only_denial_stays_blocked_and_requires_grounded_restriction`, `test_visible_source_binding_is_provenance_only_and_future_ids_fail`, `test_multiple_restrictions_cannot_manufacture_canonical_binding`, `test_grounded_denial_does_not_require_redundant_empty_binding`, `test_source_candidate_ids_are_exact_and_do_not_map_arbitrary_numbers`, `test_only_exact_already_supplied_source_can_become_candidate`, `test_disjoint_prompt_fragments_cannot_be_joined_into_a_fake_candidate` |
| `tests/test_v8_slot_graph.py` | `test_graph_retrieves_original_sources_and_reuses_only_unchanged_prefix`, `test_failed_graph_extension_does_not_commit_partial_memory` |
| `tests/test_v8_state_projector.py` | `test_projection_excludes_cached_future_revocation`, `test_projection_exposes_source_grounded_delete_tombstone`, `test_same_scene_observations_merge_only_after_their_turn`, `test_same_scene_same_field_distinct_subjects_do_not_overwrite`, `test_scopes_conditions_and_issuers_are_preserved`, `test_expired_permission_and_pending_revocation`, `test_whole_resource_revocation_supersedes_scoped_allow`, `test_lifecycle_events_without_grantee_reach_query_context_without_future_leakage`, `test_future_start_remains_pending_even_if_expiry_is_present` |

One stale test in `test_v8_nested_denial.py` was aligned with the existing normalizer: a legacy whole-candidate KEEP without bind produces an empty permission-event binding, not a fabricated permission. Runtime behavior was not changed.
