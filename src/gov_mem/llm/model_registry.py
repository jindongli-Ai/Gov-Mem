from __future__ import annotations

from typing import Any


DEFAULT_ROLE_FALLBACKS = {
    "base": "query_planner_model",
    "memory_ingestion": "memory_ingestion_model",
    "query_planning": "query_planner_model",
    "reasoning": "query_planner_model",
    "action_decision": "query_planner_model",
    "answering": "answering_model",
}


def resolve_llm_model(config: dict[str, Any], role: str) -> str:
    llm_cfg = dict(config.get("llm") or {})
    role_overrides = dict(llm_cfg.get("role_models") or {})
    if role_overrides.get(role):
        return str(role_overrides[role])

    if llm_cfg.get("base_model"):
        return str(llm_cfg["base_model"])

    legacy_key = DEFAULT_ROLE_FALLBACKS.get(role)
    if legacy_key and llm_cfg.get(legacy_key):
        return str(llm_cfg[legacy_key])

    if llm_cfg.get("base_model"):
        return str(llm_cfg["base_model"])

    for key in ("query_planner_model", "answering_model", "memory_ingestion_model"):
        if llm_cfg.get(key):
            return str(llm_cfg[key])

    raise KeyError(f"No LLM model configured for role={role!r}.")


def build_resolved_llm_settings(config: dict[str, Any]) -> dict[str, Any]:
    llm_cfg = dict(config.get("llm") or {})
    resolved = {
        "provider": llm_cfg.get("provider"),
        "api_base": llm_cfg.get("api_base"),
        "api_key_env": llm_cfg.get("api_key_env"),
        "temperature": llm_cfg.get("temperature"),
        "max_retries": llm_cfg.get("max_retries"),
        "request_timeout": llm_cfg.get("request_timeout"),
        "retry_backoff_seconds": llm_cfg.get("retry_backoff_seconds"),
        "max_retry_backoff_seconds": llm_cfg.get("max_retry_backoff_seconds"),
        "allow_fallback": llm_cfg.get("allow_fallback"),
        "base_model": resolve_llm_model(config, "base"),
        "role_models_requested": dict(llm_cfg.get("role_models") or {}),
        "role_models_resolved": {},
    }
    for role in ("memory_ingestion", "query_planning", "reasoning", "action_decision", "answering"):
        resolved["role_models_resolved"][role] = resolve_llm_model(config, role)
    return resolved
