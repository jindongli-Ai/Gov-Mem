from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any

import requests

from gov_mem.llm.json_parser import parse_json_response


class LLMClientUnavailableError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str
    temperature: float = 0.0
    max_output_tokens: int = 4096
    max_retries: int = 3
    api_base: str | None = None
    api_key_env: str | None = None
    allow_fallback: bool = True
    request_timeout: int = 120
    retry_backoff_seconds: float = 3.0
    max_retry_backoff_seconds: float = 30.0
    cache_dir: str | None = None


YUNWU_PROVIDER_NAMES = {"yunwu", "openai-compatible-yunwu"}
JELLYFISHP_PROVIDER_NAMES = {"jellyfishp", "openai-compatible-jellyfishp"}
OPENLUX_PROVIDER_NAMES = {"openlux", "openai-compatible-openlux"}
OPENAI_COMPATIBLE_PROVIDER_NAMES = YUNWU_PROVIDER_NAMES | JELLYFISHP_PROVIDER_NAMES | OPENLUX_PROVIDER_NAMES
EMBEDDING_BATCH_SIZE = 16
LOGGER = logging.getLogger("govmem")


def _default_api_key_env(provider: str) -> str:
    if provider in YUNWU_PROVIDER_NAMES:
        return "YUNWU_API_KEY"
    if provider in JELLYFISHP_PROVIDER_NAMES:
        return "JELLYFISHP_API_KEY"
    if provider in OPENLUX_PROVIDER_NAMES:
        return "OPENLUX_API_KEY"
    return "OPENAI_API_KEY"


def _api_key_pool_env(api_key_env: str) -> str:
    return f"{api_key_env[:-3]}KEYS" if api_key_env.endswith("KEY") else f"{api_key_env}_POOL"


def is_real_llm_enabled(config: LLMConfig) -> bool:
    provider = config.provider.lower().strip()
    api_key_env = config.api_key_env or _default_api_key_env(provider)
    return bool(os.environ.get(api_key_env))


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._session = requests.Session()
        self._telemetry: dict[str, dict[str, float | int]] = {}

    def telemetry_snapshot(self) -> dict[str, dict[str, float | int]]:
        return {name: dict(values) for name, values in self._telemetry.items()}

    def _record_telemetry(self, *, operation: str, elapsed_s: float, retries: int = 0, failed: bool = False) -> None:
        row = self._telemetry.setdefault(operation, {"calls": 0, "failures": 0, "retries": 0, "elapsed_s": 0.0})
        row["calls"] = int(row["calls"]) + 1
        row["failures"] = int(row["failures"]) + int(failed)
        row["retries"] = int(row["retries"]) + int(retries)
        row["elapsed_s"] = float(row["elapsed_s"]) + float(elapsed_s)

    def _api_key(self) -> str | None:
        provider = self.config.provider.lower().strip()
        default_env = _default_api_key_env(provider)
        env_name = self.config.api_key_env or default_env
        return os.environ.get(env_name)

    def _api_key_pool(self) -> list[str]:
        """Return the configured compatible-provider key pool without logging keys."""
        if self.provider_name() not in OPENAI_COMPATIBLE_PROVIDER_NAMES:
            return []
        api_key_env = self.config.api_key_env or _default_api_key_env(self.provider_name())
        values = [value.strip() for value in os.environ.get(_api_key_pool_env(api_key_env), "").split(",")]
        return list(dict.fromkeys(value for value in values if value))

    def _retry_api_key(self, current_key: str | None) -> str | None:
        """Rotate only retry traffic so transient provider throttling can recover."""
        pool = self._api_key_pool()
        if len(pool) < 2:
            return current_key
        try:
            current_index = pool.index(str(current_key))
        except ValueError:
            current_index = -1
        return pool[(current_index + 1) % len(pool)]

    def provider_name(self) -> str:
        return self.config.provider.lower().strip()

    def is_available(self) -> bool:
        provider = self.provider_name()
        if provider in {"mock", "stub", "heuristic", "offline"}:
            return False
        return bool(self._api_key())

    def require_or_raise(self) -> None:
        if self.is_available():
            return
        if not self.config.allow_fallback:
            provider = self.provider_name()
            env_name = self.config.api_key_env or _default_api_key_env(provider)
            raise LLMClientUnavailableError(
                f"LLM API is required but not available. Missing env var {env_name} for provider={provider}."
            )

    def _resolve_api_base(self) -> str:
        provider = self.provider_name()
        if provider in YUNWU_PROVIDER_NAMES:
            default_base = "https://yunwu.ai/v1"
        elif provider in OPENLUX_PROVIDER_NAMES:
            default_base = "https://api.openlux.ai/v1"
        else:
            default_base = "https://api.openai.com/v1"
        return (
            self.config.api_base
            or os.environ.get("YUNWU_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or default_base
        ).rstrip("/")

    def _should_retry_http_error(self, exc: requests.HTTPError) -> bool:
        status_code = exc.response.status_code if exc.response is not None else None
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

    def _retry_delay(self, attempt: int, response: requests.Response | None = None) -> float:
        if response is not None:
            retry_after = str(response.headers.get("Retry-After") or "").strip()
            try:
                if retry_after:
                    return max(float(retry_after), 0.0)
            except ValueError:
                pass
        base = max(float(self.config.retry_backoff_seconds), 0.25)
        max_delay = max(float(self.config.max_retry_backoff_seconds), base)
        return min(base * (2 ** max(0, attempt - 1)), max_delay)

    def _post_json(self, *, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        retries = 0
        api_base = self._resolve_api_base()
        current_key = self._api_key()
        headers = {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            model = str(payload.get("model") or "")
            LOGGER.info(
                "[Gov-Mem] API request endpoint=%s model=%s attempt=%d/%d",
                endpoint,
                model,
                attempt,
                self.config.max_retries,
            )
            try:
                response = self._session.post(
                    f"{api_base}/{endpoint.lstrip('/')}",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.config.request_timeout,
                )
                response.raise_for_status()
                elapsed_s = time.monotonic() - started
                self._record_telemetry(operation=endpoint, elapsed_s=elapsed_s, retries=retries)
                LOGGER.info(
                    "[Gov-Mem] API success endpoint=%s model=%s elapsed_s=%.2f retries=%d",
                    endpoint,
                    model,
                    elapsed_s,
                    retries,
                )
                return response.json()
            except requests.HTTPError as exc:
                last_error = exc
                if attempt >= self.config.max_retries or not self._should_retry_http_error(exc):
                    self._record_telemetry(operation=endpoint, elapsed_s=time.monotonic() - started, retries=retries, failed=True)
                    LOGGER.warning(
                        "[Gov-Mem] API failed endpoint=%s model=%s attempt=%d error=%s",
                        endpoint,
                        model,
                        attempt,
                        type(exc).__name__,
                    )
                    raise
                retries += 1
                current_key = self._retry_api_key(current_key)
                headers["Authorization"] = f"Bearer {current_key}"
                LOGGER.warning(
                    "[Gov-Mem] API retry endpoint=%s model=%s attempt=%d error=%s",
                    endpoint,
                    model,
                    attempt,
                    type(exc).__name__,
                )
                time.sleep(self._retry_delay(attempt, exc.response))
                continue
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    self._record_telemetry(operation=endpoint, elapsed_s=time.monotonic() - started, retries=retries, failed=True)
                    LOGGER.warning(
                        "[Gov-Mem] API failed endpoint=%s model=%s attempt=%d error=%s",
                        endpoint,
                        model,
                        attempt,
                        type(exc).__name__,
                    )
                    raise
                retries += 1
                current_key = self._retry_api_key(current_key)
                headers["Authorization"] = f"Bearer {current_key}"
                LOGGER.warning(
                    "[Gov-Mem] API retry endpoint=%s model=%s attempt=%d error=%s",
                    endpoint,
                    model,
                    attempt,
                    type(exc).__name__,
                )
            time.sleep(self._retry_delay(attempt))
        if last_error is not None:
            self._record_telemetry(operation=endpoint, elapsed_s=time.monotonic() - started, retries=retries, failed=True)
            raise last_error
        raise RuntimeError("Unreachable retry loop exit in _post_json.")

    def _cache_root(self) -> Path | None:
        if not self.config.cache_dir:
            return None
        return Path(self.config.cache_dir)

    def _embedding_cache_path(self, *, model: str, text: str) -> Path | None:
        cache_root = self._cache_root()
        if cache_root is None:
            return None
        digest = md5(f"{self.provider_name()}::{self._resolve_api_base()}::{model}::{text}".encode("utf-8")).hexdigest()
        return cache_root / model / f"{digest}.json"

    def _load_cached_embedding(self, *, model: str, text: str) -> list[float] | None:
        cache_path = self._embedding_cache_path(model=model, text=text)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        embedding = raw.get("embedding")
        if isinstance(embedding, list):
            return [float(value) for value in embedding]
        return None

    def _save_cached_embedding(self, *, model: str, text: str, embedding: list[float]) -> None:
        cache_path = self._embedding_cache_path(model=model, text=text)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "provider": self.provider_name(),
                    "api_base": self._resolve_api_base(),
                    "model": model,
                    "text": text,
                    "embedding": embedding,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def chat_json(self, *, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any] | list[Any]:
        if not self.is_available():
            self.require_or_raise()
            raise LLMClientUnavailableError("LLM API key is not available.")

        provider = self.provider_name()
        if provider not in {"openai", *OPENAI_COMPATIBLE_PROVIDER_NAMES}:
            raise LLMClientUnavailableError(f"Provider {provider!r} is not implemented.")

        payload = {
            "model": model,
            "temperature": self.config.temperature,
            # GateMem paper-style runs use a 4096-token answer budget. Yunwu,
            # OpenLux, and other OpenAI-compatible endpoints expect max_tokens.
            "max_tokens": int(self.config.max_output_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        for attempt in range(1, self.config.max_retries + 1):
            raw = self._post_json(endpoint="chat/completions", payload=payload)
            content = raw["choices"][0]["message"]["content"]
            try:
                return parse_json_response(content)
            except Exception:
                if attempt >= self.config.max_retries:
                    raise
        raise RuntimeError("Unreachable retry loop exit in chat_json.")

    def embed_texts(self, *, model: str, texts: list[str]) -> list[list[float]]:
        if not self.is_available():
            self.require_or_raise()
            raise LLMClientUnavailableError("Embedding API key is not available.")

        provider = self.provider_name()
        if provider not in {"openai", *OPENAI_COMPATIBLE_PROVIDER_NAMES}:
            raise LLMClientUnavailableError(f"Embedding provider {provider!r} is not implemented.")

        unique_texts: list[str] = []
        unique_lookup: dict[str, int] = {}
        ordered_embeddings: list[list[float] | None] = [None] * len(texts)
        for idx, text in enumerate(texts):
            cached = self._load_cached_embedding(model=model, text=text)
            if cached is not None:
                ordered_embeddings[idx] = cached
                self._record_telemetry(operation="embedding_cache_hit", elapsed_s=0.0)
                continue
            self._record_telemetry(operation="embedding_cache_miss", elapsed_s=0.0)
            if text not in unique_lookup:
                unique_lookup[text] = len(unique_texts)
                unique_texts.append(text)
        fresh_map: dict[str, list[float]] = {}
        # Large episodic contexts can produce hundreds of chunks. Bound each
        # provider request while retaining exact per-text caching and order.
        for start in range(0, len(unique_texts), EMBEDDING_BATCH_SIZE):
            batch = unique_texts[start : start + EMBEDDING_BATCH_SIZE]
            raw = self._post_json(endpoint="embeddings", payload={"model": model, "input": batch})
            rows = list(raw.get("data") or []) if isinstance(raw, dict) else []
            if len(rows) != len(batch):
                raise ValueError(
                    f"Embedding response count mismatch: requested={len(batch)} received={len(rows)}"
                )
            for text, row in zip(batch, rows):
                embedding = row.get("embedding") if isinstance(row, dict) else None
                if not isinstance(embedding, list):
                    raise ValueError("Embedding response row is missing an embedding vector")
                vector = [float(value) for value in embedding]
                fresh_map[text] = vector
                self._save_cached_embedding(model=model, text=text, embedding=vector)
        for idx, text in enumerate(texts):
            if ordered_embeddings[idx] is None:
                ordered_embeddings[idx] = fresh_map[text]
        return [[float(value) for value in (embedding or [])] for embedding in ordered_embeddings]
