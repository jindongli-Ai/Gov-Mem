"""Static boundary checks for the Stateful Policy runtime."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    ROOT / "src/gov_mem/policy_schema.py",
    ROOT / "src/gov_mem/policy_state_builder.py",
    ROOT / "src/gov_mem/query_intent_parser.py",
    ROOT / "src/gov_mem/policy_selector.py",
    ROOT / "src/gov_mem/state_transition_engine.py",
    ROOT / "src/gov_mem/policy_conflict_resolver.py",
    ROOT / "src/gov_mem/policy_reasoner.py",
    ROOT / "src/gov_mem/execution_planner.py",
    ROOT / "src/gov_mem/controlled_retrieval.py",
    ROOT / "src/gov_mem/governance_executor.py",
    ROOT / "src/gov_mem/answer_projection.py",
    ROOT / "src/gov_mem/backbones/stateful_policy.py",
]
FORBIDDEN_IMPORTS = {
    "semantic_reranker",
    "claim_adjudicator",
    "typed_realization_audit",
    "evaluation.evaluator",
}


def main() -> None:
    for path in RUNTIME_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(token in name for token in FORBIDDEN_IMPORTS):
                    raise SystemExit(f"forbidden legacy import in {path}: {name}")
    pipeline = (ROOT / "src/gov_mem/pipeline.py").read_text(encoding="utf-8")
    if "semantic_reranker" in pipeline:
        raise SystemExit("pipeline contains an active semantic_reranker dependency")
    print("stateful_policy_runtime_static_check=PASS")


if __name__ == "__main__":
    main()
