from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.evolution.policy_updater import PolicyUpdater
from gov_mem.evolution.prompt_updater import PromptUpdater
from gov_mem.evolution.rule_updater import AuditableUpdate, RuleUpdater
from gov_mem.evolution.dev_guard import load_dev_attestation, require_matching_attestation
from gov_mem.experience.pattern_inducer import FailurePattern
from gov_mem.skills.governance_skill import GovernanceSkill
from gov_mem.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dev-only auditable updates from failure patterns and governance skills.")
    parser.add_argument("--pattern_path", required=True, help="Path to failure_patterns.jsonl")
    parser.add_argument("--dev_attestation", required=True, help="Development-only provenance attestation JSON.")
    parser.add_argument("--skill_path", required=True, help="Path to governance_skill_library.jsonl")
    parser.add_argument("--output_dir", default="outputs/evolution", help="Output directory")
    args = parser.parse_args()

    attestation = load_dev_attestation(args.dev_attestation)

    pattern_rows = read_jsonl(args.pattern_path)
    require_matching_attestation(artifacts=pattern_rows, attestation=attestation)
    patterns = [FailurePattern(**row) for row in pattern_rows]
    skill_rows = read_jsonl(args.skill_path)
    require_matching_attestation(
        artifacts=[{"provenance": dict((row.get("metadata") or {}))} for row in skill_rows],
        attestation=attestation,
    )
    skills = [GovernanceSkill(**row) for row in skill_rows]
    rule_updates = RuleUpdater().build(patterns=patterns, skills=skills)
    prompt_updates = PromptUpdater().build(patterns=patterns, skills=skills)
    policy_updates = PolicyUpdater().build(patterns=patterns, skills=skills)

    output_dir = Path(args.output_dir)
    all_updates = rule_updates + prompt_updates + policy_updates
    for update in all_updates:
        update.metadata["dev_attestation"] = attestation
    write_jsonl(output_dir / "rule_patches.jsonl", [asdict(item) for item in rule_updates])
    write_jsonl(output_dir / "prompt_patches.jsonl", [asdict(item) for item in prompt_updates])
    write_jsonl(output_dir / "policy_patches.jsonl", [asdict(item) for item in policy_updates])
    update_log = _merge_updates(all_updates)
    write_jsonl(output_dir / "update_log.jsonl", [asdict(item) for item in update_log])
    print(f"rule_updates={len(rule_updates)}")
    print(f"prompt_updates={len(prompt_updates)}")
    print(f"policy_updates={len(policy_updates)}")
    print(f"update_log={len(update_log)}")
    print(f"output_dir={output_dir.resolve()}")


def _merge_updates(updates: list[AuditableUpdate]) -> list[AuditableUpdate]:
    deduped: list[AuditableUpdate] = []
    seen: set[str] = set()
    for update in updates:
        if update.update_id in seen:
            continue
        seen.add(update.update_id)
        deduped.append(update)
    return deduped


if __name__ == "__main__":
    main()
