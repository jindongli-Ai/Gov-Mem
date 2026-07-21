"""Regression check: a failed graph projection preserves terminal actions."""

from gov_mem.backbones.rag_policy_amem import resolve_uncertified_graph_action


def main() -> None:
    assert resolve_uncertified_graph_action("refuse") == "refuse"
    assert resolve_uncertified_graph_action("no_memory") == "no_memory"
    assert resolve_uncertified_graph_action("answer") == "answer_redacted"
    assert resolve_uncertified_graph_action("answer_redacted") == "answer_redacted"
    print("uncertified_graph_action_smoke=PASS")


if __name__ == "__main__":
    main()
