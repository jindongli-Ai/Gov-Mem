from scripts.run_v8_late_veto_ablation import branch_ledger, computation, make_instance


def test_disable_veto_preserves_language_denial_and_does_not_mutate():
    before = {'claims': [{'delivery': 'block'}, {'delivery': 'exact'}]}
    final = {'claims': [
        {'delivery': 'block', 'symbolic_veto_reasons': ['bound_lifecycle_deletion_veto']},
        {'delivery': 'block', 'symbolic_veto_reasons': ['bound_lifecycle_deletion_veto']}]}
    result = branch_ledger(before, final, 'no_late_veto')
    assert [c['delivery'] for c in result['claims']] == ['block', 'exact']
    result['claims'][0]['delivery'] = 'exact'
    assert before['claims'][0]['delivery'] == 'block'


def test_disable_one_rule_preserves_other_sufficient_veto():
    before = {'claims': [{'delivery': 'exact'}]}
    final = {'claims': [{'delivery': 'block', 'symbolic_veto_reasons': [
        'bound_lifecycle_deletion_veto', 'bound_explicit_permission_veto']}]}
    result = branch_ledger(before, final, 'no_bound_deletion')
    assert result['claims'][0]['delivery'] == 'block'
    assert result['claims'][0]['symbolic_veto_reasons'] == ['bound_explicit_permission_veto']


def test_computation_detects_no_call_refusal_semantics():
    instance = make_instance({'question': 'What was the old value?', 'requester': {'principal_id': 'p'}}, 'c', 'd')
    ledger = {'query_slots': [{'slot': 'value'}], 'answer_action': 'no_memory', 'claims': [
        {'claim_id': 'c1', 'slot': 'value', 'value': 'secret', 'delivery': 'block'}]}
    base = computation(instance, ledger, {})
    ledger['claims'][0]['symbolic_veto_reasons'] = ['source_grounded_deletion_tombstone']
    deleted = computation(instance, ledger, {})
    assert base['request'] is deleted['request'] is None
    assert base != deleted
