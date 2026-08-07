# Stateful Policy Action Contract and Mixed Delivery Probe

## Scope

This iteration made four bounded changes around one boundary:

- recognize natural-language predecessor recovery (`earlier provisional`, `appeared before approved`) as lifecycle content;
- map explicit deleted/removed recovery requests to `no_memory`, while preserving ordinary access-window denial as `refuse`;
- allow an explicitly requested safe/logistics projection to use only active policy-approved safe carriers;
- remove delivery-instruction pseudo-fields and preserve an aggregate `authorized_safe_summary` contract without binding it to a synthetic subject.

The existing location-carrier repair was retained.

## Verification

| Check | Result |
|---|---:|
| Stateful policy unit tests | 105 passed |
| Runtime static check | PASS |
| Real API pipeline repair probe v22 | 7 checkpoints; household/office utility representatives passed; education lifecycle action mismatch remained |
| Real API action-contract probe v28 | education lifecycle action correct; household safe-summary action correct; household utility content 0.0 |

The v28 representative results were:

| Case | Gold action | Predicted action | Utility | Leakage |
|---|---|---|---:|---:|
| Education deleted exact wording | no_memory | no_memory | 0.0 | 0 |
| Household concise logistics summary | answer | answer | 0.0 | 0 |

## Conclusion

The action boundary repair is directionally correct and did not introduce a
privacy or deletion leak in the probes. The release gate is not met because
the safe-summary carrier still mixes unrelated household threads. The next
bounded repair should focus on subject/thread binding for aggregate safe
carriers. No 60-checkpoint run and no GitHub push were performed.
