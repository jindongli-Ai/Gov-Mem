"""Flat, unescaped record lines for model output; typed storage stays internal.

Tabs delimit named fields. Values are literal text, never JSON or Python code.
A source containing a literal newline/tab needs a narrower contiguous excerpt.
"""
from __future__ import annotations

EVENT_FIELDS = {
    'SCENE': ('scenes', 'scene_id'), 'ENTITY': ('entities', 'entity_id'),
    'FACT': ('facts', 'fact_id'), 'RELATION': ('relations', 'relation_id'),
    'EVENT': ('governance_events', 'event_id'),
}
LIST_FIELDS = {'participant_principal_ids', 'aliases', 'included_scopes', 'excluded_scopes', 'permission_event_ids'}
FIELDS = {
    'SCENE': {'scene_type', 'canonical_name', 'participant_principal_ids', 'status'},
    'ENTITY': {'entity_type', 'canonical_name', 'aliases', 'scene_id'},
    'FACT': {'scene_id', 'subject_id', 'field_name', 'value', 'temporal_state', 'lifecycle'},
    'RELATION': {'relation_type', 'source_id', 'target_id', 'scene_id', 'role', 'action', 'resource_category', 'subject_id'},
    'EVENT': {'event_type', 'effect', 'issuer_principal_id', 'grantee_principal_id', 'action', 'resource_id',
              'resource_surface', 'scene_id', 'included_scopes', 'excluded_scopes', 'condition',
              'valid_from_turn_id', 'valid_until'},
    'CLAIM': {'slot', 'candidate_id', 'quote', 'value', 'delivery', 'requested', 'temporal_state',
              'subject_principal_id', 'restriction_kind', 'bind', 'value_start'},
    'KEEP': {'slot', 'candidate_id', 'bind', 'requested'},
    'SLOT': {'slot', 'required'},
    'SOURCE': {'turn_id', 'span'},
    'RESTRICTION': {'turn_id', 'span'},
}

LINE_CONTRACT = """
OUTPUT PROTOCOL: plain text, one record per line. No JSON, braces, quoting,
backslash escaping, markdown, or explanations. Separate fields with actual TAB
characters. Each data line is TYPE<TAB>local_id<TAB>field=value<TAB>field=value.
Values are literal original text (quotes and backslashes require NO escaping).
Do not use literal tabs/newlines inside a value; choose a narrower exact excerpt.
Lists use semicolons; omit optional empty fields. Never omit a restriction to
fit this format. IDs must be unique within this response, reuse graph IDs.

SLOT s1 slot=question field required=true
CLAIM c1 slot=question field candidate_id=candidate_0 value=exact excerpt delivery=exact bind=NONE
The spaces separating fields above stand for TABs. CLAIM requires bind: either
NONE (you checked and no supplied event applies) or semicolon-separated event
IDs whose scope applies. This is a semantic decision, not a lexical match.
Use applicable event IDs from existing graph OR new EVENT lines in this reply.
For a blocked claim add restriction_kind and at least one RESTRICTION line:
RESTRICTION c1 turn_id=visible turn span=exact restriction excerpt
Optional CLAIM: quote (defaults to value), requested=true|false,
temporal_state=current|historical|deleted, subject_principal_id, value_start.

When ingestion is present output EVENTS on its own line, even if no changes.
Graph record types/fields:
SCENE id scene_type=... canonical_name=... participant_principal_ids=id;id status=active
ENTITY id entity_type=... canonical_name=... aliases=... scene_id=...
FACT id field_name=... value=exact excerpt lifecycle=update|supersede|cancel|delete temporal_state=current|historical subject_id=... scene_id=...
RELATION id relation_type=... source_id=... target_id=... scene_id=...
For operational_duty RELATION include role, action, resource_category, subject_id.
EVENT id event_type=permission|lifecycle effect=allow|deny|revoke|require_permission|delete|update|cancel action=access|receive|share|disclose|use|delete resource_id=... resource_surface=exact resource phrase issuer_principal_id=... grantee_principal_id=...
Optional EVENT: scene_id, included_scopes, excluded_scopes, condition,
valid_from_turn_id, valid_until. Permission events require grantee_principal_id.
Each graph record requires at least one separate SOURCE line referencing its ID:
SOURCE id turn_id=visible turn span=exact contiguous source excerpt
Multiple SOURCE lines may resolve pronouns across visible turns.
Keep events query-independent. Record all new restrictions and lifecycle changes,
not every ordinary fact. Include roles/duties only when supported by source.

Finish with ACTION<TAB>answer|partial_answer|refuse|no_memory, then END on its
own line. END is mandatory: missing END means truncated/incomplete output.
"""


def parse_lines(text: str, *, ingestion: bool, shallow: bool = False) -> dict:
    events = {name: [] for name, _ in EVENT_FIELDS.values()}
    result = {'query_slots': [], 'claims': []}
    records, citations = {}, []
    ended = marked = False
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if ended:
            raise ValueError(f'Line {number}: content after END')
        if line == 'END':
            ended = True
            continue
        if line == 'EVENTS':
            if marked or not ingestion:
                raise ValueError('Unexpected or duplicate EVENTS marker')
            marked = True
            continue
        columns = line.split('\t')
        kind = columns[0]
        if shallow and kind in {'SCENE', 'ENTITY', 'FACT', 'RELATION'}:
            raise ValueError('Shallow memory accepts only EVENT graph records')
        if kind == 'KEEP' and not shallow:
            raise ValueError('KEEP requires shallow memory mode')
        if kind == 'ACTION' and len(columns) == 2:
            if 'answer_action' in result or columns[1] not in {'answer', 'partial_answer', 'refuse', 'no_memory'}:
                raise ValueError('Invalid/duplicate ACTION')
            result['answer_action'] = columns[1]
            continue
        if kind not in FIELDS or len(columns) < 3 or not columns[1]:
            raise ValueError(f'Line {number}: expected typed tab-separated record')
        identity, record = columns[1], {}
        for field in columns[2:]:
            key, sep, value = field.partition('=')
            if not sep or key not in FIELDS[kind] or key in record or not value:
                raise ValueError(f'Line {number}: invalid/duplicate {key} field')
            if key in LIST_FIELDS:
                record[key] = value.split(';')
            elif key in {'requested', 'required'}:
                if value not in {'true', 'false'}:
                    raise ValueError('Invalid boolean field')
                record[key] = value == 'true'
            elif key == 'value_start':
                record[key] = int(value)
            else:
                record[key] = value
        if kind in {'SOURCE', 'RESTRICTION'}:
            if set(record) != {'turn_id', 'span'}:
                raise ValueError('Citation requires turn_id and span')
            citations.append((kind, identity, record))
            continue
        if identity in records:
            raise ValueError('Duplicate record ID')
        records[identity] = (kind, record)
        if kind in EVENT_FIELDS:
            if not ingestion:
                raise ValueError('Graph records without ingestion')
            collection, id_field = EVENT_FIELDS[kind]
            record[id_field] = identity
            if kind == 'RELATION':
                record['attributes'] = {k: record.pop(k) for k in ('role', 'action', 'resource_category', 'subject_id') if k in record}
            events[collection].append(record)
        elif kind == 'SLOT':
            if not record.get('slot'):
                raise ValueError('SLOT requires slot')
            result['query_slots'].append(record)
        elif kind in {'CLAIM', 'KEEP'}:
            required = ('slot', 'candidate_id', 'bind') if kind == 'KEEP' else ('slot', 'candidate_id', 'value', 'delivery', 'bind')
            if not all(record.get(k) for k in required):
                raise ValueError('CLAIM requires slot, candidate_id, value, delivery and explicit bind')
            if kind == 'KEEP':
                record['_keep_candidate'] = True
            binding = record.pop('bind')
            record['permission_event_ids'] = [] if binding == 'NONE' else binding.split(';')
            record['claim_id'] = identity
            result['claims'].append(record)
    if not ended or 'answer_action' not in result or (ingestion and not marked):
        raise ValueError('Missing END, ACTION or EVENTS marker')
    for kind, identity, citation in citations:
        if identity not in records:
            raise ValueError('Citation references unknown record')
        target_kind, record = records[identity]
        if (kind == 'SOURCE' and target_kind not in EVENT_FIELDS) or (kind == 'RESTRICTION' and target_kind != 'CLAIM'):
            raise ValueError('Citation has wrong target type')
        record.setdefault('source_spans' if kind == 'SOURCE' else 'restriction_evidence', []).append(citation)
    for kind, record in records.values():
        if kind in EVENT_FIELDS and not record.get('source_spans'):
            raise ValueError('Graph record missing SOURCE line')
    if ingestion:
        result['events'] = events
    return result


def bind_claim_events(raw: dict, graph: dict, *, source_texts: dict[str, list[str]] | None = None) -> None:
    """Project explicit LLM event selections into canonical symbolic bindings.

    No token matching, default permit, or fabricated event identity. Source
    validation still runs before any new event is committed/enforced.
    """
    policies = {}
    def visit(obj):
        if isinstance(obj, dict):
            if obj.get('event_id') and obj.get('resource_id') and ('effect' in obj or 'decision' in obj):
                old = policies.get(obj['event_id'])
                if old and any(old.get(k) != obj.get(k) for k in ('resource_id', 'action', 'scene_id')):
                    raise ValueError('Conflicting event binding identity')
                policies[obj['event_id']] = obj
            for value in obj.values(): visit(value)
        elif isinstance(obj, list):
            for value in obj: visit(value)
    visit(graph)
    visit(raw.get('events', {}))
    for claim in raw['claims']:
        ids = claim.get('permission_event_ids', [])
        if not ids:
            continue
        unknown = [identity for identity in ids if identity not in policies]
        if any(identity not in (source_texts or {}) for identity in unknown):
            raise ValueError('CLAIM bind references unknown event')
        # A visible source-turn citation is provenance, not a fabricated policy
        # edge. Keep the two types separate. Never infer a permission event by
        # word overlap or merely because an event mentions the same turn.
        if unknown:
            claim["governance_source_turn_ids"] = list(dict.fromkeys(unknown))
        ids = [identity for identity in ids if identity in policies]
        claim["permission_event_ids"] = ids
        if not ids:
            for field in ("resource_id", "action", "scene_id"):
                claim.pop(field, None)
            continue
        selected = [policies[identity] for identity in ids]
        keys = {(p.get('resource_id'), p.get('action') or 'access', p.get('scene_id')) for p in selected}
        if len(keys) != 1:
            if claim.get("delivery") == "block":
                # A grounded language denial may cite several restrictions.
                # Do not invent one canonical resource or claim a hard veto;
                # all selected IDs must still exist and denial citations are
                # validated separately before this claim can be accepted.
                for field in ("resource_id", "action", "scene_id"):
                    claim.pop(field, None)
                continue
            raise ValueError('CLAIM binds incompatible resources/actions/scenes; split the claim')
        claim['resource_id'], claim['action'], claim['scene_id'] = keys.pop()
