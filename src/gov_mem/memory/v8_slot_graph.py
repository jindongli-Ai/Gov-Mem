"""Transactional V8 adapter for the source-grounded slot graph.

Graph proposals retrieve original evidence; permission decisions still require
V8's source validation, semantic binding and deterministic symbolic critic.
"""
from copy import deepcopy
from gov_mem.memory.governed_slot_graph import build_memory_governed_slot_graph, graph_retrieval_evidence


class V8SlotGraphIndex:
    def __init__(self):
        self.cache = {}

    def retrieve(self, *, conversation_id, chunks, question, llm_client, model_name, config):
        # Compare contents as well as IDs: a changed/shorter prefix must not
        # inherit slots extracted from a different or future history.
        visible = {c.chunk_id: c.text for c in chunks}
        previous, processed = self.cache.get(conversation_id, (None, {}))
        reset = any(visible.get(k) != v for k, v in processed.items())
        if reset:
            previous, processed = None, {}
        build = build_memory_governed_slot_graph(
            chunks=chunks, llm_client=llm_client, model_name=model_name,
            config=config, existing_graph=deepcopy(previous), processed_chunk_ids=set(processed))
        # The graph is an auxiliary retrieval channel. If its provider
        # response is malformed, retain the committed prefix audit and return
        # no graph candidates so dense retrieval and symbolic governance can
        # still complete the checkpoint. Authorization never depends on this
        # optional channel.
        if build.parse_failure or build.graph.audit.get('batch_failures'):
            audit = {'enabled': True, 'cache_reset': reset, 'llm_calls': build.llm_calls,
                     'graph': build.graph.to_dict(), 'matched_entities': [],
                     'retrieved_source_ids': [], 'fallback': 'dense_only_graph_failure',
                     'audit_status': 'insufficient_evidence',
                     'error': build.error or build.graph.audit.get('batch_failures'),
                     'authorization': 'proposals_only; V8 semantic binding and symbolic critic enforce release'}
            return [], audit
        self.cache[conversation_id] = (build.graph, visible)
        cfg = config.get('memory_governed_slot_graph', {})
        lists = build.graph.retrieve_entity_lists(question=question, top_k=int(cfg.get('top_k', 20)))
        proposals = {}
        for entity in lists:
            for relation in entity['relations']:
                cid = relation['source_chunk_id']
                proposals.setdefault(cid, {'chunk_id': cid, 'score': entity['score'], 'node_ids': []})
        evidence = graph_retrieval_evidence(retrieved=list(proposals.values()), chunk_by_id={c.chunk_id: c for c in chunks})
        return evidence, {'enabled': True, 'cache_reset': reset, 'llm_calls': build.llm_calls,
                          'graph': build.graph.to_dict(), 'matched_entities': lists,
                          'retrieved_source_ids': list(proposals),
                          'audit_status': 'complete',
                          'authorization': 'proposals_only; V8 semantic binding and symbolic critic enforce release'}
