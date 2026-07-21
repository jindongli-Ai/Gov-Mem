from __future__ import annotations

from collections import defaultdict
from hashlib import md5
import re
from typing import Any

from gov_mem.graph.governed_graph import GovernedMemoryGraph, GraphEdge, GraphNode
from gov_mem.memory.governed_atom import GovernedMemoryAtom


class GovernedGraphBuilder:
    def build(
        self,
        *,
        graph_id: str,
        instance_id: str | None,
        atoms: list[GovernedMemoryAtom],
        principal_relation_ledger: dict[str, Any] | None = None,
    ) -> GovernedMemoryGraph:
        graph = GovernedMemoryGraph(
            graph_id=graph_id,
            instance_id=instance_id,
            metadata={"atom_count": len(atoms)},
        )
        semantic_nodes_by_atom: dict[str, str] = {}
        semantic_records: list[dict[str, Any]] = []
        role_node_ids_by_scope: dict[str, str] = {}
        slot_node_ids_by_key: dict[str, list[str]] = defaultdict(list)
        slot_records_by_key: dict[str, list[tuple[str, GovernedMemoryAtom]]] = defaultdict(list)
        latest_slot_node_by_identity: dict[str, tuple[str, GovernedMemoryAtom]] = {}
        # A source classified purely as policy has no answerable fact payload.
        # Mixed sources are deliberately excluded here: record decomposition
        # can retain an independent active fact alongside a policy statement.
        policy_source_message_ids = {
            str(message_id)
            for atom in atoms
            if str((atom.provenance or {}).get("source_role") or "") == "policy"
            for message_id in list((atom.provenance or {}).get("source_message_ids") or [])
            if str(message_id)
        }
        ordered_atoms = sorted(
            atoms,
            key=lambda atom: (
                atom.source_turn if atom.source_turn is not None else 10**9,
                atom.timestamp or "",
                atom.atom_id,
            ),
        )

        for atom in ordered_atoms:
            memory_node = self._make_memory_atom_node(atom)
            graph.add_node(memory_node)
            semantic_node = self._make_semantic_node(atom)
            graph.add_node(semantic_node)
            semantic_nodes_by_atom[atom.atom_id] = semantic_node.node_id
            semantic_records.append({"atom": atom, "node_id": semantic_node.node_id})
            graph.add_edge(self._make_edge("derived_from", semantic_node.node_id, memory_node.node_id, atom, {}))

            for principal_kind, principal_value in {
                "owner": atom.owner_id,
                "speaker": atom.speaker_id,
                "subject": atom.subject_id,
            }.items():
                if not principal_value:
                    continue
                principal_node = self._make_principal_node(principal_value, atom=atom)
                graph.add_node(principal_node)
                edge_type = "owns" if principal_kind == "owner" else "mentions"
                if principal_kind == "subject":
                    edge_type = "relates_to"
                graph.add_edge(
                    self._make_edge(edge_type, principal_node.node_id, semantic_node.node_id, atom, {"principal_kind": principal_kind})
                )

            # Claim subjects are dynamically extracted from this instance's
            # source text. They improve within-case entity disambiguation but
            # are never persisted as a cross-case vocabulary or rule.
            for claim in list((atom.provenance or {}).get("grounded_claims") or []):
                if not isinstance(claim, dict):
                    continue
                subject_surface = str(claim.get("subject_span") or "").strip()
                if not subject_surface:
                    continue
                principal_node = self._make_principal_node(subject_surface, atom=atom)
                graph.add_node(principal_node)
                graph.add_edge(self._make_edge(
                    "claim_subject", principal_node.node_id, semantic_node.node_id, atom,
                    {"property_label": str(claim.get("property_label") or "")},
                ))

            for scope in atom.access_scope:
                role_node = self._make_role_node(scope, atom=atom)
                graph.add_node(role_node)
                role_node_ids_by_scope[scope] = role_node.node_id
                if atom.owner_id:
                    owner_node = self._make_principal_node(atom.owner_id, atom=atom)
                    graph.add_node(owner_node)
                    graph.add_edge(self._make_edge("has_role", owner_node.node_id, role_node.node_id, atom, {"scope": scope}))

            # Only policy-derived atoms suppress observed values. A factual
            # atom may carry a policy binding that governs its release while
            # still providing source-grounded slots for Stage 3.
            observed_slots = (
                {}
                if (
                    atom.atom_type in {"policy_atom", "permission_atom"}
                    or (
                        bool((atom.provenance or {}).get("policy_binding"))
                        and str((atom.provenance or {}).get("source_role") or "").lower()
                        not in {"factual", "mixed"}
                    )
                    or bool(
                        set(str(value) for value in list((atom.provenance or {}).get("source_message_ids") or []))
                        & policy_source_message_ids
                    )
                )
                else dict(atom.slots or {})
            )
            for slot_name, slot_value in observed_slots.items():
                slot_node = self._make_slot_node(
                    slot_name,
                    slot_value,
                    atom=atom,
                    extra_attributes=self._claim_slot_attributes(
                        atom=atom,
                        slot_name=str(slot_name),
                        slot_value=str(slot_value),
                    ),
                )
                graph.add_node(slot_node)
                slot_node_ids_by_key[str(slot_name)].append(slot_node.node_id)
                slot_records_by_key[str(slot_name)].append((slot_node.node_id, atom))
                graph.add_edge(self._make_edge("has_slot", semantic_node.node_id, slot_node.node_id, atom, {"slot_name": slot_name}))
                slot_identity = self._slot_identity(atom=atom, slot_name=str(slot_name))
                prior = latest_slot_node_by_identity.get(slot_identity)
                if prior is not None and prior[0] != slot_node.node_id:
                    graph.add_edge(
                        self._make_edge(
                            "version_precedes",
                            prior[0],
                            slot_node.node_id,
                            atom,
                            {"slot_name": str(slot_name), "slot_identity": slot_identity},
                        )
                    )
                    if atom.lifecycle in {"superseded", "active"} and self._is_newer(atom, prior[1]):
                        graph.add_edge(
                            self._make_edge(
                                "supersedes_slot",
                                slot_node.node_id,
                                prior[0],
                                atom,
                                {"slot_name": str(slot_name), "slot_identity": slot_identity},
                            )
                        )
                if prior is None or self._is_newer(atom, prior[1]):
                    latest_slot_node_by_identity[slot_identity] = (slot_node.node_id, atom)

            # Preserve the ingestion model's grounded claim subject as an
            # evidence-local typed value. Some natural-language claims put
            # the answer value in subject position while the extracted slot
            # names only the predicate. This representation stays open-schema
            # and source-grounded; downstream alignment still chooses the
            # closed SlotNode and validates its verbatim source span.
            for claim_index, claim in enumerate(list((atom.provenance or {}).get("grounded_claims") or [])):
                if not isinstance(claim, dict):
                    continue
                subject_surface = str(claim.get("subject_span") or "").strip()
                property_label = str(claim.get("property_label") or "").strip()
                if not subject_surface or subject_surface not in str(atom.text or ""):
                    continue
                claim_slot_name = "claim_subject_value"
                claim_slot_node = self._make_slot_node(
                    claim_slot_name,
                    subject_surface,
                    atom=atom,
                    extra_attributes={
                        "slot_role": "claim_subject_value",
                        "claim_index": claim_index,
                        "claim_property_label": property_label,
                        "claim_span": str(claim.get("claim_span") or ""),
                    },
                    identity_suffix=f"claim_{claim_index}",
                )
                graph.add_node(claim_slot_node)
                slot_node_ids_by_key[claim_slot_name].append(claim_slot_node.node_id)
                slot_records_by_key[claim_slot_name].append((claim_slot_node.node_id, atom))
                graph.add_edge(self._make_edge(
                    "has_slot", semantic_node.node_id, claim_slot_node.node_id, atom,
                    {"slot_name": claim_slot_name, "slot_role": "claim_subject_value", "claim_property_label": property_label},
                ))

        # Bind policy only after all fact slots exist. Authorization therefore
        # does not depend on whether a policy sentence appeared before its fact.
        for atom in ordered_atoms:
            binding = dict((atom.provenance or {}).get("policy_binding") or {})
            if not binding:
                continue
            binding_scopes = list(binding.get("scopes") or [])
            binding_slots = list(binding.get("slots") or [])
            binding_effect = str(binding.get("effect") or "").lower()
            semantic_node_id = semantic_nodes_by_atom.get(atom.atom_id)
            if not semantic_node_id or binding_effect not in {"allow", "deny", "require_permission"}:
                continue
            for scope in binding_scopes:
                role_node_id = role_node_ids_by_scope.get(scope)
                if not role_node_id:
                    continue
                graph.add_edge(self._make_edge("applies_to", semantic_node_id, role_node_id, atom, {"scope": scope}))
                for slot_name in binding_slots:
                    # Do not create an empty fact slot from policy prose: an
                    # authorization path must terminate in observed evidence.
                    for slot_target, fact_atom in slot_records_by_key.get(str(slot_name), []):
                        if not self._policy_applies_to_fact(policy_atom=atom, fact_atom=fact_atom):
                            continue
                        policy_edge = "denies" if binding_effect == "deny" else "allows"
                        if binding_effect == "require_permission":
                            graph.add_edge(self._make_edge("requires_permission", role_node_id, slot_target, atom, {"slot_name": slot_name}))
                        else:
                            graph.add_edge(self._make_edge(policy_edge, role_node_id, slot_target, atom, {"slot_name": slot_name}))
                            if binding_effect == "allow" and bool(binding.get("minimal_projection")):
                                graph.add_edge(self._make_edge(
                                    "allows_projection",
                                    role_node_id,
                                    slot_target,
                                    atom,
                                    {"slot_name": slot_name, "support_spans": list(binding.get("support_spans") or [])},
                                ))

        self._attach_lifecycle_edges(graph=graph, semantic_records=semantic_records)
        self._attach_principal_relation_ledger(
            graph=graph,
            ledger=dict(principal_relation_ledger or {}),
        )
        graph.metadata["node_count"] = len(graph.nodes)
        graph.metadata["edge_count"] = len(graph.edges)
        return graph

    @staticmethod
    def _attach_principal_relation_ledger(
        *, graph: GovernedMemoryGraph, ledger: dict[str, Any]
    ) -> None:
        """Materialize only the proven, current episode-local relation proof."""
        requester_id = str(ledger.get("requester_id") or "").strip()
        owner_id = str(ledger.get("owner_id") or "").strip()
        relation = str(ledger.get("effective_relation") or "").strip()
        status = str(ledger.get("effective_status") or "").strip()
        if not requester_id or not owner_id or not relation or status != "proven":
            return
        record = next((
            row for row in list(ledger.get("records") or [])
            if isinstance(row, dict)
            and str(row.get("requester_id") or "") == requester_id
            and str(row.get("owner_id") or "") == owner_id
            and str(row.get("relation") or "") == relation
            and str(row.get("status") or "") == "proven"
            and str(row.get("direction") or "") == "requester_to_owner"
        ), None)
        if record is None:
            return
        supports = [
            dict(item) for item in list(record.get("supports") or [])
            if isinstance(item, dict) and str(item.get("message_id") or "").strip()
            and str(item.get("source_span") or "").strip()
        ]
        if relation != "owner" and not supports:
            return
        requester_node_id = f"principal::{requester_id}"
        owner_node_id = f"principal::{owner_id}"
        relation_node_id = f"relation::{requester_id}::{owner_id}::{relation}"
        provenance = {
            "source_message_ids": [str(item["message_id"]) for item in supports],
            "support_spans": [str(item["source_span"]) for item in supports],
            "evidence_kinds": [str(item.get("evidence_kind") or "") for item in supports],
            "relation_ledger": True,
        }
        graph.add_node(GraphNode(requester_node_id, "PrincipalNode", requester_id))
        graph.add_node(GraphNode(owner_node_id, "PrincipalNode", owner_id))
        graph.add_node(GraphNode(
            relation_node_id,
            "PrincipalRelationNode",
            relation,
            attributes={
                "requester_id": requester_id,
                "owner_id": owner_id,
                "relation": relation,
                "status": "proven",
                "direction": "requester_to_owner",
            },
            provenance=provenance,
        ))
        for edge_type, source_id, target_id in (
            ("has_relation", requester_node_id, relation_node_id),
            ("relation_owner", relation_node_id, owner_node_id),
        ):
            edge_key = f"{edge_type}:{source_id}:{target_id}"
            graph.add_edge(GraphEdge(
                edge_id=md5(edge_key.encode("utf-8")).hexdigest()[:16],
                edge_type=edge_type,
                source_id=source_id,
                target_id=target_id,
                provenance=dict(provenance),
            ))

    @staticmethod
    def _slot_identity(*, atom: GovernedMemoryAtom, slot_name: str) -> str:
        owner = str(atom.owner_id or "unknown_owner")
        subject = str(atom.subject_id or "general_subject")
        frame_type = str((atom.provenance or {}).get("frame_type") or atom.atom_type)
        return f"{owner}::{subject}::{frame_type}::{slot_name}"

    @staticmethod
    def _policy_applies_to_fact(
        *, policy_atom: GovernedMemoryAtom, fact_atom: GovernedMemoryAtom
    ) -> bool:
        """Prevent a policy for one principal from authorizing another's fact."""
        if fact_atom.atom_type in {"policy_atom", "permission_atom"}:
            return False
        policy_owner = str(policy_atom.owner_id or "").strip()
        fact_owner = str(fact_atom.owner_id or "").strip()
        # Cross-principal authorization is unsafe without an explicit relation
        # edge, which this graph does not yet model as a delegation certificate.
        return bool(policy_owner and fact_owner and policy_owner == fact_owner)

    @staticmethod
    def _is_newer(candidate: GovernedMemoryAtom, existing: GovernedMemoryAtom) -> bool:
        candidate_time = str(candidate.timestamp or "")
        existing_time = str(existing.timestamp or "")
        if candidate_time != existing_time:
            return candidate_time > existing_time
        candidate_turn = int(candidate.source_turn or -1)
        existing_turn = int(existing.source_turn or -1)
        if candidate_turn != existing_turn:
            return candidate_turn > existing_turn
        return float(candidate.confidence or 0.0) >= float(existing.confidence or 0.0)

    def _attach_lifecycle_edges(self, *, graph: GovernedMemoryGraph, semantic_records: list[dict[str, Any]]) -> None:
        prior_records: list[dict[str, Any]] = []
        for record in semantic_records:
            atom: GovernedMemoryAtom = record["atom"]
            node_id = str(record["node_id"])
            if atom.atom_type == "deletion_atom":
                for target in self._match_prior_targets(atom, prior_records):
                    graph.add_edge(self._make_edge("deletes", node_id, target["node_id"], atom, {"target_atom_id": target["atom"].atom_id}))
            if atom.atom_type == "supersession_atom":
                for target in self._match_prior_targets(atom, prior_records):
                    graph.add_edge(self._make_edge("supersedes", node_id, target["node_id"], atom, {"target_atom_id": target["atom"].atom_id}))
            prior_records.append(record)

    def _match_prior_targets(self, atom: GovernedMemoryAtom, prior_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        slot_names = set((atom.slots or {}).keys())
        related = set(atom.related_entities or [])
        matches: list[dict[str, Any]] = []
        for record in reversed(prior_records):
            target_atom: GovernedMemoryAtom = record["atom"]
            if target_atom.atom_type in {"deletion_atom", "supersession_atom", "policy_atom", "permission_atom"}:
                continue
            same_owner = atom.owner_id and atom.owner_id == target_atom.owner_id
            shared_slots = slot_names & set((target_atom.slots or {}).keys())
            shared_entities = related & set(target_atom.related_entities or [])
            text_overlap = self._text_overlap(atom.text, target_atom.text)
            if same_owner and (shared_slots or shared_entities or text_overlap):
                matches.append(record)
            if len(matches) >= 2:
                break
        return matches

    @staticmethod
    def _text_overlap(left: str, right: str) -> bool:
        left_tokens = {token for token in re.findall(r"[a-z0-9_]+", left.lower()) if len(token) >= 4}
        right_tokens = {token for token in re.findall(r"[a-z0-9_]+", right.lower()) if len(token) >= 4}
        return len(left_tokens & right_tokens) >= 2

    @staticmethod
    def _make_memory_atom_node(atom: GovernedMemoryAtom) -> GraphNode:
        return GraphNode(
            node_id=f"memory_atom::{atom.atom_id}",
            node_type="MemoryAtomNode",
            label=atom.text,
            attributes={"atom_type": atom.atom_type, "slots": dict(atom.slots or {})},
            provenance=_provenance(atom),
        )

    @staticmethod
    def _make_semantic_node(atom: GovernedMemoryAtom) -> GraphNode:
        type_map = {
            "fact_atom": "FactNode",
            "policy_atom": "PolicyNode",
            "permission_atom": "PolicyNode",
            "deletion_atom": "DeletionNode",
            "event_atom": "EventNode",
            "role_atom": "RoleNode",
            "relation_atom": "FactNode",
            "supersession_atom": "FactNode",
        }
        node_type = type_map.get(atom.atom_type, "FactNode")
        return GraphNode(
            node_id=f"semantic::{atom.atom_id}",
            node_type=node_type,
            label=atom.text,
            attributes={
                "atom_type": atom.atom_type,
                "owner_id": atom.owner_id,
                "subject_id": atom.subject_id,
                "lifecycle": atom.lifecycle,
                "sensitivity": atom.sensitivity,
                "access_scope": list(atom.access_scope or []),
                "slots": dict(atom.slots or {}),
            },
            provenance=_provenance(atom),
        )

    @staticmethod
    def _make_principal_node(principal_id: str, *, atom: GovernedMemoryAtom) -> GraphNode:
        return GraphNode(
            node_id=f"principal::{principal_id}",
            node_type="PrincipalNode",
            label=principal_id,
            attributes={},
            provenance=_provenance(atom),
        )

    @staticmethod
    def _make_role_node(scope: str, *, atom: GovernedMemoryAtom) -> GraphNode:
        return GraphNode(
            node_id=f"role::{scope}",
            node_type="RoleNode",
            label=scope,
            attributes={"scope": scope},
            provenance=_provenance(atom),
        )

    @staticmethod
    def _make_slot_node(
        slot_name: str,
        slot_value: Any,
        *,
        atom: GovernedMemoryAtom,
        extra_attributes: dict[str, Any] | None = None,
        identity_suffix: str = "",
    ) -> GraphNode:
        slot_value_text = "" if slot_value is None else str(slot_value)
        # A slot occurrence is evidence-local.  Reusing a node for equal text
        # from different atoms would silently merge provenance and could let a
        # certificate cite the wrong fact occurrence.
        suffix = f"::{identity_suffix}" if identity_suffix else ""
        node_id = f"slot::{atom.atom_id}::{slot_name}{suffix}::{md5(slot_value_text.encode('utf-8')).hexdigest()[:10]}"
        attributes = {"slot_name": slot_name, "slot_value": slot_value_text}
        attributes.update(dict(extra_attributes or {}))
        return GraphNode(
            node_id=node_id,
            node_type="SlotNode",
            label=f"{slot_name}={slot_value_text}".strip("="),
            attributes=attributes,
            provenance=_provenance(atom),
        )

    @staticmethod
    def _claim_slot_attributes(
        *, atom: GovernedMemoryAtom, slot_name: str, slot_value: str
    ) -> dict[str, Any]:
        """Attach source-local claim provenance to an observed SlotNode."""
        matches: list[dict[str, Any]] = []
        for claim in list((atom.provenance or {}).get("grounded_claims") or []):
            if not isinstance(claim, dict):
                continue
            value_span = str(claim.get("value_span") or "").strip()
            claim_span = str(claim.get("claim_span") or "").strip()
            if (
                not value_span
                or not claim_span
                or value_span != str(slot_value).strip()
                or value_span not in claim_span
                or claim_span not in str(atom.text or "")
            ):
                continue
            matches.append(claim)
        if not matches:
            return {}
        return {
            "slot_role": "claim_value",
            "claim_property_labels": list(dict.fromkeys(
                str(claim.get("property_label") or "").strip()
                for claim in matches
                if str(claim.get("property_label") or "").strip()
            )),
            "claim_spans": list(dict.fromkeys(
                str(claim.get("claim_span") or "").strip()
                for claim in matches
                if str(claim.get("claim_span") or "").strip()
            )),
            "claim_value_spans": list(dict.fromkeys(
                str(claim.get("value_span") or "").strip()
                for claim in matches
                if str(claim.get("value_span") or "").strip()
            )),
        }

    @staticmethod
    def _make_edge(edge_type: str, source_id: str, target_id: str, atom: GovernedMemoryAtom, attributes: dict[str, Any]) -> GraphEdge:
        edge_key = f"{edge_type}:{source_id}:{target_id}"
        return GraphEdge(
            edge_id=md5(edge_key.encode("utf-8")).hexdigest()[:16],
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            attributes=dict(attributes or {}),
            provenance=_provenance(atom),
        )

def _provenance(atom: GovernedMemoryAtom) -> dict[str, Any]:
    return {
        "source_turn": atom.source_turn,
        "speaker": atom.speaker_id,
        "timestamp": atom.timestamp,
        "source_atom_id": atom.atom_id,
        "source_memory_id": (atom.provenance or {}).get("source_memory_id"),
        # Retrieval selects source turns before one turn is expanded into one
        # or more governed atoms. Keep that immutable provenance on every
        # graph occurrence so downstream alignment can stay evidence-local
        # without requiring an accidental atom-id match.
        "source_message_ids": list((atom.provenance or {}).get("source_message_ids") or []),
        "evidence_span": (atom.provenance or {}).get("evidence_span"),
        "confidence": atom.confidence,
    }
