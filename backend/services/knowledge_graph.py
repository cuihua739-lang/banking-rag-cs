"""NetworkX knowledge graph for banking entities and relationships."""

import json
import os
from collections import defaultdict
import networkx as nx
from config.settings import KG_FILE, KG_MAX_HOPS, KG_MAX_HOPS_DEEP

# Entity types
ENTITY_TYPES = [
    "Product", "ProductVariant", "Feature", "Policy", "Term",
    "Condition", "Qualifier", "Channel", "Action", "Amount",
]

# Relationship types
RELATION_TYPES = [
    "HAS_FEATURE", "HAS_VARIANT", "HAS_POLICY", "APPLIES_TO",
    "REQUIRES", "CONSTRAINED_BY", "AVAILABLE_VIA", "USES_TERM",
    "RELATES_TO", "SIMILAR_TO", "PREREQUISITE", "SUPERIOR_TO",
    "MENTIONS",
]

_graph: nx.DiGraph | None = None
_entity_to_chunks: dict[str, set[str]] = defaultdict(set)


def _get_graph() -> nx.DiGraph:
    global _graph
    if _graph is None:
        _graph = nx.DiGraph()
        _load()
    return _graph


def _load() -> None:
    if not os.path.exists(KG_FILE):
        return
    with open(KG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Restore graph from JSON
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    for node in nodes:
        _graph.add_node(
            node["id"],
            type=node.get("type", ""),
            name=node.get("name", ""),
            aliases=node.get("aliases", []),
        )
    for edge in edges:
        _graph.add_edge(
            edge["source"],
            edge["target"],
            relation=edge.get("relation", ""),
        )
    # Restore entity-to-chunk mapping
    emap = data.get("entity_to_chunks", {})
    for entity_id, chunk_ids in emap.items():
        _entity_to_chunks[entity_id] = set(chunk_ids)


def save() -> None:
    g = _get_graph()
    data = {
        "nodes": [
            {
                "id": n,
                "type": g.nodes[n].get("type", ""),
                "name": g.nodes[n].get("name", ""),
                "aliases": g.nodes[n].get("aliases", []),
            }
            for n in g.nodes
        ],
        "edges": [
            {"source": u, "target": v, "relation": g.edges[u, v].get("relation", "")}
            for u, v in g.edges
        ],
        "entity_to_chunks": {k: list(v) for k, v in _entity_to_chunks.items()},
    }
    os.makedirs(os.path.dirname(KG_FILE), exist_ok=True)
    with open(KG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_document(
    doc_id: str,
    chunk_id: str,
    text: str,
    entities: list[dict],
    relations: list[dict],
) -> None:
    """Add a chunk's entities and relations to the knowledge graph."""
    g = _get_graph()

    for ent in entities:
        ent_id = ent.get("id", ent.get("name", ""))
        ent_name = ent.get("name", ent_id)
        ent_type = ent.get("type", "Term")
        aliases = ent.get("aliases", [])

        if ent_id not in g:
            g.add_node(ent_id, type=ent_type, name=ent_name, aliases=aliases)

        # Also add aliases as graph references
        for alias in aliases:
            if alias not in g:
                g.add_node(alias, type="Alias", name=alias, aliases=[])
            g.add_edge(alias, ent_id, relation="ALIAS_OF")

        _entity_to_chunks[ent_id].add(chunk_id)

    for rel in relations:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rtype = rel.get("relation", "RELATES_TO")
        if src and tgt:
            if src not in g:
                g.add_node(src, type="Term", name=src, aliases=[])
            if tgt not in g:
                g.add_node(tgt, type="Term", name=tgt, aliases=[])
            g.add_edge(src, tgt, relation=rtype)


def _find_matching_entities(query_text: str) -> list[str]:
    """Find graph nodes whose name/alias appears in the query."""
    g = _get_graph()
    matched = []
    query_lower = query_text.lower()
    for node_id in g.nodes:
        node = g.nodes[node_id]
        name = node.get("name", node_id)
        aliases = node.get("aliases", [])
        candidates = [name] + aliases
        for cand in candidates:
            if cand.lower() in query_lower or query_lower in cand.lower():
                matched.append(node_id)
                break
    return matched


def traverse(
    query_text: str,
    max_hops: int = KG_MAX_HOPS,
) -> list[dict]:
    """Traverse graph from matched entities and collect related chunks."""
    matched = _find_matching_entities(query_text)
    if not matched:
        return []

    g = _get_graph()
    visited_nodes: set[str] = set()
    collected_chunks: set[str] = set()
    node_scores: dict[str, float] = {}

    # BFS from each matched entity
    from collections import deque

    queue: deque = deque()
    for m in matched:
        queue.append((m, 0, 1.0))  # (node, distance, score)

    while queue:
        node_id, dist, score = queue.popleft()
        if node_id in visited_nodes or dist > max_hops:
            continue
        visited_nodes.add(node_id)

        # Collect chunks linked to this node
        node_scores[node_id] = max(node_scores.get(node_id, 0), score)
        for cid in _entity_to_chunks.get(node_id, set()):
            collected_chunks.add(cid)

        # Expand to neighbors
        if dist < max_hops:
            for _, neighbor in g.out_edges(node_id):
                rel = g.edges[node_id, neighbor].get("relation", "")
                decay = 0.7
                if rel in ("HAS_FEATURE", "APPLIES_TO", "HAS_POLICY"):
                    decay = 0.85
                elif rel in ("SIMILAR_TO", "RELATES_TO"):
                    decay = 0.6
                queue.append((neighbor, dist + 1, score * decay))
            # Also traverse inbound edges
            for predecessor, _ in g.in_edges(node_id):
                rel = g.edges[predecessor, node_id].get("relation", "")
                queue.append((predecessor, dist + 1, score * 0.5))

    # Build result list with metadata
    results = []
    for cid in collected_chunks:
        # Find best matching entity score for this chunk
        best_score = 0.0
        matched_entity = ""
        for ent_id, chunks in _entity_to_chunks.items():
            if cid in chunks:
                s = node_scores.get(ent_id, 0.3)
                if s > best_score:
                    best_score = s
                    matched_entity = ent_id

        results.append({
            "chunk_id": cid,
            "score": round(best_score, 4),
            "source": "kg",
            "matched_entity": matched_entity,
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def delete_all() -> None:
    global _graph, _entity_to_chunks
    _graph = nx.DiGraph()
    _entity_to_chunks = defaultdict(set)
    if os.path.exists(KG_FILE):
        os.remove(KG_FILE)


def stats() -> dict:
    g = _get_graph()
    type_counts: dict[str, int] = defaultdict(int)
    for n in g.nodes:
        t = g.nodes[n].get("type", "Unknown")
        type_counts[t] += 1
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "entity_types": dict(type_counts),
    }
