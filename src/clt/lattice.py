from collections import deque

from .certification import assert_fresh
from .contexts import canonical_messages
from .io import canonical
from .verification import proof_satisfied


def compile_lattice(trace, certificate, max_states=100000):
    assert_fresh(certificate, trace)
    actions = certificate["actions"]
    index = {action["id"]: offset for offset, action in enumerate(actions)}
    requirements = [sum(1 << index[parent] for parent in certificate["dependencies"][action["id"]])
                    for action in actions]
    nodes, edges, masks = [], [], {0: 0}
    pending = deque([0])
    while pending:
        mask = pending.popleft()
        source = masks[mask]
        active = set(certificate["initial_ids"]) | {action["id"] for offset, action in enumerate(actions) if mask >> offset & 1}
        terminal = proof_satisfied(certificate["proof"], active)
        nodes.append({"id": source, "mask": mask, "terminal": terminal,
                      "messages": canonical_messages(trace, certificate["goal"], active)})
        if terminal:
            continue
        for offset, action in enumerate(actions):
            if mask >> offset & 1 or requirements[offset] & mask != requirements[offset]:
                continue
            child = mask | 1 << offset
            if child not in masks:
                if len(masks) >= max_states:
                    raise ValueError("Lattice exceeds max_states; increase the explicit compilation limit")
                masks[child] = len(masks)
                pending.append(child)
            edges.append({"id": len(edges), "source": source, "target": masks[child],
                          "action_id": action["id"], "serialized": action["serialized"],
                          "cost": action["latency_seconds"], "multiplicity": 1})
    nodes.sort(key=lambda node: node["id"])
    graph = {"nodes": nodes, "edges": edges, "root": 0,
             "verification_action": certificate["goal"].get("verification_action"),
             "certificate_id": certificate["certificate_id"], "task_id": trace["task_id"],
             "checkpoint_verified": certificate["verification_observed"],
             "data_kind": trace["data_kind"], "original_contexts": len(nodes)}
    graph["already_satisfied"] = nodes[0]["terminal"]
    graph["has_completion"] = any(node["terminal"] for node in nodes) and not graph["already_satisfied"]
    return graph


def topological_order(graph):
    outgoing = [[] for _ in graph["nodes"]]
    incoming = [0] * len(outgoing)
    for edge in graph["edges"]:
        outgoing[edge["source"]].append(edge)
        incoming[edge["target"]] += 1
    pending = deque(index for index, count in enumerate(incoming) if count == 0)
    order = []
    while pending:
        source = pending.popleft()
        order.append(source)
        for edge in outgoing[source]:
            incoming[edge["target"]] -= 1
            if incoming[edge["target"]] == 0:
                pending.append(edge["target"])
    if len(order) != len(outgoing):
        raise ValueError("The execution graph must be acyclic")
    return order, outgoing


def compress_lattice(graph):
    """Symbolic refinement preserves equality throughout parameter updates."""
    if "original_to_group" in graph:
        return graph
    order, outgoing = topological_order(graph)
    groups, signatures, members = {}, {}, []
    for source in reversed(order):
        node = graph["nodes"][source]
        transitions = sorted((edge["serialized"], edge["cost"], groups[edge["target"]], edge["multiplicity"])
                             for edge in outgoing[source])
        signature = canonical([node["terminal"], node["messages"], graph["verification_action"], transitions])
        if signature not in signatures:
            signatures[signature] = len(members)
            members.append([])
        group = signatures[signature]
        groups[source] = group
        members[group].append(source)
    nodes, edges = [], []
    for group, original in enumerate(members):
        representative = graph["nodes"][original[0]]
        nodes.append({**representative, "id": group, "members": original})
        # Outgoing distributions belong to one state. Summing all member states'
        # distributions would multiply completion mass by the group size.
        merged = {}
        for edge in outgoing[original[0]]:
            key = (edge["serialized"], edge["cost"], groups[edge["target"]])
            if key not in merged:
                merged[key] = {**edge, "source": group, "target": key[2], "multiplicity": 0}
            merged[key]["multiplicity"] += edge["multiplicity"]
        for edge in merged.values():
            edges.append({**edge, "id": len(edges)})
    return {**graph, "nodes": nodes, "edges": edges, "root": groups[graph["root"]],
            "original_to_group": groups}


def path_count(graph):
    order, outgoing = topological_order(graph)
    counts = [0] * len(order)
    for source in reversed(order):
        counts[source] = (1 if graph["nodes"][source]["terminal"] else
                          sum(edge["multiplicity"] * counts[edge["target"]] for edge in outgoing[source]))
    return counts[graph["root"]]
