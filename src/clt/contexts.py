from .io import canonical, digest

CONTEXT_FORMAT_VERSION = 2


def canonical_messages(trace, goal, active_ids):
    """The same state serialization is used for scoring and live continuation."""
    active_ids = set(active_ids)
    observations = []
    boundary = goal["declared_after"]
    initial = trace["actions"][:boundary]
    immutable_ids = {action["id"] for action in initial}
    current = sorted(trace["actions"][boundary:], key=lambda record: record["id"])
    for action in [*initial, *current]:
        if action["id"] in immutable_ids or action["id"] in active_ids:
            observations.append({"action_id": action["id"], "request": action["serialized"],
                                 "response": action["response"], "status": action.get("status", "unknown"),
                                 "documents": [trace["documents"][key] for key in sorted(action.get("document_ids", []))
                                               if key in trace["documents"]]})
    payload = {"task": trace["question"], "initial_context": trace.get("initial_context", ""),
               "initial_resources": sorted(trace.get("initial_resources", [])),
               "context_format_version": CONTEXT_FORMAT_VERSION,
               "requirement": {"id": goal["id"], "text": goal["text"]},
               "snapshot_id": trace["snapshot_id"], "tools": trace["tools"],
               "observations": observations}
    return [{"role": "system", "content": "Conduct the research task using the available tools. Emit a complete serialized tool action."},
            {"role": "user", "content": canonical(payload)}]


def context_id(messages):
    return digest(messages)
