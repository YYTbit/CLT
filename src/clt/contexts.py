from .io import canonical, digest


def canonical_messages(trace, goal, active_ids):
    """The same state serialization is used for scoring and live continuation."""
    active_ids = set(active_ids)
    observations = []
    for action in sorted(trace["actions"], key=lambda record: record["id"]):
        if action["id"] in active_ids:
            observations.append({"action_id": action["id"], "request": action["serialized"],
                                 "response": action["response"],
                                 "documents": [trace["documents"][key] for key in sorted(action.get("document_ids", []))]})
    payload = {"task": trace["question"], "initial_context": trace.get("initial_context", ""),
               "requirement": {"id": goal["id"], "text": goal["text"]},
               "snapshot_id": trace["snapshot_id"], "tools": trace["tools"],
               "observations": observations}
    return [{"role": "system", "content": "Conduct the research task using the available tools. Emit a complete serialized tool action."},
            {"role": "user", "content": canonical(payload)}]


def context_id(messages):
    return digest(messages)
