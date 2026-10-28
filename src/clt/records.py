import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .io import digest


@dataclass
class TraceRecorder:
    """Capture exact requests and observations from an existing tool environment."""

    task_id: str
    question: str
    snapshot_id: str
    initial_context: str = ""
    tools: dict = field(default_factory=dict)
    initial_resources: list = field(default_factory=list)
    data_kind: str = "captured_trace"
    actions: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    documents: dict = field(default_factory=dict)

    def declare(self, goal_id, text, proof, verification_action=None):
        if any(goal["id"] == goal_id for goal in self.goals):
            raise ValueError("Requirement identifiers must be unique within a task")
        goal = {"id": goal_id, "text": text, "proof": proof,
                "declared_after": len(self.actions), "verification_action": verification_action}
        self.goals.append(goal)
        return goal

    def record(self, action_id, serialized, tool, arguments, response, latency_seconds,
               *, requires=(), provides=(), documents=(), status="ok", stable=False,
               expires_at=None, bindings=(), bindings_complete=False):
        if any(action["id"] == action_id for action in self.actions):
            raise ValueError("Action identifiers must be unique")
        if not math.isfinite(latency_seconds) or latency_seconds < 0:
            raise ValueError("Tool latency must be finite and nonnegative")
        for document in documents:
            document = dict(document)
            document["content_digest"] = digest(document["text"])
            previous = self.documents.get(document["id"])
            if previous is not None and previous != document:
                raise ValueError("A document identifier cannot change within a snapshot")
            self.documents[document["id"]] = document
        self.actions.append({
            "id": action_id, "serialized": serialized, "tool": tool, "arguments": arguments,
            "response": response, "response_digest": digest(response),
            "requires": list(requires), "provides": list(provides),
            "document_ids": [document["id"] for document in documents],
            "snapshot_id": self.snapshot_id, "stable": stable, "status": status,
            "latency_seconds": latency_seconds, "expires_at": expires_at,
            "bindings": list(bindings), "bindings_complete": bindings_complete,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def to_dict(self):
        return asdict(self)


def validate_trace(trace):
    for key in ("task_id", "question", "snapshot_id", "tools", "actions", "goals", "documents"):
        if key not in trace:
            raise ValueError(f"Trace is missing {key}")
    if trace.get("data_kind") not in {"captured_trace", "software_fixture"}:
        raise ValueError("Declare captured_trace or software_fixture")
    identifiers = [action["id"] for action in trace["actions"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate action identifiers")
    goals = [goal["id"] for goal in trace["goals"]]
    if len(set(goals)) != len(goals):
        raise ValueError("Duplicate requirement identifiers")
    for goal in trace["goals"]:
        if not isinstance(goal["declared_after"], int) or not 0 <= goal["declared_after"] <= len(identifiers):
            raise ValueError("Invalid requirement declaration boundary")
    for action in trace["actions"]:
        if not action["serialized"] or action["tool"] not in trace["tools"]:
            raise ValueError("An action needs its complete serialization and a tool schema")
        request = json.loads(action["serialized"])
        if request != {"tool": action["tool"], "arguments": action["arguments"]}:
            raise ValueError("Serialized action differs from the captured request")
        latency = action["latency_seconds"]
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("Invalid tool latency")
    return trace
