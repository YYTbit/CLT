import json
import time
from dataclasses import dataclass

from .certification import certify
from .contexts import canonical_messages
from .io import digest
from .lattice import compile_lattice, compress_lattice


@dataclass
class ToolResult:
    response: object
    documents: tuple = ()
    provides: tuple = ()
    requires: tuple = ()
    stable: bool = False
    expires_at: str | None = None
    status: str = "ok"
    bindings: tuple = ()
    bindings_complete: bool = False
    latency_seconds: float | None = None


class SnapshotTools:
    """Execute immutable, captured tool responses keyed by exact action text."""

    def __init__(self, actions, documents, initial_resources=(), tools=None):
        self.responses = {}
        for action in actions:
            key = action["serialized"]
            if key in self.responses and digest(self.responses[key]["response"]) != digest(action["response"]):
                raise ValueError("An immutable snapshot cannot return conflicting responses")
            self.responses[key] = action
        self.documents = documents
        self.available = set(initial_resources)
        self.tools = tools or {}
        self.calls = 0
        self.executed = {}

    def __call__(self, serialized):
        current_id = f"action_{self.calls:04d}"
        self.calls += 1
        if serialized not in self.responses:
            return ToolResult({"error": "request_unavailable_in_snapshot"}, status="unavailable")
        action = self.responses[serialized]
        required = set(action.get("requires", []))
        for path in self.tools.get(action["tool"], {}).get("resource_arguments", []):
            value = action["arguments"]
            for part in path.split("."):
                value = value[part]
            required.add(value)
        if not required <= self.available:
            return ToolResult({"error": "request_resource_unavailable"}, status="unavailable")
        bindings = []
        for binding in action.get("bindings", []):
            if binding["source_action"] not in self.executed:
                return ToolResult({"error": "request_binding_unavailable"}, status="unavailable")
            bindings.append({**binding, "source_action": self.executed[binding["source_action"]]})
        if action["status"] == "ok":
            self.available.update(action.get("provides", []))
            self.executed[action["id"]] = current_id
        return ToolResult(action["response"],
                          tuple(self.documents[key] for key in action.get("document_ids", [])),
                          tuple(action.get("provides", [])), tuple(action.get("requires", [])),
                          action.get("stable", False), action.get("expires_at"), action["status"],
                          tuple(bindings), action.get("bindings_complete", False), action["latency_seconds"])


class OnlineAgent:
    def __init__(self, trainer, verifier, threshold=0.65, compress=True):
        self.trainer, self.verifier = trainer, verifier
        self.threshold, self.compress = threshold, compress

    def checkpoint(self, trace, goal_id, checkpoint_dir=None):
        certificate = certify(trace, goal_id, self.verifier, self.threshold)
        graph = compile_lattice(trace, certificate, self.trainer.config["max_states"])
        if self.compress:
            graph = compress_lattice(graph)
        report = self.trainer.adapt(graph, trace=trace, certificate=certificate, checkpoint_dir=checkpoint_dir)
        return report, certificate, graph

    def next_action(self, trace, goal_id, max_new_tokens=512):
        goal = next(goal for goal in trace["goals"] if goal["id"] == goal_id)
        messages = canonical_messages(trace, goal, {action["id"] for action in trace["actions"]})
        return self.trainer.policy.generate(messages, max_new_tokens)

    def run(self, recorder, tool_executor, max_new_tokens=512, checkpoint_dir=None):
        """Adapt and continue through requirements declared before execution."""
        self.trainer.start_task(recorder.task_id)
        reports = []
        for goal in recorder.goals:
            while len(recorder.actions) < self.trainer.config["max_interactions"]:
                serialized = self.next_action(recorder.to_dict(), goal["id"], max_new_tokens)
                request = json.loads(serialized)
                if set(request) != {"tool", "arguments"} or request["tool"] not in recorder.tools:
                    raise ValueError("Expected a complete JSON action with tool and arguments")
                started = time.perf_counter()
                result = tool_executor(serialized)
                latency = time.perf_counter() - started
                if result.latency_seconds is not None:
                    latency = result.latency_seconds
                recorder.record(f"action_{len(recorder.actions):04d}", serialized, request["tool"],
                                request["arguments"], result.response, latency,
                                documents=result.documents, provides=result.provides, requires=result.requires,
                                stable=result.stable, expires_at=result.expires_at, status=result.status,
                                bindings=result.bindings, bindings_complete=result.bindings_complete)
                report, _, graph = self.checkpoint(recorder.to_dict(), goal["id"], checkpoint_dir)
                reports.append(report)
                if graph["checkpoint_verified"] and (graph["has_completion"] or graph["already_satisfied"]):
                    break
            else:
                break
        return recorder.to_dict(), reports
