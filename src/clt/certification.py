from datetime import datetime, timezone
from itertools import combinations

from .io import digest
from .records import validate_trace
from .verification import construct_proof


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _lookup(value, path):
    for part in path:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def certify(trace, goal_id, verifier, threshold=0.65, *, now=None, snapshot_id=None):
    validate_trace(trace)
    now = now or datetime.now(timezone.utc)
    if snapshot_id is not None and snapshot_id != trace["snapshot_id"]:
        raise ValueError("The active environment snapshot differs from the captured trace")
    goal = next((goal for goal in trace["goals"] if goal["id"] == goal_id), None)
    if goal is None:
        raise ValueError("Unknown declared requirement")
    resources = {resource: None for resource in trace.get("initial_resources", [])}
    valid, prerequisites, status = {}, {}, {}
    complete_bindings = {}
    for action in trace["actions"]:
        action_id = action["id"]
        reason = None
        tool = trace["tools"][action["tool"]]
        parents = set(action.get("prerequisites", []))
        required = set(action.get("requires", []))
        for argument in tool.get("resource_arguments", []):
            try:
                value = _lookup(action["arguments"], argument.split("."))
                required.add(value)
            except (KeyError, IndexError, TypeError):
                reason = "missing_request_argument"
        # Every URL-valued request argument carries a link-availability constraint.
        required.update(value for value in _strings(action["arguments"])
                        if value.startswith(("http://", "https://")))
        for resource in required:
            if resource not in resources:
                reason = "unavailable_request_resource"
            elif resources[resource] is not None:
                parents.add(resources[resource])
        for binding in action.get("bindings", []):
            source = binding["source_action"]
            try:
                if source not in valid or _lookup(action["arguments"], binding["argument_path"]) != _lookup(
                        valid[source]["response"], binding["response_path"]):
                    reason = "unresolved_argument_binding"
                parents.add(source)
            except (KeyError, IndexError, TypeError):
                reason = "unresolved_argument_binding"
        if not parents <= valid.keys():
            reason = "unavailable_prerequisite"
        if action.get("status") != "ok":
            reason = "failed_request"
        elif action.get("response_digest") != digest(action["response"]):
            reason = "response_digest_mismatch"
        elif action.get("snapshot_id") != trace["snapshot_id"]:
            reason = "snapshot_mismatch"
        elif not action.get("stable", False):
            reason = "unstable_response"
        expiration = action.get("expires_at")
        if expiration:
            expiry = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
            if expiry.tzinfo is None or expiry <= now:
                reason = "expired_snapshot"
        for document_id in action.get("document_ids", []):
            document = trace["documents"].get(document_id)
            if (document is None or document.get("content_digest") != digest(document["text"])
                    or document.get("snapshot_id") != trace["snapshot_id"]
                    or not any(document["text"] in value for value in _strings(action["response"]))):
                reason = "invalid_document_record"
            elif any(not any(url in value for value in _strings(action["response"]))
                     for url in document.get("cited_urls", [])):
                reason = "unobserved_source_attribution"
        if reason:
            status[action_id] = reason
            continue
        valid[action_id] = action
        prerequisites[action_id] = parents
        complete_bindings[action_id] = action.get("bindings_complete", False)
        status[action_id] = "valid"
        strings = set(_strings(action["response"]))
        for resource in action.get("provides", []):
            if not any(resource in value for value in strings):
                raise ValueError("A discovered resource must occur in its recorded response")
            resources.setdefault(resource, action_id)

    boundary = goal["declared_after"]
    initial_ids = [action["id"] for action in trace["actions"][:boundary]]
    window = trace["actions"][boundary:]
    verification_action = goal.get("verification_action")
    verification_index = next((index for index, action in enumerate(window)
                               if action["serialized"] == verification_action
                               and status[action["id"]] in {"valid", "unstable_response"}), None)
    if verification_index is not None:
        window = window[:verification_index]
    # A mutation is a replay barrier. Preserve subsequent observations as context
    # for a later declared requirement, rather than commuting reads across it.
    barrier = next((index for index, action in enumerate(window)
                    if not trace["tools"][action["tool"]].get("read_only", False)), len(window))
    eligible = [action for action in window[:barrier] if action["id"] in valid]
    eligible_ids = {action["id"] for action in eligible}
    initial_set = set(initial_ids) & valid.keys()
    eligible = [action for action in eligible
                if prerequisites[action["id"]] <= eligible_ids | initial_set]
    eligible_ids = {action["id"] for action in eligible}
    dependencies = {action["id"]: set(prerequisites[action["id"]]) & eligible_ids for action in eligible}
    exchanges, retained = [], []

    def ancestors(action_id):
        result, pending = set(), list(dependencies[action_id])
        while pending:
            parent = pending.pop()
            if parent not in result:
                result.add(parent)
                pending.extend(dependencies[parent])
        return result

    for left, right in combinations(eligible, 2):
        a, b = left["id"], right["id"]
        safe = (a not in ancestors(b) and b not in ancestors(a)
                and complete_bindings[a] and complete_bindings[b]
                and left["serialized"] != right["serialized"]
                and trace["tools"][left["tool"]].get("reusable_response", False)
                and trace["tools"][right["tool"]].get("reusable_response", False))
        if safe:
            exchanges.append([a, b])
        else:
            dependencies[b].add(a)
            retained.append([a, b])
    proof, proof_audit = construct_proof(trace, goal, eligible_ids | initial_set, verifier, threshold)
    result = {"task_id": trace["task_id"], "data_kind": trace["data_kind"], "goal": goal,
              "trace_digest": digest(trace), "snapshot_id": trace["snapshot_id"],
              "actions": eligible, "initial_ids": initial_ids, "proof": proof,
              "verification_observed": verification_action is None or verification_index is not None,
              "dependencies": {key: sorted(value) for key, value in dependencies.items()},
              "exchanges": exchanges, "retained_precedence": retained,
              "status": status, "proof_audit": proof_audit}
    result["certificate_id"] = digest(result)
    return result


def assert_fresh(certificate, trace):
    if certificate["trace_digest"] != digest(trace):
        raise ValueError("Trace or source revision changed; re-certification is required")
    expected = digest({key: value for key, value in certificate.items() if key != "certificate_id"})
    if certificate["certificate_id"] != expected:
        raise ValueError("Certificate integrity check failed")
    for action in certificate["actions"]:
        expiration = action.get("expires_at")
        if expiration and datetime.fromisoformat(expiration.replace("Z", "+00:00")) <= datetime.now(timezone.utc):
            raise ValueError("A certified snapshot expired")
