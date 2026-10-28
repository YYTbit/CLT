import re

from .io import digest
from .provenance import has_independent_support, verified_origin


class LiteralFixtureVerifier:
    """An exact-span software fixture, distinct from the NLI verification path."""

    identity = {"kind": "software_fixture", "method": "literal_span"}

    def entailment(self, premise, hypothesis):
        return float(premise.strip() == hypothesis.strip())


class NLIVerifier:
    def __init__(self, model_id, revision=None, device="cpu", max_length=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision).to(device).eval()
        self.device, self.max_length = device, max_length
        labels = self.model.config.id2label
        indices = [int(index) for index, label in labels.items() if label.lower() == "entailment"]
        if len(indices) != 1:
            raise ValueError("The NLI configuration must explicitly identify its entailment label")
        self.entailment_index = indices[0]
        self.identity = {"kind": "nli", "model": model_id,
                         "revision": self.model.config._commit_hash or revision}

    def entailment(self, premise, hypothesis):
        encoded = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > self.max_length:
            raise ValueError("A candidate span exceeds the verifier context; provide a shorter exact span")
        with self.torch.no_grad():
            logits = self.model(**encoded.to(self.device)).logits[0].float()
        return logits.softmax(-1)[self.entailment_index].item()


def construct_proof(trace, goal, eligible_ids, verifier, threshold=0.65):
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid entailment threshold")
    if isinstance(verifier, LiteralFixtureVerifier) and trace["data_kind"] != "software_fixture":
        raise ValueError("Literal verification is restricted to software fixtures")
    acquired = {}
    for action in trace["actions"]:
        if action["id"] in eligible_ids:
            for document_id in action.get("document_ids", []):
                acquired.setdefault(document_id, []).append(action["id"])
    accepted, rejected = [], []
    known_claims = {}

    def visit(node):
        operation = node.get("op")
        if operation in {"and", "or"}:
            children = node.get("children", [])
            if not children:
                raise ValueError("AND/OR proof nodes need children")
            return {"op": operation, "children": [visit(child) for child in children]}
        template = node.get("hypothesis_template")
        if operation != "claim" or not (node.get("hypothesis") or template):
            raise ValueError("Proof leaves require a declared hypothesis or a factual template")
        if template and not node.get("extraction_pattern"):
            raise ValueError("Factual templates require a declared named-group extraction pattern")
        pattern = re.compile(node["extraction_pattern"]) if template else None
        claim_id = node["id"]
        signature = (node.get("hypothesis", template), node.get("min_origins", 1), node.get("extraction_pattern"))
        if claim_id in known_claims and known_claims[claim_id] != signature:
            raise ValueError("A claim identifier has inconsistent requirements")
        known_claims[claim_id] = signature
        support = []
        if not isinstance(signature[1], int) or signature[1] < 1:
            raise ValueError("min_origins must be a positive integer")
        candidates = node.get("candidates")
        if candidates is None:
            candidates = []
            for document_id in sorted(acquired):
                document = trace["documents"].get(document_id)
                if document is None:
                    continue
                # Sentence spans are selected from acquired text. Their offsets
                # continue to refer to the original, digest-bound document.
                for match in re.finditer(r"[^\n.!?]+(?:[.!?]|$)", document["text"]):
                    start, end = match.span()
                    while start < end and document["text"][start].isspace():
                        start += 1
                    if 0 < end - start <= 1200:
                        candidates.append({"document_id": document_id, "start": start, "end": end})
        for candidate in candidates:
            document_id = candidate["document_id"]
            document = trace["documents"].get(document_id)
            reason = None
            if document is None or document_id not in acquired:
                reason = "unacquired_document"
            elif digest(document["text"]) != document.get("content_digest"):
                reason = "document_digest_mismatch"
            elif document.get("snapshot_id") != trace["snapshot_id"]:
                reason = "document_snapshot_mismatch"
            origin = verified_origin(document) if document is not None else None
            if reason is None and origin is None:
                reason = "unresolved_origin"
            start, end = candidate["start"], candidate["end"]
            if reason is None and not (isinstance(start, int) and isinstance(end, int)
                                       and 0 <= start < end <= len(document["text"])):
                reason = "invalid_span"
            if reason is None:
                passage = document["text"][start:end]
                hypothesis = node.get("hypothesis")
                if pattern is not None:
                    match = pattern.search(passage)
                    if match is None:
                        reason = "requirement_pattern_mismatch"
                    else:
                        hypothesis = template.format(**match.groupdict())
                probability = verifier.entailment(passage, hypothesis) if reason is None else 0.0
                if not 0 <= probability <= 1:
                    raise ValueError("Invalid verifier probability")
                if reason is None and probability < threshold:
                    reason = "not_entailed"
            record = {"claim_id": claim_id, "document_id": document_id, "start": start, "end": end}
            if reason:
                rejected.append({**record, "reason": reason})
                continue
            entry = {**record, "actions": sorted(acquired[document_id]), "origin": list(origin),
                     "probability": probability, "hypothesis": hypothesis}
            accepted.append(entry)
            support.append(entry)
        return {"op": "claim", "id": claim_id, "hypothesis": node.get("hypothesis", template),
                "min_origins": signature[1], "support": support}

    proof = visit(goal["proof"])
    return proof, {"accepted": accepted, "rejected": rejected,
                   "verifier": verifier.identity, "threshold": threshold}


def proof_satisfied(node, active_ids):
    if node["op"] == "and":
        return all(proof_satisfied(child, active_ids) for child in node["children"])
    if node["op"] == "or":
        return any(proof_satisfied(child, active_ids) for child in node["children"])
    # Independent passages must agree on the same instantiated factual claim.
    groups = {}
    for support in node["support"]:
        if set(support["actions"]) & active_ids:
            groups.setdefault(support["hypothesis"], []).append(tuple(support["origin"]))
    return any(has_independent_support(origins, node["min_origins"]) for origins in groups.values())
