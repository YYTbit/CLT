import copy
from datetime import datetime, timedelta, timezone

import pytest

from clt.certification import assert_fresh, certify
from clt.contexts import canonical_messages
from clt.io import canonical, digest
from clt.lattice import compile_lattice
from clt.provenance import has_independent_support, verified_origin
from clt.verification import LiteralFixtureVerifier, proof_satisfied


def test_discovered_links_keep_their_prerequisite(certificate):
    assert "R" in certificate["dependencies"]["B"]
    assert ["A", "C"] in certificate["exchanges"]


def test_republication_is_one_origin(certificate):
    assert not proof_satisfied(certificate["proof"], {"A", "B"})
    assert proof_satisfied(certificate["proof"], {"A", "C"})
    assert proof_satisfied(certificate["proof"], {"R", "B", "C"})


def test_suffix_list_and_unresolved_origins():
    def document(url):
        return {"url": url, "origin": {"status": "verified", "cited_root": url}}
    first = verified_origin(document("https://one.co.uk/page"))
    same = verified_origin(document("https://news.one.co.uk/article"))
    other = verified_origin(document("https://two.co.uk/page"))
    assert not has_independent_support([first, same], 2)
    assert has_independent_support([first, other], 2)
    assert not has_independent_support([first, None], 2)


def test_unconfirmed_attribution_is_rejected(trace):
    trace["documents"]["republication"].pop("cited_urls")
    result = certify(trace, "start_year", LiteralFixtureVerifier())
    assert any(item["document_id"] == "republication" for item in result["proof_audit"]["rejected"])


def test_unknown_exchange_retains_observed_order(trace):
    trace["actions"][1]["bindings_complete"] = False
    result = certify(trace, "start_year", LiteralFixtureVerifier())
    assert "A" in result["dependencies"]["C"]


@pytest.mark.parametrize("field,value,reason", [
    ("stable", False, "unstable_response"), ("status", "failed", "failed_request"),
    ("response_digest", "wrong", "response_digest_mismatch"),
    ("snapshot_id", "old", "snapshot_mismatch"),
])
def test_invalid_records_are_explicit(trace, field, value, reason):
    trace["actions"][0][field] = value
    result = certify(trace, "start_year", LiteralFixtureVerifier())
    assert result["status"]["R"] == reason
    assert "B" not in {action["id"] for action in result["actions"]}


def test_expired_snapshot_is_excluded(trace):
    trace["actions"][1]["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    result = certify(trace, "start_year", LiteralFixtureVerifier())
    assert result["status"]["A"] == "expired_snapshot"


def test_source_revision_invalidates_certificate(trace, certificate):
    trace["documents"]["primary"]["text"] += " Updated."
    with pytest.raises(ValueError, match="revision"):
        assert_fresh(certificate, trace)


def test_no_future_observations_in_context(trace):
    messages = canonical_messages(trace, trace["goals"][0], {"R"})
    assert '"document_id":"primary"' not in canonical(messages)
    assert '"start_year"' in messages[-1]["content"]
    assert '"actions"' not in messages[-1]["content"]


def test_prior_actions_are_immutable_context(trace):
    trace["goals"][0]["declared_after"] = 2
    certificate = certify(trace, "start_year", LiteralFixtureVerifier())
    graph = compile_lattice(trace, certificate)
    assert "A" not in {edge["action_id"] for edge in graph["edges"]}
    assert '"action_id":"A"' in graph["nodes"][0]["messages"][-1]["content"]


def test_fixture_verifier_is_not_a_benchmark_verifier(trace):
    trace["data_kind"] = "captured_trace"
    with pytest.raises(ValueError, match="restricted"):
        certify(trace, "start_year", LiteralFixtureVerifier())


def test_request_serialization_is_checked(trace):
    trace["actions"][0]["arguments"]["query"] = "different"
    with pytest.raises(ValueError, match="Serialized"):
        certify(trace, "start_year", LiteralFixtureVerifier())


def test_claim_span_is_bound_to_observed_document(trace):
    trace["goals"][0]["proof"]["candidates"] = [{"document_id": "primary", "start": 0, "end": 999}]
    result = certify(trace, "start_year", LiteralFixtureVerifier())
    assert not result["proof_audit"]["accepted"]
    assert result["proof_audit"]["rejected"][0]["reason"] == "invalid_span"


def test_mutating_actions_form_a_barrier(trace):
    trace["tools"]["search"]["read_only"] = False
    result = certify(trace, "start_year", LiteralFixtureVerifier())
    assert not result["actions"]
    assert not compile_lattice(trace, result)["has_completion"]


def test_duplicate_action_strings_cannot_exchange(trace):
    repeated = copy.deepcopy(trace["actions"][1])
    repeated["id"] = "A2"
    trace["actions"].insert(2, repeated)
    result = certify(trace, "start_year", LiteralFixtureVerifier())
    assert "A" in result["dependencies"]["A2"]


def test_alternative_certificates_do_not_duplicate_paths(trace):
    goal = trace["goals"][0]
    original = copy.deepcopy(goal["proof"])
    baseline = compile_lattice(trace, certify(trace, goal["id"], LiteralFixtureVerifier()))
    goal["proof"] = {"op": "or", "children": [original, copy.deepcopy(original)]}
    graph = compile_lattice(trace, certify(trace, goal["id"], LiteralFixtureVerifier()))
    assert len(graph["edges"]) == len(baseline["edges"])


def test_certificate_payload_is_bound(trace, certificate):
    certificate["dependencies"]["B"] = []
    with pytest.raises(ValueError, match="integrity"):
        assert_fresh(certificate, trace)


def test_snapshot_executor_checks_link_availability(trace):
    from clt.session import SnapshotTools

    executor = SnapshotTools(trace["actions"], trace["documents"], trace["initial_resources"], trace["tools"])
    assert executor(trace["actions"][2]["serialized"]).status == "unavailable"
    assert executor(trace["actions"][0]["serialized"]).status == "ok"
    assert executor(trace["actions"][2]["serialized"]).status == "ok"


def test_explicit_verification_is_a_terminal_event(trace):
    request = {"tool": "verify", "arguments": {"goal": "start_year"}}
    trace["goals"][0]["verification_action"] = canonical(request)
    pending = certify(trace, "start_year", LiteralFixtureVerifier())
    assert not pending["verification_observed"]
    trace["tools"]["verify"] = {"read_only": True, "reusable_response": True}
    trace["actions"].append({
        "id": "V", "serialized": canonical(request), **request, "response": {"verified": True},
        "response_digest": digest({"verified": True}), "latency_seconds": 0.1,
        "snapshot_id": trace["snapshot_id"], "status": "ok", "stable": True, "bindings_complete": True,
    })
    certificate = certify(trace, "start_year", LiteralFixtureVerifier())
    assert certificate["verification_observed"]
    assert "V" not in {action["id"] for action in certificate["actions"]}


def test_unknown_fact_is_bound_from_acquired_evidence(trace):
    trace["goals"][0]["proof"] = {
        "op": "claim", "id": "start", "min_origins": 2,
        "hypothesis_template": "The project started in {year}.",
        "extraction_pattern": r"The project started in (?P<year>\d{4})\.",
    }
    certificate = certify(trace, "start_year", LiteralFixtureVerifier())
    assert proof_satisfied(certificate["proof"], {"A", "C"})
    assert {support["hypothesis"] for support in certificate["proof"]["support"]} == {"The project started in 2020."}


def test_independent_sources_must_agree_on_the_factual_value(trace):
    trace["goals"][0]["proof"] = {
        "op": "claim", "id": "start", "min_origins": 2,
        "hypothesis_template": "The project started in {year}.",
        "extraction_pattern": r"The project started in (?P<year>\d{4})\.",
    }
    document = trace["documents"]["independent"]
    document["text"] = "The project started in 2021."
    document["content_digest"] = digest(document["text"])
    action = trace["actions"][3]
    action["response"]["text"] = document["text"]
    action["response_digest"] = digest(action["response"])
    certificate = certify(trace, "start_year", LiteralFixtureVerifier())
    assert not proof_satisfied(certificate["proof"], {"A", "C"})
