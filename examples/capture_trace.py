"""Create a deterministic software fixture with discovery and provenance constraints."""

import argparse

from clt.io import canonical, write_json
from clt.records import TraceRecorder


def make_trace():
    recorder = TraceRecorder(
        task_id="source_verification_fixture", question="Verify the project's start year using independent sources.",
        snapshot_id="fixture-v1", data_kind="software_fixture",
        initial_resources=["https://example.com/project", "https://example.net/history"],
        tools={"read": {"read_only": True, "reusable_response": True, "resource_arguments": ["url"]},
               "search": {"read_only": True, "reusable_response": True, "resource_arguments": []}})
    recorder.declare("start_year", "Confirm that the project started in 2020 using two independent origins.",
                     {"op": "claim", "id": "start", "hypothesis": "The project started in 2020.", "min_origins": 2})
    search = {"tool": "search", "arguments": {"query": "project history"}}
    recorder.record("R", canonical(search), search["tool"], search["arguments"],
                    {"links": ["https://example.org/republication"]}, 0.2,
                    provides=["https://example.org/republication"], stable=True, bindings_complete=True)
    for action_id, document_id, url, origin, latency in [
        ("A", "primary", "https://example.com/project", "https://example.com/project", 0.5),
        ("B", "republication", "https://example.org/republication", "https://example.com/project", 0.4),
        ("C", "independent", "https://example.net/history", "https://example.net/history", 0.6),
    ]:
        request = {"tool": "read", "arguments": {"url": url}}
        text = "The project started in 2020."
        document = {"id": document_id, "url": url, "text": text, "snapshot_id": "fixture-v1",
                    "origin": {"status": "verified", "cited_root": origin,
                               "relation": "republication" if action_id == "B" else "primary"}}
        if action_id == "B":
            document["cited_urls"] = [origin]
        recorder.record(action_id, canonical(request), "read", request["arguments"],
                        {"text": text, "document_id": document_id, "attribution": origin},
                        latency, documents=[document], stable=True, bindings_complete=True)
    trace = recorder.to_dict()
    for action in trace["actions"]:
        action["timestamp"] = "2026-01-01T00:00:00+00:00"
    return trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/fixture/trace.json")
    write_json(parser.parse_args().output, make_trace())
