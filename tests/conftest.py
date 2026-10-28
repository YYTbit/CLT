import importlib.util
from pathlib import Path

import pytest

from clt.certification import certify
from clt.lattice import compile_lattice
from clt.verification import LiteralFixtureVerifier


@pytest.fixture
def trace():
    path = Path(__file__).parents[1] / "examples" / "capture_trace.py"
    specification = importlib.util.spec_from_file_location("capture_trace", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.make_trace()


@pytest.fixture
def certificate(trace):
    return certify(trace, "start_year", LiteralFixtureVerifier())


@pytest.fixture
def graph(trace, certificate):
    return compile_lattice(trace, certificate)


def diamond(verification_action=None):
    messages = [{"role": "user", "content": "The current evidence state."}]
    nodes = [{"id": index, "mask": index, "terminal": index == 3, "messages": messages}
             for index in range(4)]
    edges = [{"id": index, "source": source, "target": target, "serialized": action,
              "action_id": action, "cost": 1.0, "multiplicity": 1}
             for index, (source, target, action) in enumerate([(0, 1, "a"), (0, 2, "b"), (1, 3, "c"), (2, 3, "c")])]
    return {"nodes": nodes, "edges": edges, "root": 0, "verification_action": verification_action,
            "original_contexts": 4, "has_completion": True, "already_satisfied": False,
            "data_kind": "software_fixture", "task_id": "diamond"}
