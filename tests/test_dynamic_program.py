import math

import numpy as np
import pytest

from clt.dynamic_program import inside_outside
from clt.lattice import compress_lattice, path_count, topological_order
from clt.policy import scoring_rows
from conftest import diamond


def enumerate_weights(graph, edge_scores, terminal_scores, cost_weight):
    _, outgoing = topological_order(graph)
    paths = []
    def visit(source, weight, selected):
        if graph["nodes"][source]["terminal"]:
            gamma = terminal_scores[source] if graph["verification_action"] else 0
            paths.append((weight * math.exp(gamma), selected, source))
            return
        for edge in outgoing[source]:
            value = math.exp(edge_scores[edge["id"]] - cost_weight * edge["cost"])
            visit(edge["target"], weight * value * edge["multiplicity"], selected + [edge["id"]])
    visit(graph["root"], 1, [])
    return paths


@pytest.mark.parametrize("explicit", [False, True])
def test_inside_outside_matches_exhaustive_paths(graph, explicit):
    graph["verification_action"] = "verify" if explicit else None
    scores = np.log(np.linspace(0.03, 0.08, len(graph["edges"])))
    terminal = np.full(len(graph["nodes"]), math.log(0.3))
    result = inside_outside(graph, scores, terminal, cost_weight=0.02)
    paths = enumerate_weights(graph, scores, terminal, 0.02)
    mass = sum(weight for weight, _, _ in paths)
    assert math.exp(result.log_mass) == pytest.approx(mass)
    for edge in graph["edges"]:
        expected = sum(weight for weight, selected, _ in paths if edge["id"] in selected) / mass
        assert result.edge[edge["id"]] == pytest.approx(expected)
    assert result.terminal.sum() == pytest.approx(1)


def test_no_candidate_probability_renormalization():
    result = inside_outside(diamond(), [math.log(0.2)] * 4, cost_weight=0)
    assert math.exp(result.log_mass) == pytest.approx(0.08)


def test_first_completion_stops_all_outgoing_actions(graph):
    _, outgoing = topological_order(graph)
    assert all(not outgoing[node["id"]] for node in graph["nodes"] if node["terminal"])


def test_underflow_retains_a_finite_log_mass():
    result = inside_outside(diamond(), [-10000.0] * 4, cost_weight=0)
    assert math.isfinite(result.log_mass)
    assert result.report()["completion_mass"] == 0
    assert result.terminal.sum() == pytest.approx(1)


def test_zero_mass_has_zero_posteriors():
    result = inside_outside(diamond(), [-math.inf] * 4)
    assert result.log_mass == -math.inf
    assert not result.edge.any()
    assert result.report()["log_completion_mass"] is None


@pytest.mark.parametrize("value", [0.1, math.nan, math.inf])
def test_invalid_log_probability(value):
    with pytest.raises(ValueError):
        inside_outside(diamond(), [value] * 4)


def test_candidate_events_cannot_exceed_probability_one():
    with pytest.raises(ValueError, match="exceed"):
        inside_outside(diamond(), [math.log(0.6)] * 4)


@pytest.mark.parametrize("explicit", [False, True])
def test_compression_preserves_mass_paths_and_original_kl_weights(explicit):
    original = diamond("verify" if explicit else None)
    compressed = compress_lattice(original)
    assert len(compressed["nodes"]) == 3
    assert path_count(original) == path_count(compressed) == 2
    results = [inside_outside(graph, [math.log(0.2)] * len(graph["edges"]),
                              [math.log(0.4)] * len(graph["nodes"])) for graph in (original, compressed)]
    assert results[0].log_mass == pytest.approx(results[1].log_mass)
    for graph in (original, compressed):
        rows = scoring_rows(graph)
        assert sum(row["kl_weight"] for row in rows) == pytest.approx(1)
    original_c = sum(row["kl_weight"] for row in scoring_rows(original) if row["action"] == "c")
    compressed_c = sum(row["kl_weight"] for row in scoring_rows(compressed) if row["action"] == "c")
    assert original_c == pytest.approx(compressed_c)


def test_accidentally_equal_scores_do_not_authorize_compression():
    graph = diamond()
    graph["nodes"][2]["messages"] = [{"role": "user", "content": "A distinct state"}]
    assert len(compress_lattice(graph)["nodes"]) == 4


def test_repeated_compression_preserves_regularizer_weights():
    once = compress_lattice(diamond())
    twice = compress_lattice(once)
    weight = lambda graph: sum(row["kl_weight"] for row in scoring_rows(graph) if row["action"] == "c")
    assert weight(once) == pytest.approx(weight(twice))


def test_detached_posterior_gradient_matches_differentiated_path_sum(graph):
    torch = pytest.importorskip("torch")
    scores = torch.linspace(-4, -3, len(graph["edges"]), dtype=torch.float64, requires_grad=True)
    result = inside_outside(graph, scores.detach().numpy())
    _, outgoing = topological_order(graph)
    def inside(source):
        if graph["nodes"][source]["terminal"]:
            return scores.new_tensor(0)
        return torch.logsumexp(torch.stack([scores[edge["id"]] - 0.02 * edge["cost"] + inside(edge["target"])
                                           for edge in outgoing[source]]), 0)
    exact, = torch.autograd.grad(inside(graph["root"]), scores)
    surrogate, = torch.autograd.grad((scores * scores.new_tensor(result.edge).detach()).sum(), scores)
    assert torch.allclose(exact, surrogate, atol=1e-12)


def test_six_independent_reads_share_64_states():
    nodes, edges = [], []
    for mask in range(64):
        nodes.append({"id": mask, "terminal": mask == 63, "messages": []})
        for action in range(6):
            if not mask >> action & 1:
                edges.append({"id": len(edges), "source": mask, "target": mask | 1 << action,
                              "serialized": str(action), "cost": 0, "multiplicity": 1})
    graph = {"nodes": nodes, "edges": edges, "root": 0, "verification_action": None}
    assert path_count(graph) == 720
    assert len(edges) == 192
    result = inside_outside(graph, [math.log(0.1)] * len(edges))
    assert math.exp(result.log_mass) == pytest.approx(720 * 0.1**6)
    assert result.edge.sum() == pytest.approx(6)


def test_explicit_terminal_score_gradient():
    torch = pytest.importorskip("torch")
    graph = diamond("verify")
    parameters = torch.tensor([-3.0, -4.0, -2.0, -5.0], dtype=torch.float64, requires_grad=True)
    edge = parameters[:3][torch.tensor([0, 1, 2, 2])]
    gamma = np.full(4, parameters[3].detach().item())
    result = inside_outside(graph, edge.detach().numpy(), gamma, cost_weight=0)
    exact = torch.logsumexp(torch.stack([parameters[0] + parameters[2] + parameters[3],
                                         parameters[1] + parameters[2] + parameters[3]]), 0)
    exact_gradient, = torch.autograd.grad(exact, parameters)
    surrogate = (edge * edge.new_tensor(result.edge)).sum() + parameters[3] * result.terminal.sum()
    actual, = torch.autograd.grad(surrogate, parameters)
    assert torch.allclose(exact_gradient, actual, atol=1e-12)
