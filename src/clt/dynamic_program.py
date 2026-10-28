import math
from dataclasses import dataclass

import numpy as np

from .lattice import topological_order


@dataclass
class Marginals:
    log_mass: float
    edge: np.ndarray
    terminal: np.ndarray
    inside: np.ndarray
    forward: np.ndarray

    def report(self):
        return {"log_completion_mass": self.log_mass if math.isfinite(self.log_mass) else None,
                "completion_mass": math.exp(self.log_mass),
                "edge_posterior": self.edge.tolist(), "terminal_posterior": self.terminal.tolist()}


def inside_outside(graph, edge_log_probability, terminal_log_probability=None, cost_weight=0.02):
    if not math.isfinite(cost_weight) or cost_weight < 0:
        raise ValueError("cost_weight must be finite and nonnegative")
    order, outgoing = topological_order(graph)
    edge_log_probability = np.asarray(edge_log_probability, dtype=np.float64)
    if edge_log_probability.shape != (len(graph["edges"]),):
        raise ValueError("One complete-action log probability is required per edge")
    if np.isnan(edge_log_probability).any() or (edge_log_probability > 1e-7).any():
        raise ValueError("Invalid complete-action log probability")
    terminal = np.zeros(len(order), dtype=np.float64)
    if graph["verification_action"] is not None:
        if terminal_log_probability is None:
            raise ValueError("Explicit verification actions require policy scores")
        terminal = np.asarray(terminal_log_probability, dtype=np.float64)
        if terminal.shape != (len(order),) or np.isnan(terminal).any() or (terminal > 1e-7).any():
            raise ValueError("Invalid terminal log probabilities")
    weights = np.empty(len(graph["edges"]), dtype=np.float64)
    for source in order:
        mass = sum(math.exp(edge_log_probability[edge["id"]]) * edge["multiplicity"] for edge in outgoing[source])
        if mass > 1 + 1e-6:
            raise ValueError("Outgoing policy events exceed probability one; do not renormalize candidates")
        if graph["nodes"][source]["terminal"] and outgoing[source]:
            raise ValueError("First-completion terminals cannot have outgoing edges")
        for edge in outgoing[source]:
            if not math.isfinite(edge["cost"]) or edge["cost"] < 0 or edge["multiplicity"] < 1:
                raise ValueError("Invalid edge cost or multiplicity")
            weights[edge["id"]] = (edge_log_probability[edge["id"]] - cost_weight * edge["cost"]
                                    + math.log(edge["multiplicity"]))
    inside, forward = np.full(len(order), -np.inf), np.full(len(order), -np.inf)
    for source in reversed(order):
        if graph["nodes"][source]["terminal"]:
            inside[source] = terminal[source]
        for edge in outgoing[source]:
            inside[source] = np.logaddexp(inside[source], weights[edge["id"]] + inside[edge["target"]])
    forward[graph["root"]] = 0.0
    for source in order:
        for edge in outgoing[source]:
            target = edge["target"]
            forward[target] = np.logaddexp(forward[target], forward[source] + weights[edge["id"]])
    root = float(inside[graph["root"]])
    edge_posterior, terminal_posterior = np.zeros(len(weights)), np.zeros(len(order))
    if math.isfinite(root):
        for edge in graph["edges"]:
            edge_posterior[edge["id"]] = math.exp(forward[edge["source"]] + weights[edge["id"]]
                                                  + inside[edge["target"]] - root)
        for node in graph["nodes"]:
            if node["terminal"]:
                terminal_posterior[node["id"]] = math.exp(forward[node["id"]] + terminal[node["id"]] - root)
    return Marginals(root, edge_posterior, terminal_posterior, inside, forward)
