import copy
import math
import os
from pathlib import Path

import numpy as np

from .certification import assert_fresh
from .dynamic_program import inside_outside
from .io import digest
from .policy import scoring_rows


class LatticeTrainer:
    def __init__(self, policy, config):
        import torch
        from peft import get_peft_model_state_dict

        self.torch, self.policy, self.config = torch, policy, copy.deepcopy(config)
        self.initial_adapter = {key: value.detach().cpu().clone()
                                for key, value in get_peft_model_state_dict(policy.model).items()}
        # The reference is the entering backbone. Fresh LoRA adapters have zero
        # contribution, so disabling them preserves that reference without a copy.
        if any(value.abs().max().item() != 0 for key, value in self.initial_adapter.items() if "lora_B" in key):
            raise ValueError("Initialize CLT from fresh zero-contribution LoRA adapters")
        self.task_id = None
        self.completed = set()
        self.checkpoint_count, self.version = 0, 0
        self.history = []
        self.active_update = None
        self._new_optimizer()

    def _new_optimizer(self):
        self.optimizer = self.torch.optim.AdamW(
            [parameter for parameter in self.policy.model.parameters() if parameter.requires_grad],
            lr=self.config["learning_rate"], weight_decay=self.config.get("weight_decay", 0.01))

    def start_task(self, task_id):
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.policy.model, self.initial_adapter)
        self._new_optimizer()
        self.task_id, self.completed = task_id, set()
        self.checkpoint_count, self.version, self.history = 0, 0, []
        self.active_update = None

    def _scores(self, graph, rows):
        edge = np.full(len(graph["edges"]), -np.inf)
        terminal = np.full(len(graph["nodes"]), -np.inf)
        with self.torch.no_grad():
            for start in range(0, len(rows), self.config["batch_size"]):
                batch = rows[start:start + self.config["batch_size"]]
                scores, _ = self.policy.score(batch)
                for row, value in zip(batch, scores.cpu().double().tolist()):
                    (edge if row["kind"] == "edge" else terminal)[row["index"]] = value
        return edge, terminal

    def adapt(self, graph, *, trace, certificate, checkpoint_dir=None):
        assert_fresh(certificate, trace)
        if graph["certificate_id"] != certificate["certificate_id"]:
            raise ValueError("Graph and certificate do not match")
        if self.task_id != graph["task_id"]:
            raise ValueError("Call start_task before adapting an independent task")
        if not graph.get("checkpoint_verified", True):
            return {"status": "verification_event_unobserved", "version": self.version}
        if len(trace["actions"]) > self.config["max_interactions"]:
            raise ValueError("Task interaction budget exceeded")
        key = (certificate["goal"]["id"], certificate["certificate_id"])
        if any(goal_id == key[0] for goal_id, _ in self.completed):
            return {"status": "already_updated", "version": self.version}
        if self.active_update is not None and tuple(self.active_update["key"]) != key:
            raise ValueError("Resume the active checkpoint update before adapting another requirement")
        if not graph["has_completion"]:
            return {"status": "no_new_certified_completion", "version": self.version}
        if self.checkpoint_count >= self.config["max_checkpoints"]:
            return {"status": "checkpoint_budget_exhausted", "version": self.version}
        rows = scoring_rows(graph)
        if self.active_update is None:
            self.active_update = {"key": list(key), "steps": []}
        reports = self.active_update["steps"]
        failure = None
        for step in range(len(reports), self.config["update_steps"]):
            assert_fresh(certificate, trace)
            edge, terminal = self._scores(graph, rows)
            if np.isnan(edge).any() or np.isnan(terminal).any() or np.isposinf(edge).any() or np.isposinf(terminal).any():
                failure = "completion_mass_unavailable"
                break
            posterior = inside_outside(graph, edge, terminal, self.config["cost_weight"])
            if not math.isfinite(posterior.log_mass):
                failure = "completion_mass_unavailable"
                break
            self.optimizer.zero_grad(set_to_none=True)
            divergence = 0.0
            for start in range(0, len(rows), self.config["batch_size"]):
                batch = rows[start:start + self.config["batch_size"]]
                scores, kl = self.policy.score(batch, with_kl=True)
                weights = scores.new_tensor([
                    (posterior.edge if row["kind"] == "edge" else posterior.terminal)[row["index"]]
                    for row in batch])
                regularizer = (scores.new_tensor([row["kl_weight"] for row in batch]) * kl).sum()
                loss = -(weights.detach() * scores).sum() + self.config["kl_weight"] * regularizer
                if not self.torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    failure = "nonfinite_gradient_step"
                    break
                loss.backward()
                divergence += regularizer.detach().item()
            if failure is not None:
                break
            norm = self.torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.policy.model.parameters() if parameter.requires_grad],
                self.config["gradient_clip"])
            if not self.torch.isfinite(norm):
                self.optimizer.zero_grad(set_to_none=True)
                failure = "nonfinite_gradient_step"
                break
            self.optimizer.step()
            self.version += 1
            reports.append({"step": step + 1, "version": self.version,
                            "log_completion_mass": posterior.log_mass,
                            "reference_to_current_kl": divergence,
                            "objective": -posterior.log_mass + self.config["kl_weight"] * divergence,
                            "gradient_norm": norm.item()})
            if checkpoint_dir is not None:
                self.save(checkpoint_dir)
        # A handled numerical failure closes this verification event. Exceptions
        # retain the saved active update so an interruption can resume its steps.
        self.completed.add(key)
        self.checkpoint_count += 1
        self.active_update = None
        report = {"status": ("updated" if len(reports) == self.config["update_steps"] else
                             "updated_partial" if reports else failure),
                  "task_id": self.task_id, "data_kind": graph["data_kind"],
                  "certificate_id": certificate["certificate_id"], "steps": reports,
                  "version": self.version, "checkpoint_count": self.checkpoint_count}
        if failure is not None:
            report["skipped_step_reason"] = failure
        self.history.append(report)
        if checkpoint_dir is not None:
            self.save(checkpoint_dir)
        return report

    def save(self, directory):
        from peft import get_peft_model_state_dict

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {"format_version": 1, "identity": self.policy.identity, "config": self.config,
                 "task_id": self.task_id, "completed": sorted(self.completed),
                 "checkpoint_count": self.checkpoint_count, "version": self.version,
                 "adapter": get_peft_model_state_dict(self.policy.model),
                 "optimizer": self.optimizer.state_dict(), "history": self.history,
                 "active_update": self.active_update,
                 "initial_adapter": self.initial_adapter,
                 "torch_rng": self.torch.get_rng_state()}
        temporary = directory / ".pending-state.pt"
        self.torch.save(state, temporary)
        os.replace(temporary, directory / "state.pt")
        self.policy.tokenizer.save_pretrained(directory / "tokenizer")

    def restore(self, directory):
        from peft import set_peft_model_state_dict

        state = self.torch.load(Path(directory) / "state.pt", map_location="cpu", weights_only=True)
        if digest(state["identity"]) != digest(self.policy.identity) or state["config"] != self.config:
            raise ValueError("Checkpoint model, tokenizer, or optimizer configuration changed")
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        self.initial_adapter = state["initial_adapter"]
        set_peft_model_state_dict(self.policy.model, state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.task_id, self.completed = state["task_id"], set(map(tuple, state["completed"]))
        self.checkpoint_count, self.version = state["checkpoint_count"], state["version"]
        self.history = state["history"]
        self.active_update = state["active_update"]
        self.torch.set_rng_state(state["torch_rng"])
