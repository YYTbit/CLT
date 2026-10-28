"""Run CLT against an immutable captured environment with model-selected actions."""

import argparse

from clt.config import load_config
from clt.io import read_json, write_json
from clt.policy import LanguageModelPolicy
from clt.records import TraceRecorder
from clt.session import OnlineAgent, SnapshotTools
from clt.training import LatticeTrainer
from clt.verification import NLIVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--config", default="configs/qwen3-8b.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshot, config = read_json(args.snapshot), load_config(args.config)
    if snapshot["data_kind"] != "captured_trace":
        raise ValueError("Provide a captured research environment for the model run")
    recorder = TraceRecorder(snapshot["task_id"], snapshot["question"], snapshot["snapshot_id"],
                             initial_context=snapshot.get("initial_context", ""), tools=snapshot["tools"],
                             initial_resources=snapshot.get("initial_resources", []))
    for goal in snapshot["goals"]:
        if goal["declared_after"] != 0:
            raise ValueError("This runner expects requirements declared before interaction")
        recorder.declare(goal["id"], goal["text"], goal["proof"], goal.get("verification_action"))
    verifier = config["verifier"]
    agent = OnlineAgent(LatticeTrainer(LanguageModelPolicy.from_config(config), config),
                        NLIVerifier(verifier["id"], verifier.get("revision"), verifier["device"]),
                        verifier["threshold"])
    trace, reports = agent.run(recorder, SnapshotTools(snapshot["actions"], snapshot["documents"],
                                                     snapshot.get("initial_resources", []), snapshot["tools"]),
                               checkpoint_dir=args.output)
    write_json(f"{args.output}/trace.json", trace)
    write_json(f"{args.output}/adaptation.json", reports)


if __name__ == "__main__":
    main()
