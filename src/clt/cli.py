import argparse
import json
import math
from pathlib import Path

from .certification import certify
from .config import load_config
from .dynamic_program import inside_outside
from .evaluation import summarize
from .io import read_json, write_json
from .lattice import compile_lattice, compress_lattice, path_count
from .verification import LiteralFixtureVerifier, NLIVerifier


def _verifier(args, config):
    if args.verifier == "literal-fixture":
        return LiteralFixtureVerifier()
    settings = config["verifier"]
    return NLIVerifier(settings["id"], settings.get("revision"), settings.get("device", "cpu"))


def main():
    parser = argparse.ArgumentParser(description="Causal-Lattice Training")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("compile", "adapt"):
        command = commands.add_parser(name)
        command.add_argument("--trace", required=True)
        command.add_argument("--goal", required=True)
        command.add_argument("--config", default="configs/qwen3-8b.yaml")
        command.add_argument("--verifier", choices=["nli", "literal-fixture"], default="nli")
        command.add_argument("--output", required=True)
        command.add_argument("--uncompressed", action="store_true")
        if name == "adapt":
            command.add_argument("--resume")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--graph", required=True)
    inspect.add_argument("--fixture-probability", type=float, required=True)
    inspect.add_argument("--cost-weight", type=float, default=0.02)
    inspect.add_argument("--output", required=True)
    continuation = commands.add_parser("continue")
    continuation.add_argument("--trace", required=True)
    continuation.add_argument("--goal", required=True)
    continuation.add_argument("--config", default="configs/qwen3-8b.yaml")
    continuation.add_argument("--checkpoint", required=True)
    continuation.add_argument("--max-new-tokens", type=int, default=512)
    aggregate = commands.add_parser("summarize")
    aggregate.add_argument("--scores", required=True)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        records = [json.loads(line) for line in Path(args.scores).read_text().splitlines() if line.strip()]
        write_json(args.output, summarize(records))
        return
    if args.command == "inspect":
        graph = read_json(args.graph)
        if graph["data_kind"] != "software_fixture":
            parser.error("Fixture probabilities are restricted to software_fixture graphs")
        if not 0 < args.fixture_probability <= 1:
            parser.error("fixture-probability must lie in (0, 1]")
        log_probability = math.log(args.fixture_probability)
        marginals = inside_outside(graph, [log_probability] * len(graph["edges"]),
                                   [log_probability] * len(graph["nodes"]), args.cost_weight)
        write_json(args.output, {"data_kind": "software_fixture", "path_count": path_count(graph),
                                 "states": len(graph["nodes"]), "edges": len(graph["edges"]), **marginals.report()})
        return
    config, trace = load_config(args.config), read_json(args.trace)
    if args.command == "continue":
        from .policy import LanguageModelPolicy
        from .session import OnlineAgent
        from .training import LatticeTrainer

        trainer = LatticeTrainer(LanguageModelPolicy.from_config(config), config)
        trainer.restore(args.checkpoint)
        if trainer.task_id != trace["task_id"]:
            parser.error("Continuation task differs from the checkpoint task")
        print(OnlineAgent(trainer, None).next_action(trace, args.goal, args.max_new_tokens))
        return
    verifier = _verifier(args, config)
    certificate = certify(trace, args.goal, verifier, config["verifier"]["threshold"])
    graph = compile_lattice(trace, certificate, config["max_states"])
    if not args.uncompressed:
        graph = compress_lattice(graph)
    output = Path(args.output)
    write_json(output / "certificate.json", certificate)
    write_json(output / "lattice.json", graph)
    if args.command == "adapt":
        from .policy import LanguageModelPolicy
        from .training import LatticeTrainer

        trainer = LatticeTrainer(LanguageModelPolicy.from_config(config), config)
        if args.resume:
            trainer.restore(args.resume)
        else:
            trainer.start_task(trace["task_id"])
        report = trainer.adapt(graph, trace=trace, certificate=certificate, checkpoint_dir=output)
        trainer.save(output)
        write_json(output / "adaptation.json", report)
    print(json.dumps({"states": len(graph["nodes"]), "edges": len(graph["edges"]),
                      "certified_paths": path_count(graph), "output": str(output)}))


if __name__ == "__main__":
    main()
