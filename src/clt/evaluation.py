import math
import statistics
from collections import defaultdict


def summarize(records):
    """Aggregate native evaluator scores without changing their units."""
    groups = defaultdict(dict)
    for record in records:
        if record.get("data_kind") != "benchmark_result":
            raise ValueError("Benchmark aggregation requires native benchmark_result records")
        for key in ("dataset_revision", "evaluator_revision", "model_revision", "split"):
            if not record.get(key):
                raise ValueError(f"Evaluation record requires {key}")
        key = (record["benchmark"], record["split"], record["metric"], record["model_revision"],
               record["dataset_revision"], record["evaluator_revision"])
        seed, value = record["seed"], record["value"]
        if seed in groups[key] or not math.isfinite(value):
            raise ValueError("Duplicate seed or nonfinite native score")
        groups[key][seed] = value
    result = []
    for key, values in sorted(groups.items()):
        scores = list(values.values())
        result.append({"benchmark": key[0], "split": key[1], "metric": key[2], "model_revision": key[3],
                       "dataset_revision": key[4], "evaluator_revision": key[5], "seeds": sorted(values),
                       "mean": statistics.mean(scores),
                       "sample_std": statistics.stdev(scores) if len(scores) > 1 else None})
    return result
