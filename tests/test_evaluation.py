import pytest

from clt.evaluation import summarize


def record(seed, value):
    return {"data_kind": "benchmark_result", "benchmark": "native_example", "split": "test",
            "metric": "native_score", "dataset_revision": "dataset-v1", "model_revision": "backbone-v1",
            "evaluator_revision": "evaluator-v1", "seed": seed, "value": value}


def test_native_units_and_sample_standard_deviation():
    result = summarize([record(17, 70), record(29, 80), record(43, 90)])[0]
    assert result["mean"] == 80
    assert result["sample_std"] == 10


def test_protocol_fixtures_are_excluded():
    value = record(17, 1)
    value["data_kind"] = "software_fixture"
    with pytest.raises(ValueError, match="benchmark_result"):
        summarize([value])


def test_duplicate_seed_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([record(17, 1), record(17, 2)])


def test_missing_evaluator_revision_is_rejected():
    value = record(17, 1)
    value.pop("evaluator_revision")
    with pytest.raises(ValueError, match="evaluator_revision"):
        summarize([value])
