# Learning Beyond a Single Trace: Causal Execution Lattices for Test-Time Training of Web Agents

Long-horizon Web agents conduct extended research with intermittent verification feedback. Existing test-time training (TTT) methods adapt large language models from execution experience, yet a single trajectory can reinforce incidental research decisions. To address this challenge, we propose Causal-Lattice Training (CLT), which recovers a certified execution space from one verified Web trace. Specifically, CLT combines action dependencies with source-aware evidence proofs to represent alternative orders, evidence combinations, and valid completion paths. We maximize the aggregate cost-weighted mass of certified completions and derive an exact dynamic-programming solution for this mass and its gradient. We further characterize conditions for sound path recovery and state compression, providing a theoretical perspective on the algorithm design. We evaluate CLT with three backbones on five Web research benchmarks covering navigation, multi-document reasoning, and report generation. CLT leads all benchmark metrics, with 3.2–5.3-point gains in answer quality and citation accuracy over the strongest baselines. Under matched test-time updates, CLT raises subsequent verification by up to 9.4 points over the frozen policy and reduces reference-policy KL by 64.6–66.0 percent relative to single-trace training. Code is available at <https://anonymous.4open.science/r/CLT-CEC5/>.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Adaptation from a Web research execution.">
</p>

## Installation

Use Python 3.10 or later and a PyTorch installation appropriate for the available hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[models,test]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

Graph compilation and dynamic programming require the core dependencies. Install `.[test]` for the CPU graph checks, or `.[models,test]` for policy training and the random-model checks.

## Quick start

This deterministic **software fixture** exercises link discovery, source attribution, first-completion compilation, and exact marginalization. It runs without pretrained models.

```bash
python examples/capture_trace.py --output outputs/fixture/trace.json

clt compile \
  --trace outputs/fixture/trace.json \
  --goal start_year \
  --verifier literal-fixture \
  --output outputs/fixture/compiled

clt inspect \
  --graph outputs/fixture/compiled/lattice.json \
  --fixture-probability 0.1 \
  --output outputs/fixture/marginals.json
```

The example requires two independent origins. A primary document and its republication share one origin; a third document supplies independent support. The discovered URL retains its search prerequisite. `literal-fixture` uses exact matching for this example; research traces use the NLI verifier.

## Prepare a research trace

Capture tool calls with `clt.records.TraceRecorder`. Retain complete serialized requests, actual responses, tool latency, discovered resources, document spans, content digests, and snapshot metadata. Declare each requirement before the actions that establish it. The [trace reference](docs/trace-format.md) describes the record format and provenance rules.

```python
from clt.records import TraceRecorder

recorder = TraceRecorder(
    task_id="research_task", question=question, snapshot_id=snapshot_id,
    tools=tool_schemas, initial_resources=known_urls,
)
recorder.declare("requirement", requirement_text, declared_proof)
```

`question`, `snapshot_id`, `tool_schemas`, `known_urls`, `requirement_text`, and `declared_proof` come from the task environment. Record observations as they arrive and serialize `recorder.to_dict()` to a configured trace file. Acquired passages supply NLI candidates; benchmark reference answers and final evaluator feedback belong to evaluation.

## Test-time training

Choose a backbone configuration and set `model.revision` and `verifier.revision` to the desired immutable Hub commits. Gated backbones require the corresponding access permission. The NLI configuration uses a DeBERTa-v3-large entailment checkpoint with threshold 0.65.

```bash
clt adapt \
  --trace inputs/research_trace.json \
  --goal requirement \
  --config configs/qwen3-8b.yaml \
  --output outputs/task
```

The command certifies the captured observations, compiles the lattice, recomputes posterior path weights at each gradient step, and saves the adapter, optimizer, and task state. `inputs/research_trace.json` is a user-prepared trace in the documented format. `--uncompressed` selects the original first-completion DAG.

Resume an interrupted checkpoint or continue with a later declared requirement from the same task:

```bash
clt adapt \
  --trace inputs/research_trace.json \
  --goal requirement \
  --config configs/qwen3-8b.yaml \
  --resume outputs/task \
  --output outputs/task

clt continue \
  --trace inputs/research_trace.json \
  --goal requirement \
  --config configs/qwen3-8b.yaml \
  --checkpoint outputs/task
```

Continuation uses the same canonical context builder as policy scoring. `clt.session.OnlineAgent` connects checkpoint adaptation to a tool executor; [online_adaptation.py](examples/online_adaptation.py) provides an executable captured-snapshot runner. `LatticeTrainer.start_task` restores the entering policy and optimizer state for an independent task.

## Configuration

| Setting | Value |
| --- | --- |
| Backbones | Qwen3-8B, Llama-3.1-8B-Instruct, Gemma-3-4B-it |
| LoRA | Rank 16, scale 32 |
| Optimizer | AdamW, learning rate 1e-5, gradient clipping 1.0 |
| Checkpoint update | 2 steps, reference-to-current KL coefficient 0.05 |
| Action-cost coefficient | 0.02, recorded latency in seconds |
| Task budget | 40 interactions, 8 checkpoints |
| Inference | BF16, context length 32768, scoring batch size 4 |
| Evaluation seeds | 17, 29, 43, 61, 83 |

Configurations are in [configs/](configs). Target modules, model revisions, verifier device, and the explicit graph-size limit are configurable. Resolved model and tokenizer identities are checked during checkpoint restoration. The [implementation notes](docs/trace-format.md#policy-scoring-and-updates) specify token scoring and regularizer weights.

## Evaluation

Prepare tasks and score completed answers with the native benchmark protocols. Record dataset, model, and evaluator revisions with each score.

| Benchmark | Official resources | Metric |
| --- | --- | --- |
| WebWalkerQA | [Dataset](https://huggingface.co/datasets/callanwu/WebWalkerQA) | Official question-answer scoring |
| FRAMES | [Dataset](https://huggingface.co/datasets/google/frames-benchmark) | Factual answer scoring |
| FanOutQA | [Code and submission protocol](https://github.com/zhudotexe/fanoutqa) | Open-book loose accuracy, blind test |
| DeepResearchBench | [Code and evaluator](https://github.com/Ayanami0730/deep_research_bench) | RACE and FACT |
| ScholarQA-CS2 | [AstaBench](https://github.com/allenai/asta-bench) | Global report metric |

Aggregate native per-seed scores with their original units:

```bash
clt summarize --scores inputs/native_scores.jsonl --output outputs/summary.json
```

Each input line contains `data_kind: benchmark_result`, `benchmark`, `split`, `metric`, `seed`, `value`, `dataset_revision`, `model_revision`, and `evaluator_revision`. The output reports the mean and sample standard deviation for each revision-matched group.

## Software checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

Checks cover certificate freshness, source independence, first-completion semantics, exact path mass and gradients, symbolic compression, full-action likelihood, KL direction, interrupted updates, and independent-task restoration. Neural checks use a small randomly initialized model on CPU.
