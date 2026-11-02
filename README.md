# Learning Beyond a Single Trace: Causal Execution Lattices for Test-Time Training of Web Agents

Long-horizon Web agents conduct extended research with intermittent verification
feedback. Existing test-time training (TTT) methods adapt large language models
from execution experience, yet a single trajectory can reinforce incidental
research decisions. To address this challenge, we propose Causal-Lattice
Training (CLT), which recovers a certified execution space from one verified Web
trace. Specifically, CLT combines action dependencies with source-aware evidence
proofs to represent alternative orders, evidence combinations, and valid
completion paths. We maximize the aggregate cost-weighted mass of certified
completions and derive an exact dynamic-programming solution for this mass and
its gradient. We further characterize conditions for sound path recovery and
state compression, providing a theoretical perspective on the algorithm design.
We evaluate CLT with three backbones on five Web research benchmarks covering
navigation, multi-document reasoning, and report generation. CLT leads all
benchmark metrics, with 3.2–5.3-point gains in answer quality and citation
accuracy over the strongest baselines. Under matched test-time updates, CLT
raises subsequent verification by up to 9.4 points over the frozen policy and
reduces reference-policy KL by 64.6–66.0 percent relative to single-trace
training.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Adaptation from a Web research execution.">
</p>

## Installation

Use Python 3.10 or later and install PyTorch for your system.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,test]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## Quick start

Compile the bundled example trace:

```bash
python examples/capture_trace.py --output outputs/example/trace.json

clt compile --trace outputs/example/trace.json --goal start_year \
  --verifier literal-fixture --output outputs/example/compiled
```

## Test-time training

Record tool calls and supporting evidence with `TraceRecorder`, following the
[trace guide](docs/trace-format.md). Choose a model profile from [configs/](configs)
and run adaptation on a verified requirement:

```bash
clt adapt --trace inputs/research_trace.json --goal requirement \
  --config configs/qwen3-8b.yaml --output outputs/task
```

To resume an interrupted run, add `--resume outputs/task` to the same command.
For online adaptation, see [online_adaptation.py](examples/online_adaptation.py).

## Evaluation

Evaluate completed answers with the official tools for
[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA),
[FRAMES](https://huggingface.co/datasets/google/frames-benchmark),
[FanOutQA](https://github.com/zhudotexe/fanoutqa),
[DeepResearchBench](https://github.com/Ayanami0730/deep_research_bench), and
[ScholarQA-CS2](https://github.com/allenai/asta-bench).

Use the [score format](docs/trace-format.md#evaluation) to collect benchmark
results, then summarize them:

```bash
clt summarize --scores inputs/native_scores.jsonl --output outputs/summary.json
```

## Tests

```bash
python -m pytest -q
```

## License

[MIT License](LICENSE).
