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

## Algorithm Flow

To construct a reliable training signal from one Web research execution, CLT first certifies action dependencies and evidence requirements using recorded interactions.

1. Retain eligible read-only actions and their observations. Certify prerequisites, exchanges, and snapshot validity.
2. Construct the dependency graph and proof graph. Compile the first-completion lattice.
3. If a certified completion is reachable, build canonical states and refine compatible state groups. At each update, score continuing actions and terminal events, then compute completion mass, forward prefix mass, and edge and terminal posterior probabilities.
4. If completion mass is finite and positive, update the policy using the CLT objective.
5. Continue Web research with the resulting policy.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Adaptation from a Web research execution.">
</p>

**Figure 1.** Adaptation from a Web research execution. The illustrated task traces doctoral-advisor relationships through linked biographies. After link discovery, six independent reads needed for verification admit 720 orders over 64 acquired-evidence states. Context reuse, single-trace supervision, and path sampling use different portions of this experience. CLT shares partial evidence states and computes the aggregate training contribution exactly within the constructed execution lattice.

## Results

**Table 1.** Performance on five public benchmarks and three backbones. WWQA, FR, and FO report answer scores, with FO denoting the FanOutQA blind test; DRB reports RACE, citation accuracy (C. Acc.), and effective citations (E. Cit.); SQA reports the global score. Entries report five-seed means; ± denotes sample standard deviation. Bold marks the best score per backbone.

![Main results from Table 1](assets/main_results.png)

**CLT improves answer quality across model families and retrieval requirements.** On WWQA, it leads the strongest external method by 4.9, 3.3, and 3.2 points for Qwen, Llama, and Gemma. The corresponding FRAMES gains are 4.9, 3.5, and 3.5 points. FanOutQA gains are 5.3, 4.1, and 4.7 points under the official blind evaluation. Crediting certified alternatives supports both navigation and evidence gathering.

**The gains extend to report quality and citation accuracy.** CLT improves RACE by 5.1, 3.8, and 3.6 points and ScholarQA-CS2 by 3.8, 3.2, and 3.2 points over the strongest external results. Citation-accuracy gains are 4.6, 3.3, and 3.6 points. Source-aware proofs align completion supervision with supporting passages and their origins.

## Getting Started

### Environment Setup

Use Python 3.10 or later and install PyTorch for your system.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

### Data Preparation

Record tool calls and supporting evidence with `TraceRecorder`, following the
[trace guide](docs/trace-format.md). Choose a model profile from [configs/](configs).

### Training and Evaluation

Run test-time training on a verified research requirement:

```bash
clt adapt --trace inputs/research_trace.json --goal requirement \
  --config configs/qwen3-8b.yaml --output outputs/task
```

Evaluate completed answers with the official tools for
[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA),
[FRAMES](https://huggingface.co/datasets/google/frames-benchmark),
[FanOutQA](https://github.com/zhudotexe/fanoutqa),
[DeepResearchBench](https://github.com/Ayanami0730/deep_research_bench), and
[ScholarQA-CS2](https://github.com/allenai/asta-bench).

Collect benchmark results using the [score format](docs/trace-format.md#evaluation),
then summarize them:

```bash
clt summarize --scores inputs/native_scores.jsonl --output outputs/summary.json
```

## License

[MIT License](LICENSE).
