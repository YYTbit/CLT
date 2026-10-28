# Trace format and implementation notes

## Captured observations

`TraceRecorder.to_dict()` produces a JSON object with `task_id`, `question`, `initial_context`, `snapshot_id`, `data_kind`, `tools`, `initial_resources`, `actions`, `documents`, and `goals`. Set `data_kind` to `captured_trace` for research observations. The executable example uses `software_fixture`.

Each tool entry declares `read_only`, `reusable_response`, and `resource_arguments`. The latter contains dotted argument paths such as `url`. Include the environment's argument schema and tool description in the same entry. These declarations describe the actual environment's capabilities.

An action record contains the following fields.

| Field | Meaning |
| --- | --- |
| `id` | Unique action identifier in this task |
| `serialized` | Complete emitted JSON text with `tool` and `arguments` |
| `tool`, `arguments` | Parsed request, checked against the exact text |
| `response`, `response_digest` | Actual tool response and its SHA-256 digest |
| `requires`, `provides` | Required resources and resources discovered in the response |
| `bindings` | Request arguments obtained from earlier responses |
| `bindings_complete` | Environment confirmation that all such bindings were captured |
| `document_ids` | Documents present in this response |
| `snapshot_id`, `stable` | Snapshot identity and observation reusability |
| `status`, `expires_at` | Request status and optional UTC expiration |
| `latency_seconds`, `timestamp` | Recorded tool cost and acquisition time |

`TraceRecorder.record` computes response and document digests. A binding has `source_action`, `argument_path`, and `response_path`; the two paths are lists of dictionary keys or list indices. The compiler checks the values against the captured request and earlier response. URL-valued arguments additionally require availability through `initial_resources` or an earlier recorded discovery. Keep `bindings_complete=false` when the environment cannot establish that all dependencies were captured.

The compiler certifies exchanges of reusable read-only requests with resolved dependencies and stable snapshots. Unresolved pairs retain their observed precedence. Identical complete action strings retain their order, preventing duplicate policy events. A state-changing operation forms a replay barrier for the current declared requirement. Declare a subsequent requirement after the barrier to train on its later read-only window.

Source revisions, altered observations, and expired snapshots invalidate affected certificates. Re-capture or re-certify before further adaptation. Canonical contexts contain the immutable prefix preceding requirement declaration and the observations acquired at the current ideal. Future responses and proof candidates are excluded from the policy input.

## Documents and origins

Each document provides `id`, `url`, `text`, `content_digest`, `snapshot_id`, and an `origin` object. `origin.status` is `verified` when the environment has confirmed the attribution. `origin.cited_root` identifies the originating document. `origin.relation` is `primary`, `quotation`, or `republication`.

For a primary document, the cited root matches the final URL in its recorded `redirects` chain, or `url` when there is no redirect. Quotations and republications carry `cited_urls`; their cited root occurs in this list and in the recorded response. Acquisition adapters retain the observed attribution links and redirect chain. Missing or unresolved attribution cannot establish independent support.

URL normalization removes fragments and standard ports. Registrable domains use the bundled Public Suffix List through `tldextract`, including private suffixes. Two verified origins are independent when both their registrable domains and cited roots differ. Redirected copies and attributed republications share the cited origin.

## Requirements and evidence proofs

Declare a requirement with `TraceRecorder.declare` before its supporting actions. `declared_after` records the immutable prefix length. The proof is a nested AND/OR expression with claim leaves:

```json
{
  "op": "and",
  "children": [
    {
      "op": "claim",
      "id": "declared_fact",
      "hypothesis": "A factual claim declared by the research process.",
      "min_origins": 2
    }
  ]
}
```

The verifier selects sentence spans from acquired documents and tests their entailment of each declared hypothesis. An optional `candidates` list supplies exact `document_id`, `start`, and `end` offsets for environment-selected passages. Document digests and span bounds are checked before NLI scoring. Acceptance requires entailment probability at least 0.65. Input length is checked against the NLI context limit.

For questions with an unknown factual value, declare `hypothesis_template` and `extraction_pattern` in place of `hypothesis`. Named regex groups bind values from acquired passages, for example `The project started in (?P<year>\d{4})\.` with template `The project started in {year}.`. The template and extraction rule are declared before acquisition; the answer value comes from observed text. NLI verifies each instantiated claim. Independent supports must agree on the same instantiated claim.

`NLIVerifier` loads a sequence-classification checkpoint with an explicit entailment label. The default checkpoint is [DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli). Its resolved revision is recorded in the certificate. `LiteralFixtureVerifier` is restricted to software fixtures.

A claim becomes true when its currently acquired supports meet the independent-origin count. AND nodes require every child; OR nodes accept any child. Multiple certificates establishing the same action sequence contribute one completion path. Requirements already satisfied by the immutable prefix produce no new checkpoint update.

## Policy scoring and updates

The context builder orders acquired observations by stable action identifiers. Scoring and continuation share this builder and the backbone chat template. Gemma's system instruction is folded into the first user turn. Context overflow raises an explicit error.

Each policy event is the complete captured action text followed by the backbone's native assistant turn-ending token when present in its vocabulary, with `tokenizer.eos_token_id` as the fallback. `model.end_token_id` can select an explicit ending. Teacher forcing sums token log probabilities under the original language model distribution. Candidate actions are never renormalized within the lattice. Greedy continuation uses the same end token and requires a complete action. The Qwen3 configuration sets the native chat-template option `enable_thinking=false` for direct JSON tool actions; this option is shared by training and continuation and is part of checkpoint identity.

For action cost `c`, the continuing weight is `pi(action | state) * exp(-lambda * c)`. Costs are recorded tool latencies in seconds. Automatic terminals have weight one. Set `verification_action` in a declared goal to its full serialized request for an explicit verification event; its complete-action likelihood supplies the terminal weight. Record that emitted event before adaptation. The event is excluded from continuing edges, and an unobserved event leaves the policy unchanged.

The inside and forward passes operate in the log domain. Edge and terminal posteriors are detached for each gradient step and recomputed after the update. A finite log mass remains usable even when exponentiation underflows. Unavailable mass skips the step.

The objective is `-log Z + beta * KL(reference || current)`. The regularizer uses full-vocabulary token KL. Predetermined weights are uniform over scored original states, uniform over their outgoing complete actions, and uniform over each action's predicted token positions. Explicit terminal events participate as scored states. These weights remain independent of path posteriors. Symbolic compression retains original-state multiplicity for the regularizer.

Fresh LoRA adapters have zero initial contribution. Disabling them therefore evaluates the fixed entering backbone without duplicating its weights. The configured target modules cover attention and feed-forward projections; Gemma adapters are confined to its language model. AdamW uses weight decay 0.01. These module selections, token ending, KL averaging weights, and graph limit are explicit implementation settings; the shared algorithm settings are listed in the README.

Each successful requirement/certificate pair receives two optimizer steps. Intermediate adapter and optimizer states are committed after each step when a checkpoint directory is configured. Resume completes the remaining steps and retains the processed-pair registry. Call `start_task` for an independent task; it restores the declared initial adapter and a fresh optimizer. A changed model, tokenizer, or configuration fails checkpoint restoration.

## Native environment connection

`OnlineAgent.next_action` generates a complete request from the current canonical context. `OnlineAgent.checkpoint` certifies acquired evidence, compiles the graph, and adapts the policy. An environment can call these methods at its own interaction and verification boundaries.

`OnlineAgent.run` accepts a `TraceRecorder` with declared goals and a callable tool executor. The executor receives the complete action string and returns `ToolResult`, including the actual response, documents, resource discovery, stability, and captured bindings. This connects acquisition directly to certification and continued action generation.

`SnapshotTools` supplies a concrete executor for a captured immutable environment. It enforces captured resource availability and reuses recorded tool latency. `examples/online_adaptation.py` loads its snapshot, selects actions with the backbone, and invokes checkpoint updates. Prepare public tasks, tool snapshots, and source attribution with the native benchmark environment before this run. Native evaluators consume completed research outputs independently of checkpoint verification.
