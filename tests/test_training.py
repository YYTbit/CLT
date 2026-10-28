import pytest

from clt.policy import LanguageModelPolicy, scoring_rows, token_reference_kl
from clt.training import LatticeTrainer


@pytest.fixture
def policy():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(17)
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "read": 3, "search": 4,
                  "tool": 5, "arguments": 6, "url": 7, "query": 8, "verify": 9,
                  **{f"token_{index}": index + 10 for index in range(22)}}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] + '\\n' }}{% endfor %}"
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                       max_position_embeddings=4096, attention_dropout=0.0))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                           lora_dropout=0, task_type="CAUSAL_LM"))
    return LanguageModelPolicy(model, tokenizer, context_length=4096)


@pytest.fixture
def config():
    return {"learning_rate": 0.001, "gradient_clip": 1.0, "update_steps": 2,
            "batch_size": 2, "kl_weight": 0.05, "cost_weight": 0.02,
            "max_interactions": 40, "max_checkpoints": 8, "max_states": 100000}


def test_full_action_likelihood_includes_end_token(policy, graph):
    torch = policy.torch
    rows = scoring_rows(graph)[:1]
    prompt, target = policy.encode(rows[0]["messages"], rows[0]["action"])
    assert target[-1] == policy.end_token_id
    with torch.no_grad():
        scores, _ = policy.score(rows)
        logits = policy.model(torch.tensor([prompt + target]), use_cache=False).logits[0]
        predictions = logits[len(prompt) - 1:-1].log_softmax(-1)
        expected = predictions.gather(-1, torch.tensor(target)[:, None]).sum()
    assert scores[0].item() == pytest.approx(expected.item(), rel=1e-6)


def test_batch_padding_preserves_action_scores(policy, graph):
    torch = policy.torch
    rows = scoring_rows(graph)[:3]
    with torch.no_grad():
        together, _ = policy.score(rows)
        separate = torch.cat([policy.score([row])[0] for row in rows])
    assert torch.allclose(together, separate, atol=1e-5)


def test_native_turn_end_and_template_options(policy, graph):
    policy.model.config.model_type = "qwen3"
    policy.tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>"]})
    policy.tokenizer.chat_template = "{% if enable_thinking %}reasoning{% endif %}{% for message in messages %}{{ message['content'] }}{% endfor %}"
    candidate = LanguageModelPolicy(policy.model, policy.tokenizer, context_length=4096,
                                    chat_template_kwargs={"enable_thinking": False})
    row = scoring_rows(graph)[0]
    _, target = candidate.encode(row["messages"], row["action"])
    assert target[-1] == policy.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert candidate.identity["chat_template_kwargs"] == {"enable_thinking": False}


def test_reference_kl_direction():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    current = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    result = token_reference_kl(reference.log(), current.log())
    expected = (reference * (reference.log() - current.log())).sum(-1)
    assert torch.allclose(result.double(), expected, atol=1e-7)
    assert not torch.allclose(result.double(), (current * (current.log() - reference.log())).sum(-1))


def test_two_steps_resume_deduplication_and_task_reset(policy, graph, trace, certificate, config, tmp_path):
    torch = policy.torch
    trainer = LatticeTrainer(policy, config)
    trainer.start_task(trace["task_id"])
    rows = scoring_rows(graph)[:1]
    prepared = policy.prepare(rows)
    with torch.no_grad():
        reference_before = policy._logits(prepared, reference=True).clone()
    initial = {key: parameter.detach().clone() for key, parameter in policy.model.named_parameters() if parameter.requires_grad}
    report = trainer.adapt(graph, trace=trace, certificate=certificate, checkpoint_dir=tmp_path)
    assert report["status"] == "updated"
    assert report["version"] == 2
    assert len(report["steps"]) == 2
    assert trainer.adapt(graph, trace=trace, certificate=certificate)["status"] == "already_updated"
    assert any(not torch.equal(initial[key], parameter) for key, parameter in policy.model.named_parameters() if key in initial)
    with torch.no_grad():
        assert torch.equal(reference_before, policy._logits(prepared, reference=True))
    trainer.restore(tmp_path)
    assert trainer.version == 2 and trainer.checkpoint_count == 1
    assert trainer.adapt(graph, trace=trace, certificate=certificate)["status"] == "already_updated"
    trainer.start_task("independent_task")
    assert trainer.version == 0 and not trainer.optimizer.state
    assert all(torch.equal(initial[key], parameter) for key, parameter in policy.model.named_parameters() if key in initial)


def test_interrupted_checkpoint_resumes_remaining_steps(policy, graph, trace, certificate, config, tmp_path, monkeypatch):
    trainer = LatticeTrainer(policy, config)
    trainer.start_task(trace["task_id"])
    save = trainer.save
    class SimulatedInterruption(Exception):
        pass
    def interrupt_after_save(directory):
        save(directory)
        raise SimulatedInterruption
    monkeypatch.setattr(trainer, "save", interrupt_after_save)
    with pytest.raises(SimulatedInterruption):
        trainer.adapt(graph, trace=trace, certificate=certificate, checkpoint_dir=tmp_path)
    assert trainer.version == 1
    monkeypatch.setattr(trainer, "save", save)
    trainer.restore(tmp_path)
    result = trainer.adapt(graph, trace=trace, certificate=certificate, checkpoint_dir=tmp_path)
    assert result["status"] == "updated"
    assert trainer.version == 2
    assert [step["version"] for step in result["steps"]] == [1, 2]


def test_no_completion_preserves_parameters(policy, graph, trace, certificate, config):
    trainer = LatticeTrainer(policy, config)
    trainer.start_task(trace["task_id"])
    graph = {**graph, "has_completion": False}
    result = trainer.adapt(graph, trace=trace, certificate=certificate)
    assert result["status"] == "no_new_certified_completion"
    assert trainer.version == 0 and not trainer.optimizer.state


def test_unobserved_verification_event_preserves_parameters(policy, graph, trace, certificate, config):
    trainer = LatticeTrainer(policy, config)
    trainer.start_task(trace["task_id"])
    graph = {**graph, "checkpoint_verified": False}
    result = trainer.adapt(graph, trace=trace, certificate=certificate)
    assert result["status"] == "verification_event_unobserved"
    assert trainer.version == 0


def test_changed_resume_configuration_is_rejected(policy, config, tmp_path):
    trainer = LatticeTrainer(policy, config)
    trainer.start_task("task")
    trainer.save(tmp_path)
    trainer.config["cost_weight"] = 1
    with pytest.raises(ValueError, match="configuration"):
        trainer.restore(tmp_path)


def test_restore_rejects_a_changed_chat_template(policy, config, tmp_path):
    trainer = LatticeTrainer(policy, config)
    trainer.start_task("task")
    trainer.save(tmp_path)
    policy.tokenizer.chat_template += "Changed instruction."
    changed = LanguageModelPolicy(policy.model, policy.tokenizer, context_length=4096)
    with pytest.raises(ValueError, match="tokenizer"):
        LatticeTrainer(changed, config).restore(tmp_path)


def test_a_verified_requirement_is_not_retrained_on_a_later_prefix(policy, graph, trace, certificate, config):
    from clt.certification import certify
    from clt.io import canonical, digest
    from clt.lattice import compile_lattice
    from clt.verification import LiteralFixtureVerifier

    trainer = LatticeTrainer(policy, config)
    trainer.start_task(trace["task_id"])
    trainer.adapt(graph, trace=trace, certificate=certificate)
    request = {"tool": "search", "arguments": {"query": "another topic"}}
    response = {"links": []}
    trace["actions"].append({"id": "R2", "serialized": canonical(request), **request,
                              "response": response, "response_digest": digest(response),
                              "snapshot_id": trace["snapshot_id"], "status": "ok", "stable": True,
                              "bindings_complete": True, "latency_seconds": 0.1})
    updated_certificate = certify(trace, "start_year", LiteralFixtureVerifier())
    updated_graph = compile_lattice(trace, updated_certificate)
    result = trainer.adapt(updated_graph, trace=trace, certificate=updated_certificate)
    assert result["status"] == "already_updated"
    assert trainer.version == 2 and trainer.checkpoint_count == 1


def test_nonfinite_gradient_step_leaves_no_pending_update(policy, graph, trace, certificate, config, monkeypatch):
    trainer = LatticeTrainer(policy, config)
    trainer.start_task(trace["task_id"])
    score = policy.score
    def invalid_gradient(rows, with_kl=False):
        scores, kl = score(rows, with_kl)
        return (scores * float("nan") if with_kl else scores), kl
    monkeypatch.setattr(policy, "score", invalid_gradient)
    result = trainer.adapt(graph, trace=trace, certificate=certificate)
    assert result["status"] == "nonfinite_gradient_step"
    assert trainer.active_update is None
    assert trainer.version == 0 and trainer.checkpoint_count == 1


def test_unavailable_second_step_finishes_the_checkpoint(policy, graph, trace, certificate, config, monkeypatch):
    import numpy as np

    trainer = LatticeTrainer(policy, config)
    trainer.start_task(trace["task_id"])
    original = trainer._scores
    calls = []
    def unavailable_after_first_step(current, rows):
        calls.append(1)
        if len(calls) == 1:
            return original(current, rows)
        return np.full(len(current["edges"]), -np.inf), np.full(len(current["nodes"]), -np.inf)
    monkeypatch.setattr(trainer, "_scores", unavailable_after_first_step)
    result = trainer.adapt(graph, trace=trace, certificate=certificate)
    assert result["status"] == "updated_partial"
    assert trainer.active_update is None
    assert trainer.version == 1 and trainer.checkpoint_count == 1
    assert trainer.adapt(graph, trace=trace, certificate=certificate)["status"] == "already_updated"


def test_context_overflow_is_explicit(policy, graph):
    policy.context_length = 1
    with pytest.raises(ValueError, match="context_length"):
        policy.score(scoring_rows(graph)[:1])


def test_online_capture_adapt_and_continue(policy, trace, config, monkeypatch):
    from clt.records import TraceRecorder
    from clt.session import OnlineAgent, SnapshotTools
    from clt.verification import LiteralFixtureVerifier

    actions = iter([trace["actions"][1]["serialized"], trace["actions"][3]["serialized"]])
    monkeypatch.setattr(policy, "generate", lambda messages, max_new_tokens: next(actions))
    recorder = TraceRecorder("online_fixture", trace["question"], trace["snapshot_id"],
                             tools=trace["tools"], initial_resources=trace["initial_resources"],
                             data_kind="software_fixture")
    recorder.declare("start_year", trace["goals"][0]["text"], trace["goals"][0]["proof"])
    trainer = LatticeTrainer(policy, config)
    agent = OnlineAgent(trainer, LiteralFixtureVerifier())
    result, reports = agent.run(recorder, SnapshotTools(trace["actions"], trace["documents"],
                                                        trace["initial_resources"], trace["tools"]))
    assert reports[-1]["status"] == "updated"
    assert trainer.version == 2
    assert [action["latency_seconds"] for action in result["actions"]] == [0.5, 0.6]
    assert len(result["actions"]) == 2


def test_online_verification_continues_when_update_mass_is_unavailable(policy, trace, config, monkeypatch):
    import numpy as np
    from clt.records import TraceRecorder
    from clt.session import OnlineAgent, SnapshotTools
    from clt.verification import LiteralFixtureVerifier

    requests = iter([trace["actions"][1]["serialized"], trace["actions"][3]["serialized"]])
    monkeypatch.setattr(policy, "generate", lambda messages, max_new_tokens: next(requests))
    recorder = TraceRecorder("frozen_fixture", trace["question"], trace["snapshot_id"],
                             tools=trace["tools"], initial_resources=trace["initial_resources"],
                             data_kind="software_fixture")
    recorder.declare("start_year", trace["goals"][0]["text"], trace["goals"][0]["proof"])
    trainer = LatticeTrainer(policy, config)
    monkeypatch.setattr(trainer, "_scores", lambda current, rows: (
        np.full(len(current["edges"]), -np.inf), np.full(len(current["nodes"]), -np.inf)))
    result, reports = OnlineAgent(trainer, LiteralFixtureVerifier()).run(
        recorder, SnapshotTools(trace["actions"], trace["documents"], trace["initial_resources"], trace["tools"]))
    assert reports[-1]["status"] == "completion_mass_unavailable"
    assert len(result["actions"]) == 2 and trainer.version == 0


@pytest.mark.parametrize("backbone", ["qwen3", "gemma3"])
def test_backbone_specific_text_scoring(policy, graph, backbone):
    torch = policy.torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Gemma3Config, Qwen3Config

    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096)
    if backbone == "qwen3":
        config = Qwen3Config(**text)
        targets = ["q_proj", "v_proj"]
    else:
        text.update(layer_types=["full_attention"], query_pre_attn_scalar=8)
        vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, image_size=16, patch_size=4)
        config = Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=4)
        targets = r".*language_model\..*\.(q_proj|v_proj)"
    model = get_peft_model(AutoModelForCausalLM.from_config(config),
                          LoraConfig(r=2, lora_alpha=4, target_modules=targets, task_type="CAUSAL_LM"))
    candidate = LanguageModelPolicy(model, policy.tokenizer, context_length=4096)
    scores, kl = candidate.score(scoring_rows(graph)[:1], with_kl=True)
    (-scores.mean() + 0.05 * kl.mean()).backward()
    assert torch.isfinite(scores).all() and torch.isfinite(kl).all()
    if backbone == "gemma3":
        assert all("language_model" in key for key, parameter in model.named_parameters() if parameter.requires_grad)
