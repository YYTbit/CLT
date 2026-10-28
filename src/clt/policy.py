import hashlib
import inspect
from contextlib import nullcontext

from .contexts import CONTEXT_FORMAT_VERSION
from .io import digest


def parameter_digest(model):
    import torch

    checksum = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        checksum.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
        for chunk in parameter.detach().reshape(-1).split(1048576):
            checksum.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def token_reference_kl(reference_logits, current_logits):
    """KL(reference || current), retaining the full vocabulary."""
    import torch

    reference = torch.log_softmax(reference_logits.float(), dim=-1)
    current = torch.log_softmax(current_logits.float(), dim=-1)
    return (reference.exp() * (reference - current)).sum(-1)


def scoring_rows(graph):
    outgoing = {node["id"]: [] for node in graph["nodes"]}
    for edge in graph["edges"]:
        outgoing[edge["source"]].append(edge)
    scored = [node for node in graph["nodes"]
              if outgoing[node["id"]] or (node["terminal"] and graph["verification_action"] is not None)]
    total = sum(len(node.get("members", [node["id"]])) for node in scored)
    rows = []
    for node in scored:
        node_weight = len(node.get("members", [node["id"]])) / total
        edges = outgoing[node["id"]]
        count = sum(edge["multiplicity"] for edge in edges)
        for edge in edges:
            rows.append({"kind": "edge", "index": edge["id"], "messages": node["messages"],
                         "action": edge["serialized"],
                         "kl_weight": node_weight * edge["multiplicity"] / count})
        if node["terminal"] and graph["verification_action"] is not None:
            rows.append({"kind": "terminal", "index": node["id"], "messages": node["messages"],
                         "action": graph["verification_action"], "kl_weight": node_weight})
    return rows


class LanguageModelPolicy:
    def __init__(self, model, tokenizer, context_length=32768, end_token_id=None, identity=None,
                 chat_template_kwargs=None):
        import torch

        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.context_length = context_length
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        native_marker = {"qwen3": "<|im_end|>", "llama": "<|eot_id|>", "gemma3": "<end_of_turn>"}.get(model.config.model_type)
        default_end = (tokenizer.convert_tokens_to_ids(native_marker) if native_marker in tokenizer.get_vocab()
                       else tokenizer.eos_token_id)
        self.end_token_id = default_end if end_token_id is None else end_token_id
        if self.end_token_id is None:
            raise ValueError("A complete action requires an explicit end token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = self.end_token_id
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.supports_logit_limit = "logits_to_keep" in inspect.signature(base.forward).parameters
        self.identity = identity or {"model_type": model.config.model_type,
                                     "architecture": digest(model.config.to_dict())}
        if not self.identity.get("revision"):
            self.identity["frozen_parameters"] = parameter_digest(model)
        self.identity = {**self.identity, "tokenizer": digest(tokenizer.get_vocab()),
                         "chat_template": digest(tokenizer.chat_template),
                         "special_tokens": digest(tokenizer.special_tokens_map),
                         "context_format_version": CONTEXT_FORMAT_VERSION,
                         "end_token_id": self.end_token_id, "context_length": context_length,
                         "chat_template_kwargs": self.chat_template_kwargs}
        if hasattr(tokenizer, "backend_tokenizer"):
            self.identity["tokenizer_backend"] = digest(tokenizer.backend_tokenizer.to_str())
        self.model.eval()

    @classmethod
    def from_config(cls, config):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(config["seed"])
        model_config = config["model"]
        model_id, revision = model_config["id"], model_config.get("revision")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[model_config["dtype"]]
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype, device_map=model_config.get("device_map", "auto"))
        adapter = config["adapter"]
        target_modules = adapter["target_modules"]
        if base.config.model_type == "gemma3":
            target_modules = r".*language_model\..*\.(" + "|".join(target_modules) + ")"
        model = get_peft_model(base, LoraConfig(r=adapter["rank"], lora_alpha=adapter["alpha"],
                                               target_modules=target_modules,
                                               lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        return cls(model, tokenizer, config["context_length"], model_config.get("end_token_id"),
                   {"model": model_id, "revision": base.config._commit_hash or revision,
                    "adapter": adapter, "dtype": model_config["dtype"]}, model_config.get("chat_template_kwargs"))

    def _prompt(self, messages):
        if self.model.config.model_type == "gemma3" and messages[0]["role"] == "system":
            messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]},
                        *messages[2:]]
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   **self.chat_template_kwargs)

    def encode(self, messages, action):
        prompt = self._prompt(messages)
        target = self.tokenizer.encode(action, add_special_tokens=False) + [self.end_token_id]
        if not prompt or not target or self.end_token_id in target[:-1]:
            raise ValueError("Invalid prompt or an embedded action end marker")
        if len(prompt) + len(target) > self.context_length:
            raise ValueError("Canonical state exceeds context_length; context truncation would change the state")
        return prompt, target

    def _logits(self, batch, reference=False):
        if reference and not hasattr(self.model, "disable_adapter"):
            raise ValueError("A frozen reference requires a PEFT policy")
        context = self.model.disable_adapter() if reference else nullcontext()
        kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                  "position_ids": batch["position_ids"], "use_cache": False}
        if self.supports_logit_limit:
            kwargs["logits_to_keep"] = batch["max_target"] + 1
        with context:
            return self.model(**kwargs).logits

    def prepare(self, rows):
        torch = self.torch
        encoded = [self.encode(row["messages"], row["action"]) for row in rows]
        width = max(len(prompt) + len(target) for prompt, target in encoded)
        device = self.model.get_input_embeddings().weight.device
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        for index, (prompt, target) in enumerate(encoded):
            length = len(prompt) + len(target)
            ids[index, -length:] = torch.tensor(prompt + target, device=device)
            attention[index, -length:] = 1
        positions = (attention.cumsum(-1) - 1).clamp(min=0)
        return {"input_ids": ids, "attention_mask": attention, "position_ids": positions,
                "targets": [target for _, target in encoded],
                "max_target": max(len(target) for _, target in encoded)}

    def score(self, rows, with_kl=False):
        torch = self.torch
        batch = self.prepare(rows)
        current = self._logits(batch)
        reference = None
        if with_kl:
            with torch.no_grad():
                reference = self._logits(batch, reference=True)
        scores, divergences = [], []
        for index, target in enumerate(batch["targets"]):
            length = len(target)
            logits = current[index, -length - 1:-1].float()
            labels = torch.tensor(target, device=logits.device)
            log_probability = logits.log_softmax(-1).gather(-1, labels[:, None]).sum()
            scores.append(log_probability)
            if reference is not None:
                divergences.append(token_reference_kl(reference[index, -length - 1:-1], logits).mean())
        return torch.stack(scores), torch.stack(divergences) if divergences else None

    def generate(self, messages, max_new_tokens=512):
        torch = self.torch
        prompt = self._prompt(messages)
        if len(prompt) + max_new_tokens > self.context_length:
            raise ValueError("Continuation exceeds the model context budget")
        device = self.model.get_input_embeddings().weight.device
        inputs = torch.tensor([prompt], device=device)
        with torch.no_grad():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         max_new_tokens=max_new_tokens, do_sample=False,
                                         eos_token_id=self.end_token_id, pad_token_id=self.tokenizer.pad_token_id)
        generated = output[0, len(prompt):].tolist()
        if not generated or generated[-1] != self.end_token_id:
            raise ValueError("Policy did not emit a complete action within max_new_tokens")
        return self.tokenizer.decode(generated[:-1], skip_special_tokens=False)
