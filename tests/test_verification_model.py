import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from clt.verification import NLIVerifier


@pytest.fixture
def local_verifier(tmp_path):
    path = tmp_path / "verifier"
    path.mkdir()
    (path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nevidence\nclaim\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(path / "vocab.txt"), model_max_length=32)
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    model = transformers.BertForSequenceClassification(transformers.BertConfig(
        vocab_size=7, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=32, num_labels=3,
        id2label={0: "contradiction", 1: "neutral", 2: "entailment"}))
    model.save_pretrained(path)
    return path


def test_local_verifier_identity_changes_with_weights(local_verifier):
    first = NLIVerifier(str(local_verifier), max_length=16)
    assert all(not parameter.requires_grad for parameter in first.model.parameters())
    assert first.identity["parameters_sha256"]
    with torch.no_grad():
        first.model.classifier.bias[2].add_(1)
    first.model.save_pretrained(local_verifier)
    second = NLIVerifier(str(local_verifier), max_length=16)
    assert first.identity["parameters_sha256"] != second.identity["parameters_sha256"]


def test_verifier_identity_binds_tokenization_and_context(local_verifier):
    from tokenizers.normalizers import BertNormalizer

    first = NLIVerifier(str(local_verifier), max_length=16)
    repeated = NLIVerifier(str(local_verifier), max_length=16)
    assert first.identity == repeated.identity
    first.tokenizer.backend_tokenizer.normalizer = BertNormalizer(lowercase=False)
    first.tokenizer.do_lower_case = False
    first.tokenizer.save_pretrained(local_verifier)
    changed = NLIVerifier(str(local_verifier), max_length=16)
    assert first.identity["tokenizer_backend"] != changed.identity["tokenizer_backend"]
    longer = NLIVerifier(str(local_verifier), max_length=24)
    assert changed.identity != longer.identity
