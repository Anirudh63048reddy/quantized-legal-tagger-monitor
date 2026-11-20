"""Unit tests covering Milestones 1 through 6."""
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from app import core


class _MockFeatures(dict):
    def __init__(self, names: list[str]):
        super().__init__()
        self.names = names


class _MockSplit(list):
    def __init__(self, rows: list[dict], label_names: list[str]):
        super().__init__(rows)
        self.features = {"label": _MockFeatures(label_names)}

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


@pytest.fixture()
def isolated_pipeline(tmp_path, monkeypatch):
    """Redirects artifacts and provides a tiny local GPT-2 + LEDGAR fixture for fast offline tests."""
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(core, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(core, "SUBSET_PATH", artifacts / "ledgar_subset.json")
    monkeypatch.setattr(core, "FP32_MODEL_DIR", artifacts / "gpt2_legal_fp32")
    monkeypatch.setattr(core, "METRICS_PATH", artifacts / "metrics.json")

    label_names = ["Governing Laws", "Notices"]
    train_rows = [
        {"text": "This agreement is governed by state law.", "label": 0},
        {"text": "Disputes shall follow local governing laws.", "label": 0},
        {"text": "All notices must be delivered in writing.", "label": 1},
        {"text": "Written notices shall be sent to counsel.", "label": 1},
    ]
    test_rows = [
        {"text": "Governed by the laws of Delaware.", "label": 0},
        {"text": "Notices must be sent by certified mail.", "label": 1},
    ]
    mock_ds = {
        "train": _MockSplit(train_rows, label_names),
        "test": _MockSplit(test_rows, label_names),
    }
    monkeypatch.setattr(core, "load_dataset", lambda *args, **kwargs: mock_ds)

    base_model_dir = tmp_path / "tiny_gpt2"
    vocab = {
        "<|endoftext|>": 0,
        "[UNK]": 1,
        "Clause:": 2,
        "Tag:": 3,
        "Governing": 4,
        "Laws": 5,
        "Notices": 6,
        "agreement": 7,
        "notices": 8,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    fast_tok.save_pretrained(base_model_dir)

    tiny_cfg = GPT2Config(
        vocab_size=len(vocab),
        n_positions=256,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    tiny_model = GPT2LMHeadModel(tiny_cfg)
    tiny_model.save_pretrained(base_model_dir)
    monkeypatch.setattr(core, "MODEL_NAME", str(base_model_dir))


# Milestone 1: Prepare a subset of the LEDGAR dataset for legal clause tagging.
def test_milestone_1(isolated_pipeline):
    summary = core.milestone_1(num_labels=2, train_per_label=2, test_per_label=1)
    assert summary["labels"] == ["Governing Laws", "Notices"]
    assert summary["train_count"] == 4
    assert summary["test_count"] == 2
    assert core.SUBSET_PATH.exists()


# Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.
def test_milestone_2(isolated_pipeline):
    core.milestone_1(num_labels=2, train_per_label=2, test_per_label=1)
    result = core.milestone_2(epochs=1, batch_size=2)
    assert core.FP32_MODEL_DIR.exists()
    assert isinstance(result["final_train_loss"], float)


# Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
def test_milestone_3(isolated_pipeline):
    core.milestone_1(num_labels=2, train_per_label=2, test_per_label=1)
    core.milestone_2(epochs=1, batch_size=2)
    stats = core.milestone_3()
    assert 0.0 <= stats["accuracy"] <= 100.0
    assert stats["avg_latency_ms"] > 0
    assert stats["size_mb"] > 0


# Milestone 4: Apply dynamic and static post-training quantization to the trained model.
def test_milestone_4(isolated_pipeline):
    core.milestone_1(num_labels=2, train_per_label=2, test_per_label=1)
    core.milestone_2(epochs=1, batch_size=2)
    dynamic_model, static_model = core.milestone_4()
    assert isinstance(dynamic_model, nn.Module)
    assert isinstance(static_model, nn.Module)


# Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
def test_milestone_5(isolated_pipeline):
    core.milestone_1(num_labels=2, train_per_label=2, test_per_label=1)
    core.milestone_2(epochs=1, batch_size=2)
    results = core.milestone_5()
    assert len(results) == 3
    assert core.METRICS_PATH.exists()
    assert results[1]["size_mb"] <= results[0]["size_mb"]
    assert results[2]["size_mb"] <= results[0]["size_mb"]


# Milestone 6: Present a dashboard-like output of model efficiency metrics.
def test_milestone_6(isolated_pipeline, capsys):
    core.milestone_1(num_labels=2, train_per_label=2, test_per_label=1)
    core.milestone_2(epochs=1, batch_size=2)
    core.milestone_5()
    output = core.milestone_6()
    captured = capsys.readouterr().out
    assert "Quantized Legal Tagger Efficiency Dashboard" in output
    assert "Original (FP32)" in captured
    assert "Dynamic Quantized (INT8)" in captured
    assert "Static Quantized (INT8)" in captured