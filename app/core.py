"""Quantized Legal Tagger Monitor — Milestones 1 through 6."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import tempfile
import time

from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

ARTIFACTS_DIR = Path("artifacts")
SUBSET_PATH = ARTIFACTS_DIR / "ledgar_subset.json"
FP32_MODEL_DIR = ARTIFACTS_DIR / "gpt2_legal_fp32"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_NAME = "distilgpt2"


class ClauseTagDataset(Dataset):
    """Tokenizes 'Clause: ...\\nTag: <label>' sequences for causal LM fine-tuning."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int = 192):
        self.encodings = []
        for item in samples:
            clause_words = item["text"].split()[:110]
            prompt = f"Clause: {' '.join(clause_words)}\nTag: {item['label_name']}{tokenizer.eos_token}"
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.encodings.append(
                {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            )

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encodings[idx]


class QuantizableLinear(nn.Module):
    """Wraps nn.Linear with QuantStub/DeQuantStub for native static quantization."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.linear = linear
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.linear(self.quant(x)))


class Int8QuantizedLinear(nn.Module):
    """Fast INT8 linear layer used on Windows where FBGEMM/QNNPACK kernels are unavailable."""

    def __init__(self, linear: nn.Linear, static: bool = False):
        super().__init__()
        self.static = static
        w = linear.weight.detach().float()
        w_scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        w_int8 = torch.clamp(torch.round(w / w_scale), -128, 127).to(torch.int8)
        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("weight_scale", w_scale)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().float())
        else:
            self.bias = None
        self.register_buffer("act_scale", torch.tensor(1.0 / 127.0))
        self.calibrating = static
        self._max_act = 0.0

    def finish_calibration(self) -> None:
        self.calibrating = False
        if self._max_act > 0:
            self.act_scale.fill_(max(self._max_act / 127.0, 1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self._max_act = max(self._max_act, float(x.detach().abs().max().item()))
            w_dequant = self.weight_int8.to(x.dtype) * self.weight_scale.to(x.dtype)
            return nn.functional.linear(x, w_dequant, self.bias)

        if self.static:
            a_scale = self.act_scale.to(x.dtype)
        else:
            a_scale = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-8)

        x_int8 = torch.clamp(torch.round(x / a_scale), -128, 127).to(torch.int8)
        x_dequant = x_int8.to(x.dtype) * a_scale
        w_dequant = self.weight_int8.to(x.dtype) * self.weight_scale.to(x.dtype)
        return nn.functional.linear(x_dequant, w_dequant, self.bias)


def _convert_conv1d_to_linear(
    module: nn.Module,
    wrap_for_static: bool = False,
    fallback_static: bool | None = None,
) -> nn.Module:
    """Replaces Hugging Face Conv1D layers with nn.Linear or INT8 quantized linear layers."""
    for name, child in list(module.named_children()):
        if isinstance(child, Conv1D):
            in_features, out_features = child.weight.shape
            linear = nn.Linear(in_features, out_features, bias=child.bias is not None)
            with torch.no_grad():
                linear.weight.copy_(child.weight.t())
                if child.bias is not None:
                    linear.bias.copy_(child.bias)
            if fallback_static is not None:
                replacement = Int8QuantizedLinear(linear, static=fallback_static)
            elif wrap_for_static:
                replacement = QuantizableLinear(linear)
            else:
                replacement = linear
            setattr(module, name, replacement)
        else:
            _convert_conv1d_to_linear(
                child, wrap_for_static=wrap_for_static, fallback_static=fallback_static
            )
    return module


def _configure_quant_backend() -> str | None:
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    # Skip 'onednn' on Windows CPUs without VNNI because it can stall during generation
    for candidate in ("fbgemm", "x86", "qnnpack"):
        if candidate in supported:
            torch.backends.quantized.engine = candidate
            return candidate
    return None


def _measure_model_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model.state_dict(), tmp_path)
        return os.path.getsize(tmp_path) / (1024 ** 2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _evaluate_model(model: nn.Module, tokenizer, test_samples: list[dict], label: str = "") -> dict:
    """Runs greedy tag generation on test clauses and measures accuracy + latency."""
    model.eval()
    correct = 0
    latencies_ms = []
    total = max(len(test_samples), 1)

    with torch.no_grad():
        for idx, item in enumerate(test_samples, start=1):
            clause_words = item["text"].split()[:110]
            prompt = f"Clause: {' '.join(clause_words)}\nTag:"
            inputs = tokenizer(prompt, return_tensors="pt")

            start = time.perf_counter()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            predicted_label = generated.splitlines()[0].strip().lower() if generated else ""
            expected_label = item["label_name"].strip().lower()

            if expected_label in predicted_label or predicted_label == expected_label:
                correct += 1

            if label and idx % 25 == 0:
                print(f"  [{label}] Evaluated {idx}/{total} clauses...")

    avg_latency = sum(latencies_ms) / total
    return {
        "accuracy": round((correct / total) * 100.0, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "throughput_clauses_per_sec": round(1000.0 / max(avg_latency, 1e-6), 2),
    }


# --- Milestone 1: Prepare a subset of the LEDGAR dataset for legal clause tagging.
def milestone_1(num_labels: int = 5, train_per_label: int = 60, test_per_label: int = 15) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    label_names = ds["train"].features["label"].names

    counts = Counter(ds["train"]["label"])
    top_label_ids = [lbl for lbl, _ in counts.most_common(num_labels)]

    def sample_split(split_data, per_label_limit: int) -> list[dict]:
        collected = {lbl: [] for lbl in top_label_ids}
        for row in split_data:
            lbl = row["label"]
            if lbl in collected and len(collected[lbl]) < per_label_limit:
                collected[lbl].append(
                    {"text": row["text"], "label": lbl, "label_name": label_names[lbl]}
                )
            if all(len(v) >= per_label_limit for v in collected.values()):
                break
        return [item for group in collected.values() for item in group]

    subset = {
        "labels": [label_names[i] for i in top_label_ids],
        "train": sample_split(ds["train"], train_per_label),
        "test": sample_split(ds["test"], test_per_label),
    }
    SUBSET_PATH.write_text(json.dumps(subset, indent=2))
    return {"labels": subset["labels"], "train_count": len(subset["train"]), "test_count": len(subset["test"])}


# --- Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using classification as generation.
def milestone_2(epochs: int = 2, batch_size: int = 8, lr: float = 5e-5) -> dict:
    if not SUBSET_PATH.exists():
        milestone_1()
    subset = json.loads(SUBSET_PATH.read_text())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    train_dataset = ClauseTagDataset(subset["train"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    final_loss = 0.0
    for _ in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        final_loss = total_loss / max(len(train_loader), 1)

    model.cpu()
    FP32_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FP32_MODEL_DIR)
    tokenizer.save_pretrained(FP32_MODEL_DIR)
    return {"model_dir": str(FP32_MODEL_DIR), "epochs": epochs, "final_train_loss": round(final_loss, 4)}


# --- Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
def milestone_3() -> dict:
    if not FP32_MODEL_DIR.exists():
        milestone_2()
    subset = json.loads(SUBSET_PATH.read_text())
    tokenizer = AutoTokenizer.from_pretrained(FP32_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(FP32_MODEL_DIR)

    eval_stats = _evaluate_model(model, tokenizer, subset["test"], label="FP32")
    eval_stats["size_mb"] = round(_measure_model_size_mb(model), 2)
    return eval_stats


# --- Milestone 4: Apply dynamic and static post-training quantization to the trained model.
def milestone_4() -> tuple[nn.Module, nn.Module]:
    if not FP32_MODEL_DIR.exists():
        milestone_2()
    engine = _configure_quant_backend()
    subset = json.loads(SUBSET_PATH.read_text())
    tokenizer = AutoTokenizer.from_pretrained(FP32_MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(FP32_MODEL_DIR).cpu().eval()

    print("Applying dynamic and static INT8 quantization...")
    if engine is not None:
        dyn_prep = _convert_conv1d_to_linear(copy.deepcopy(base_model), wrap_for_static=False)
        dynamic_model = torch.ao.quantization.quantize_dynamic(
            dyn_prep, {nn.Linear}, dtype=torch.qint8
        )

        static_prep = _convert_conv1d_to_linear(copy.deepcopy(base_model), wrap_for_static=True)
        static_prep.eval()
        qconfig = torch.ao.quantization.get_default_qconfig(engine)
        for mod in static_prep.modules():
            if isinstance(mod, QuantizableLinear):
                mod.qconfig = qconfig

        torch.ao.quantization.prepare(static_prep, inplace=True)
        with torch.no_grad():
            for item in subset["train"][:24]:
                prompt = f"Clause: {' '.join(item['text'].split()[:110])}\nTag:"
                inputs = tokenizer(prompt, return_tensors="pt")
                static_prep(**inputs)
        static_model = torch.ao.quantization.convert(static_prep, inplace=False)
        return dynamic_model, static_model

    dynamic_model = _convert_conv1d_to_linear(copy.deepcopy(base_model), fallback_static=False)
    static_model = _convert_conv1d_to_linear(copy.deepcopy(base_model), fallback_static=True)
    with torch.no_grad():
        for item in subset["train"][:24]:
            prompt = f"Clause: {' '.join(item['text'].split()[:110])}\nTag:"
            inputs = tokenizer(prompt, return_tensors="pt")
            static_model(**inputs)
    for mod in static_model.modules():
        if isinstance(mod, Int8QuantizedLinear):
            mod.finish_calibration()
    return dynamic_model, static_model


# --- Milestone 5: Compare original and quantized model sizes and simulated inference times.
def milestone_5() -> list[dict]:
    if not FP32_MODEL_DIR.exists():
        milestone_2()
    subset = json.loads(SUBSET_PATH.read_text())
    tokenizer = AutoTokenizer.from_pretrained(FP32_MODEL_DIR)
    fp32_model = AutoModelForCausalLM.from_pretrained(FP32_MODEL_DIR).cpu().eval()
    dynamic_model, static_model = milestone_4()

    variants = [
        ("Original (FP32)", fp32_model),
        ("Dynamic Quantized (INT8)", dynamic_model),
        ("Static Quantized (INT8)", static_model),
    ]

    results = []
    base_size = None
    base_latency = None

    for name, mod in variants:
        print(f"Benchmarking {name}...")
        size_mb = round(_measure_model_size_mb(mod), 2)
        stats = _evaluate_model(mod, tokenizer, subset["test"], label=name)
        if base_size is None:
            base_size = size_mb
            base_latency = stats["avg_latency_ms"]

        size_reduction = round((1.0 - (size_mb / base_size)) * 100.0, 1)
        speedup = round(base_latency / max(stats["avg_latency_ms"], 1e-6), 2)
        results.append(
            {
                "variant": name,
                "size_mb": size_mb,
                "size_reduction_pct": size_reduction,
                "avg_latency_ms": stats["avg_latency_ms"],
                "speedup_x": speedup,
                "accuracy_pct": stats["accuracy"],
            }
        )

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    return results


# --- Milestone 6: Present a dashboard-like output of model efficiency metrics.
def milestone_6() -> str:
    results = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else milestone_5()

    header = (
        f"\n{'Model Variant':<26} | {'Size (MB)':>10} | {'Size Saved':>10} | "
        f"{'Latency (ms)':>12} | {'Speedup':>8} | {'Accuracy':>9}"
    )
    divider = "-" * len(header)
    rows = [
        "\n=== Quantized Legal Tagger Efficiency Dashboard ===",
        header,
        divider,
    ]
    for r in results:
        rows.append(
            f"{r['variant']:<26} | {r['size_mb']:>9.2f}M | {r['size_reduction_pct']:>9.1f}% | "
            f"{r['avg_latency_ms']:>11.2f} | {r['speedup_x']:>7.2f}x | {r['accuracy_pct']:>8.2f}%"
        )
    rows.append(divider)

    dashboard = "\n".join(rows)
    print(dashboard)
    return dashboard