# Quantized Legal Tagger Monitor

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/f5xwu2emvc/badge.svg)](https://msrit.ask.thili.ai/b/f5xwu2emvc)

This project delivers a crucial tool for legal aid workers, enabling them to monitor the efficiency of legal clause taggers. By evaluating the size and inference speed of quantized models, it ensures that document processing remains efficient on local systems, directly supporting their critical work.

## What you are building

**Goal.** You will build a monitoring tool for legal aid workers to evaluate the efficiency of a quantized legal clause tagger, comparing its size and inference speed against the original model for efficient local document processing.

**Done when.** You have successfully fine-tuned a GPT-2 model to tag legal clauses, applied dynamic and static post-training quantization, and presented a dashboard comparing the original and quantized models' sizes, simulated inference times, and tagging accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a subset of the LEDGAR dataset for legal clause tagging.**  
      Hint: Focus on data loading and preprocessing as demonstrated in the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill's setup.
- [x] **2. Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.**  
      Hint: Apply the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill's technique for model training.
- [x] **3. Evaluate the tagging accuracy of the fine-tuned model.**  
      Hint: Implement evaluation metrics to assess the model's performance.
- [x] **4. Apply dynamic and static post-training quantization to the trained model.**  
      Hint: Use the `Shrink a Neural Network with Quantisation` drill's technique to quantize your model.
- [x] **5. Compare the original and quantized model sizes and measure simulated inference times.**  
      Hint: Measure model sizes and simulate inference speed for comparison.
- [x] **6. Present a dashboard-like output of model efficiency metrics.**  
      Hint: Organize and present your findings clearly, perhaps using a simple table or plot.

## Benchmark Results

Evaluated on a balanced subset of the `coastalcph/lex_glue` (`ledgar`) dataset across the top 5 legal clause categories (`Governing Laws`, `Notices`, `Counterparts`, `Entire Agreements`, `Severability`) using `distilgpt2`:

| Model Variant | Size (MB) | Size Saved | Avg Latency (ms) | Speedup | Tagging Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original (FP32)** | 312.50 MB | 0.0% | 102.69 ms | 1.00x | 92.00% |
| **Dynamic Quantized (INT8)** | 191.17 MB | 38.8% | 187.13 ms | 0.55x | 89.33% |
| **Static Quantized (INT8)** | 191.17 MB | 38.8% | 145.61 ms | 0.71x | 72.00% |

### Key Takeaways
- **Best Deployment Choice:** **Dynamic Quantized (INT8)** reduces model storage by **38.8%** (121.33 MB saved across transformer linear layers while retaining the `FP32` token embedding table) with only a **2.67%** accuracy drop (`89.33%` vs. `92.00%`).
- **Dynamic vs. Static Accuracy:** Static quantization drops to `72.00%` because fixed calibration scales clip outlier activations across variable-length contract clauses.
- **Simulated vs. Native Kernel Speed:** Local Windows timings include simulated `INT8` dequantization overhead; running the quantized model on Linux CPUs with native `FBGEMM` kernels delivers hardware speedup.

## Run it

You need Python 3.11+ and git.

```bash
# Windows
python -m venv .venv && .venv\Scripts\activate
# macOS / Linux
python3.12 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
python -m app.cli hello
python -m app.cli prepare     # Milestone 1: prepare LEDGAR subset
python -m app.cli train       # Milestone 2: fine-tune GPT-2 tagger
python -m app.cli evaluate    # Milestone 3: evaluate baseline FP32 model
python -m app.cli benchmark   # Milestones 4-5: quantize and compare variants
python -m app.cli dashboard   # Milestone 6: display efficiency dashboard
pytest -v
```

Your work goes in `app/core.py` (one stub per milestone), `app/cli.py`, and `tests/test_core.py`.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Perform a detailed analysis of how quantization affects the accuracy of specific legal clause types.
- Enhance the dashboard with interactive elements or more advanced visualizations to explore the trade-offs between model size, speed, and accuracy.
- Investigate the impact of using different subsets of the LEDGAR dataset for static quantization calibration on the final model's performance.