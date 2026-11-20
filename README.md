# Quantized Legal Tagger Monitor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/f5xwu2emvc/badge.svg)](https://ask.thili.ai/b/f5xwu2emvc)

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

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Perform a detailed analysis of how quantization affects the accuracy of specific legal clause types.
- Enhance the dashboard with interactive elements or more advanced visualizations to explore the trade-offs between model size, speed, and accuracy.
- Investigate the impact of using different subsets of the LEDGAR dataset for static quantization calibration on the final model's performance.
