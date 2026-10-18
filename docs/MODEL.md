# Model Card — Fine-tuned DeBERTa Hallucination Verifier

## Summary

| Property | Value |
|----------|-------|
| **Architecture** | DeBERTa-base (microsoft/deberta-base-mnli) |
| **Task** | Binary sequence classification — response-level hallucination detection |
| **Training dataset** | RAGTruth |
| **Input format** | `[evidence, response]` pair (max 384 tokens, longest-first truncation) |
| **Output** | `NO_ANNOTATED_HALLUCINATION` (0) / `HALLUCINATION` (1) |
| **Decision threshold** | 0.4997 (selected on validation set) |
| **Training duration** | 3 epochs, lr = 2e-5 |

---

## Performance

### Full test set (n = 2,626)

| Metric | Fine-tuned model |
|--------|-----------------|
| Accuracy | 78.0 % |
| Precision (hallucination) | 66.1 % |
| Recall (hallucination) | 72.8 % |
| F1 (hallucination) | 69.3 % |
| Macro F1 | 76.1 % |
| ROC-AUC | **0.843** |
| PR-AUC | 0.758 |

### Comparison against zero-shot baseline (shared subset, n = 500)

| Metric | Baseline (zero-shot) | Fine-tuned |
|--------|---------------------|------------|
| Accuracy | 54.0 % | **78.4 %** |
| F1 (hallucination) | 58.3 % | **70.5 %** |
| ROC-AUC | 0.674 | **0.851** |

The fine-tuned model outperforms the zero-shot baseline by a large margin on every metric.

---

## Labels

| Label ID | Label name | Definition |
|----------|-----------|------------|
| 0 | `NO_ANNOTATED_HALLUCINATION` | Response has no annotated hallucination spans |
| 1 | `HALLUCINATION` | Response contains one or more non-implicit hallucination spans |

Responses where all annotations were marked `implicit_true` were excluded from training and evaluation.

---

## Data splits

| Split | Size |
|-------|------|
| Train | 12,898 |
| Validation | 1,443 |
| Test | 2,626 |
| Baseline shared subset | 500 |

---

## Limitations

1. **Response-level only** — The model outputs a single verdict per response. It does not
   identify *which* specific sentence or phrase contains the hallucination.

2. **Domain generalisation** — Trained on RAGTruth (QA and summarisation tasks).
   Performance may degrade on domains very different from the training data.

3. **Max length** — Input is truncated at 384 tokens. Very long documents or responses
   may have context cut off.

4. **No factual knowledge** — The model compares evidence to response using learned
   linguistic patterns, not an internal knowledge base. If the evidence itself is wrong,
   the model will still classify the response as consistent.

5. **False positives** — Precision is 66 %. One in three "hallucination" flags may be
   incorrect. Always treat the output as a signal to investigate, not a definitive verdict.

---

## Model files

| File | Description |
|------|-------------|
| `config.json` | Architecture configuration (DeBERTa-base, 2 labels) |
| `model.safetensors` | Fine-tuned weights in SafeTensors format |
| `tokenizer.json` | Tokenizer vocabulary and rules |
| `tokenizer_config.json` | Tokenizer class and special token mappings |
| `vocab.json` | Vocabulary (GPT-2 style BPE) |
| `merges.txt` | BPE merge rules |
| `special_tokens_map.json` | PAD / CLS / SEP / UNK / MASK token definitions |
| `evaluation_metrics.json` | Full evaluation results and selected threshold |
| `training_args.bin` | Hugging Face Trainer arguments used during fine-tuning |
| `test_predictions.csv` | Raw model predictions on the test set |
| `baseline_comparison.csv` | Side-by-side baseline vs. fine-tuned on shared subset |
