# BERT Triton Inference Server (Tokenizer + ONNX + Ensemble)

This project packages a BERT CLS-embedding pipeline on NVIDIA Triton using:

* Python backend for tokenization
* ONNX Runtime for the BERT model
* Triton Ensemble to expose a single `TEXT -> EMBEDDINGS` API

## Architecture

```
TEXT
  │
  ▼
bert_tokenizer (Python backend)
  └─> input_ids, attention_mask, token_type_ids
  │
  ▼
bert_model (ONNX Runtime)
  └─> cls_output (768-dim)
  │
  ▼
bert_ensemble
  └─> EMBEDDINGS
```

## Requirements

* Docker
* NVIDIA GPU (optional, CPU works)
* Python 3.12 (for export)

## Export Model & Tokenizer

```bash
make save-onnx-model
```

This will:

* Download `bert-base-uncased`
* Save tokenizer to `model_repository/bert_tokenizer/1`
* Export CLS-only BERT to ONNX in `model_repository/bert_model/1/model.onnx`
* Set dynamic batch & sequence length
* Downgrade ONNX IR for Triton compatibility

## Run Triton

```bash
docker build -t bert-triton .
docker run --rm -p 8000:8000 -p 8001:8001 bert-triton
```

Triton loads models from `/models` and exposes:

* HTTP: `localhost:8000`
* gRPC: `localhost:8001`

## **Notes**

* Max sequence length: 512 (set in tokenizer backend)
* Dynamic batching supported (Triton `max_batch_size: 32`)
* CLS embedding = `last_hidden_state[:, 0, :]`
* Easily swappable with any HuggingFace BERT-like encoder

## Files of Interest

* `scripts/export_model.py` – ONNX export + tokenizer save
* `model_repository/bert_tokenizer` – Python backend
* `model_repository/bert_model` – ONNX BERT
* `model_repository/bert_ensemble` – End-to-end pipeline config
