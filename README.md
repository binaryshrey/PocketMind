# PocketMind [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/binaryshrey/PocketMind/blob/main/PocketMind.ipynb)

### Privacy-Focused On-Device Personal Knowledge Summarization Application

## Overview

**PocketMind** is a privacy-first, on-device natural language processing system that summarizes personal documents using a lightweight large language model (LLM). The system operates entirely locally, without cloud APIs, making it suitable for privacy-sensitive use cases such as personal notes, research papers, and private documents.

## Problem Statement

Modern LLM-based summarization systems rely heavily on cloud inference, introducing privacy risks, latency and lack of user control. Additionally, personal documents often exceed the limited context windows of on-device models, increasing hallucination risk.

**Goal:**  
Design an on-device summarization system that preserves privacy, handles long documents via retrieval-based context selection and produces concise, faithful summaries.

## Key Design Principles

- Privacy-first (no cloud calls)
- On-device realism
- Faithfulness over fluency
- System-level ML design
- Deterministic, bounded outputs

## System Architecture

```Personal Documents → Chunking → Local Retrieval → On-Device LLM → Controlled Generation → Summary```

## Model Choice

**Qwen2.5-3B-Instruct**

- Instruction-tuned for clean summaries
- Strong technical summarization
- Suitable for edge deployment when quantized

```
"""
Qwen2.5-3B-Instruct
- Instruction-tuned → clean summaries
- Strong technical summarization
- Stable generation
- Suitable for edge / on-device scenarios
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

model.eval()
```

## Data Handling

Supports TXT and PDF documents. All data remains local and is never logged or transmitted.

## Context Window Optimization

Documents are split into overlapping chunks (≈400 tokens, 50-token overlap). Only top-K relevant chunks are retrieved for summarization.

## Retrieval

- Sentence embeddings: all-MiniLM-L6-v2
- Similarity: cosine
- In-memory retrieval (no vector DB)

## Summarization Pipeline

- Single-shot instruction prompt
- Deterministic decoding
- Decode only newly generated tokens
- Hard post-generation truncation

## Hallucination Mitigation

A heuristic faithfulness metric estimates grounding by comparing summaries with retrieved chunks.

## Evaluation Metrics

- Latency
```
"""
Latency measurement

Measures end-to-end summarization time.
Simulates on-device responsiveness.
"""

import time

start = time.time()
_ = summarize(query)
latency = time.time() - start

print(f"Latency: {latency:.2f} seconds")
Latency: 14.57 seconds
```


- Context size
```
context_sizes = [2, 3, 4, 5]
```

- Hallucination rate
```
"""
Faithfulness metric (heuristic)

Purpose:
- Estimate how much of the summary is grounded
- Not perfect, but useful for trend analysis
"""

def hallucination_rate(summary, source_chunks):
    supported = 0
    for chunk in source_chunks:
        if any(word in summary.lower() for word in chunk.lower().split()[:30]):
            supported += 1
    return 1 - supported / max(1, len(source_chunks))
rate = hallucination_rate(summary, retrieved_chunks)
print(f"Hallucination rate (lower is better): {rate:.2f}")
Hallucination rate (lower is better): 0.00
```
- Tradeoff plots (latency vs context)
![Latency](https://raw.githubusercontent.com/binaryshrey/PocketMind/refs/heads/main/assets/latencyVcontextsize.png)
![Context size](https://github.com/binaryshrey/PocketMind/blob/main/assets/faithfullnessVcontextsize.png)

## Privacy by Design

- No cloud inference
- No APIs
- No telemetry
- All processing is local

## Limitations

- Heuristic hallucination detection
- No UI
- No personalization

## Future Work

- Quantization
- LoRA fine-tuning
- NLI-based faithfulness
- Mobile deployment
