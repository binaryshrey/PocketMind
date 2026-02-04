# PocketMind

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

Personal Documents → Chunking → Local Retrieval → On-Device LLM → Controlled Generation → Summary

## Model Choice

**Qwen2.5-3B-Instruct**

- Instruction-tuned for clean summaries
- Strong technical summarization
- Suitable for edge deployment when quantized

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
- Context size
- Hallucination rate
- Tradeoff plots (latency vs context)

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
