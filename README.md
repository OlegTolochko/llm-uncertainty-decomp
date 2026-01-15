# Uncertainty Decomposition in Large Language Models

This repository contains a Python implementation of clarification-based uncertainty quantification methods for LLMs, focusing on detecting ambiguous questions.
Specifically it supports the AmbigQA dataset, extension to other datasets should be done based on the dataset specifications.

## Methods
Setup based on the paper "Fine-Grained Uncertainty Decomposition in Large Language Models: A Spectral Approach" by Walha et al.

### Base Pipeline for Data Generation
Based on the idea that ambiguous questions can be interpreted in multiple ways. Pipeline:
1. Given a question, generate multiple clarifications (different interpretations)
2. For each clarification, sample multiple answers from the target LLM

### Input Clarification Ensembling

3. Cluster answers by semantic equivalence using an NLI model (e.g. DeBERTa)
4. Compute entropy over the cluster assignment distribution

Measures how spread out the answers are across different semantic meanings.

### Spectral Uncertainty

3. Compute sentence embeddings for all answers (e.g. using the all-mpnet-base-v2 model)
4. Compute RBF kernel similarity matrices over answer embeddings
5. Perform eigenvalue decomposition to RBF kernel to decompose uncertainty into aleatoric and epistemic uncertainty

## Installation

```bash
uv sync
```

## Usage

### 1. Generate answers from AmbigQA dataset

```bash
uv run main.py ambigqa --split dev --limit 100 --output-file out/results.json
```

Options:
- `--split`: Dataset split ("dev" or "train")
- `--start-index`: Starting index in the dataset
- `--limit`: Number of items to process
- `--output-file`: Path to save results
- `--target-llm`: LLM for answering (default: microsoft/phi-4)
- `--clarification-llm`: LLM for generating clarifications (default: openai/gpt-4o)
- `--m`: Number of answer samples per clarification (default: 10)
- `--temperature`: Sampling temperature (default: 0.5)

### 2. Evaluate ambiguity detection

```bash
uv run main.py eval --input-file out/results.json --methods spectral --methods ice
```

Options:
- `--input-file`: Path to results JSON file
- `--methods`: Which methods to run ("spectral", "ice", or both)

## Project Structure

- `pipeline.py` - Main pipeline for generating clarifications and answers
- `eval_ambiguitiy.py` - Evaluation functions for ambiguity detection
- `spectral_uncertainty.py` - Spectral uncertainty computation
- `semantic_entropy.py` - Semantic clustering and entropy computation
- `ambigqa.py` - AmbigQA dataset loading
- `llm_inference.py` - LLM inference wrapper

## References

- Walha et al. "Fine-Grained Uncertainty Decomposition in Large Language Models: A Spectral Approach"
- Kuhn et al. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation"