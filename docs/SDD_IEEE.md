# Software Design Description (IEEE 1016 Style)

**System:** CAIROS Daemon (GitHub-Ready Subset)

**Document ID:** CAIROS-SDD-TRACK-AB

**Version:** 1.2

**Date:** 2026-02-17

## 1. Introduction

### 1.1 Purpose
This document describes the design of the public, GitHub-ready subset of the CAIROS daemon focused on **Track A (Python)** and **Track B (C++/ONNX)** pipelines. It covers architecture, components, interfaces, data structures, and REPA-aligned operational constraints.

### 1.2 Scope
- Track A: Python metrics pipeline using IBM Granite embeddings.
- Track B: C++/ONNX pipeline (embedding + metrics + hidden-state export).
- SecureLogger v2.0 is referenced but **not included** (private protocol).

### 1.3 Definitions, Acronyms, Abbreviations
- **Track A:** Python pipeline producing RECP metrics with anchor-centroid support.
- **Track B:** C++/ONNX pipeline producing RECP metrics with parity targets.
- **ICS:** Identity Consistency Score.
- **API:** Anchor Persistence Index.
- **ECR:** Entropy Collapse Rate.
- **LDI:** Loop Divergence Index.
- **SRV:** Signal Recursion Variance.
- **TCDM:** Token-to-word statistical variance (token count vs word count) + Welch t-test.
- **SDA:** Softmax Drift Attribution (logits-dependent).
- **ASV:** Anchor Skew Value (logits-dependent).
- **REPA:** Research Evidence & Provenance Alignment protocol.

## 2. System Overview
The system computes RECP metrics over contextual and non-contextual inputs, using IBM Granite embeddings. Track A executes via Python and the Granite loader; Track B uses ONNX Runtime to embed and compute metrics in C++.

## 3. Design Considerations
- **Determinism:** No silent fallbacks. Missing inputs or models raise errors.
- **Reproducibility:** When SecureLogger is enabled, inputs and outputs are hashed.
- **Portability:** No local absolute paths or private datasets.
- **Security:** SecureLogger is private; hooks exist but are disabled by default.
- **REPA Alignment:** No fabricated values. Missing dependencies result in `null` outputs and explicit notes.
- **Good Practice (Tooling):** Explicit compiler/toolchain versions are documented; optional components are clearly labeled; CUDA/OpenCV are not required for core metric pipelines.

## 4. Architecture

### 4.1 High-Level Components
- **Python Track A** (`python/track_a_ctxonoff_metrics.py`)
- **Granite Loader** (`python/granite_loader.py`, `python/model_hub.py`)
- **C++ Track B** (`native/`)
  - `granite_embedder` (ONNX + SentencePiece)
  - `recp_metrics` (core metric formulas)
  - CLIs: `granite_embedder_cli`, `granite_hidden_state_cli`, `identity_anchor_cli`, `recp_metrics_cli`
  - `granite_tokenizer_cli` (SentencePiece tokenizer)
  - `coords_from_embeddings_cli` (PCA coords converter)

### 4.2 Data Flow (Track A)
1. Parse contextual/non-contextual turns.
2. Embed turns and anchors via Granite loader.
3. Compute metrics: ICS, API, ECR, LDI, SRV, TCDM.
4. Write JSON output (+ optional SecureLogger metadata).

### 4.3 Data Flow (Track B)
1. Embed turns and anchors via ONNX Runtime.
2. Compute metrics using `recp_metrics`.
3. Optionally export hidden-state or logits (if model output exposes token-level tensors).

## 5. Data Design

### 5.1 Input Formats
- **Turns:** `turn_id|speaker|text` per line.
- **Anchors:** one anchor per line; Track A also accepts JSON with `anchors` array.

### 5.2 Output Formats
- **Track A metrics JSON:** `contextual`, `non_contextual`, `security`.
- **Track B metrics JSON:** `metrics`, `availability`, `embedding_source`.

### 5.3 Provenance Metadata (Track A)
- `security.sha256_inputs` hashes inputs and derived turn files.
- `security.sha256_outputs` hashes embedding JSONLs.
- `security.metric_notes` records metric derivations and missing-dependency notes.

## 6. Interface Design

### 6.1 Python CLI (Track A)
```
python python/track_a_ctxonoff_metrics.py \
  --run-dir <dir> --ctx-on <txt> --ctx-off <txt> \
  --anchors <txt_or_json> --model-dir <model_dir>
```

### 6.2 C++ CLIs (Track B)
```
recp_metrics_cli --turns <txt> --output <json> [--anchors <txt>] \
  [--embeddings <jsonl> | --model-dir <dir>]

granite_hidden_state_cli --model-dir <dir> --input <txt> --output <jsonl>

coords_from_embeddings_cli --input-root <dir_with_jsonl> --out-dir <out> --dims 3 [--anchors <anchors_jsonl>]
```

## 7. Component Design

### 7.1 Track A Metrics (Python)
- **ICS (primary):** mean cosine between each turn embedding and the **weighted anchor centroid**.
- **ICS_pairwise (audit):** mean cosine across all turn pairs (stored in `security.metric_notes`).
- **API:** anchor phrase persistence across turns.
- **ECR:** entropy collapse ratio from first to last turn.
- **LDI:** average pairwise divergence.
- **SRV:** variance of k-lag drift values.
- **TCDM:** token-to-word statistical variance (token count vs word count) + optional Welch t-test.
  If token-counts JSONL is provided, TCDM uses **model-consistent tokenization** via SentencePiece.
  Legacy char/word variance is deprecated and should not be used for REPA evidence.
- **Logits-dependent metrics:** SDA and ASV are unavailable in this repo; outputs are `null` with explicit notes.

### 7.2 Track B Metrics (C++)
`recp_metrics` mirrors Track A formulas. If anchors exist, `ics_anchor` is computed and used as primary `ics`; `ics_pairwise` is retained as an audit field.

### 7.3 Coords Converter (C++)
- Computes 2D/3D PCA coordinates from RECP JSONL embeddings.
- Uses May embeddings as PCA basis; projects Feb and anchors into the same space.
- Outputs CSV/JSON coords plus PCA metadata (mean/components/minmax).

### 7.4 Tokenizer (C++)
- `granite_tokenizer_cli` produces per-line token counts using SentencePiece from the model directory.
- Output JSONL can be fed into Track A (`--token-counts-ctxon/ctxoff`) or Track B (`--token-counts`).

## 8. Error Handling
- Missing model files or embeddings: hard error.
- Missing anchors: ICS falls back to pairwise; notes indicate anchor absence.
- Missing optional dependencies (e.g., `scipy`): corresponding values are `null` with explicit notes.
- SecureLogger enabled without key: hard error (fail-closed for secure logging).

## 9. Security & REPA Alignment
- SecureLogger v2.0 is **private** and not included in this repository.
- When SecureLogger is not present, the pipeline still computes metrics and marks security fields as unavailable.
- No metric values are fabricated. All omissions are explicit and noted.

## 10. Build & Deployment
See `docs/DEPENDENCIES.md` for versions and `README.md` for build steps.

## 11. Assumptions and Constraints
- IBM Granite-Embedding-278M model is available locally.
- ONNX Runtime and SentencePiece are installed for Track B.
- SecureLogger is optional and external.

## 12. Traceability Matrix (Summary)
- **ICS anchor-centroid** → Track A `compute_metrics` + Track B `recp_metrics`.
- **Provenance hashes** → Track A `security.sha256_inputs` / `security.sha256_outputs`.
- **REPA notes** → Track A `security.metric_notes`.

---

## 13. REPA Amendment Log
- 2026-02-17: TCDM discrepancy identified (legacy char/word variance). \
Amended to model-consistent token counts via granite_tokenizer_cli + token-count overrides in Track A/B. \
See test run: `D:\ChasingBlu_RND\Lab\final_run\LOCKED_TRACK_A_B_Final_20260217\TCDM_TOKEN_TEST_20260217`.

## 14. SecureLogger (Closed-Source) — Workflow Summary
SecureLogger v2.0 is **closed-source** and **not distributed**. This document provides only a high-level workflow description and **does not disclose implementation details, cryptographic parameters, or code snippets**.

Workflow (high-level):
- **Track B (`recp_metrics_cli`) is fail-closed** when SecureLogger is required; a key + log directory must be provided or the run aborts.
- **Track A (Python)** runs without SecureLogger unless explicitly enabled.
- When SecureLogger is enabled, the pipeline records **tamper-evident hashes** of inputs/outputs and writes an audit trail to a secure log directory.
- Any run performed **without** SecureLogger keys/signatures is considered **external/unverified** and **will not be attributed to ChasingBlu R&D Labs**.

---
**Signed:** Codex — Systems Engineering Assistant