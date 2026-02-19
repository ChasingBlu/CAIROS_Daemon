# CAIROS Daemon (GitHub-Ready Subset)

This repository is a minimal, public-ready subset of the CAIROS daemon focused on **Track A (Python)** and **Track B (C++/ONNX)** metric pipelines. It omits private components and local-only references.

**Included**
- Track A (Python): `python/track_a_ctxonoff_metrics.py`
- Track B (C++/ONNX): `native/` (embedder + metrics CLIs)
- Build metadata and IEEE SDD

**Not Included**
- SecureLogger v2.0 implementation (private protocol)
- Model weights (IBM Granite-Embedding-278M)
- Local data, logs, and experiment artifacts

## Model Requirement
This repo expects the embedding model **IBM Granite-Embedding-278M** (`granite-embedding-278m-multilingual`). The model files are **not** included.

Place the model under:
- `models/granite-embedding-278m-multilingual/`

## Track A (Python)
Run the metrics pipeline with contextual on/off inputs:

```bash
python python/track_a_ctxonoff_metrics.py \
  --run-dir runs/track_a_demo \
  --ctx-on examples/ctx_on.txt \
  --ctx-off examples/ctx_off.txt \
  --anchors examples/anchors.txt \
  --model-dir models/granite-embedding-278m-multilingual
```

Notes:
- **ICS behavior:** When anchors are provided, `ICS` is computed as the **weighted anchor-centroid cosine**. The pairwise ICS is preserved in `security.metric_notes.ICS_pairwise`. If no anchors are supplied, `ICS` falls back to pairwise.
- SecureLogger is optional and **not provided**. To enable private logging, set `CAIROS_SECURE_LOGGER=1` and provide your SecureLogger module on `PYTHONPATH` plus key env vars `SECURE_LOGGER_KEY` or `SECURE_LOGGER_KEY_FILE`.
- REPA alignment: no metric values are fabricated. Missing dependencies yield `null` outputs with explicit notes in `security.metric_notes`.

## Track B (C++ / ONNX)
Build the native tools:

```bash
cmake -S native -B build \
  -DONNXRUNTIME_DIR=PATH_TO_ONNXRT \
  -DSENTENCEPIECE_DIR=PATH_TO_SPM
cmake --build build --config Release
```

Executables:
- `granite_embedder_cli`
- `granite_hidden_state_cli`
- `identity_anchor_cli`
- `recp_metrics_cli`

## Documentation
- `docs/SDD_IEEE.md` — IEEE-style Software Design Description
- `docs/DEPENDENCIES.md` — Python/C++ dependency versions
- `docs/SECURELOGGER_NOTICE.md` — SecureLogger scope statement

## Repository Hygiene
- No local absolute paths.
- No private datasets.
- No model weights.
\n## Coords Converter (C++ Primary)\nGenerate PCA coords (2D/3D) from RECP JSONL embeddings.\n\n`ash\ncoords_from_embeddings_cli --input-root <dir_with_jsonl> --out-dir <out> --dims 3 --anchors <anchors_jsonl>\n`\n

## Tokenization (Model-Consistent)
Generate token counts using the Granite tokenizer (SentencePiece).

```bash
granite_tokenizer_cli --model-dir <model_dir> --input <txt> --output <jsonl>
```

Python parity tool (optional):

```bash
python python/granite_tokenizer.py --model-dir <model_dir> --input <txt> --output <jsonl>
```

Use `--token-counts-ctxon`/`--token-counts-ctxoff` in Track A to compute TCDM from model-consistent token counts.


## TCDM Correction (Token Counts Required)
TCDM is defined as **token-to-word variance using model-consistent token counts**.
Legacy character/word variance is deprecated and should not be used for REPA evidence.

Use the tokenizer output with:
- Track A: `--token-counts-ctxon` / `--token-counts-ctxoff`
- Track B: `recp_metrics_cli --token-counts <jsonl>`

## SecureLogger Requirement & Opt-Out (Not Recommended)
**Current behavior**
- **Track B (`recp_metrics_cli`) is fail-closed**: it requires `--secure-log-dir` and `--secure-key`. Without SecureLogger keys, the CLI exits.
- **Track A (Python)** runs without SecureLogger by default unless `CAIROS_SECURE_LOGGER=1` is set.

**Why this matters**
SecureLogger v2.0 is **private** and not distributed. Any runs performed **without** SecureLogger keys/signatures are considered **external/unverified** and **will not be attributed to ChasingBlu R&D Labs**.

**Opt-out instructions (Track B C++ only)**
If you still need to run without SecureLogger, you must **modify the source** and rebuild:
1. Edit `native/src/recp_metrics_cli.cpp`:
   - Remove the fail-closed key check and SecureLogger initialization block.
   - Remove `--secure-log-dir` / `--secure-key` from the CLI usage and argument parsing.
   - Remove `sl_log_event(...)`, `sl_verify(...)`, and `sl_free(...)` calls.
2. Rebuild the native target:
   - `cmake --build build --config Release`

**Opt-out (Track A Python)**
- Do **not** set `CAIROS_SECURE_LOGGER=1`.
- The pipeline will run but outputs are **non-REPA** without SecureLogger evidence.
