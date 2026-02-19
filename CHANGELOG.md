# Changelog

## 2026-02-17 23:45:00 +02:00
- REPA amendment: corrected TCDM to use model-consistent token counts (SentencePiece) across Track A/B.
- `recp_metrics` now accepts optional token-counts; CLI + Track A runner pass token counts when available.
- Legacy char/word variance marked deprecated (non-REPA for evidence).

All notable changes to this GitHub-ready subset are documented here, in chronological order.

## 2026-02-17\n- Added model-consistent tokenization (C++ tokenizer CLI + optional Python tokenizer) and token-count overrides for TCDM.\n- Added tokenization source field to metrics outputs.\n- Track A: `ICS` now reflects weighted anchor-centroid when anchors exist; pairwise ICS is recorded in `security.metric_notes.ICS_pairwise` for audit.
- Documentation: README and IEEE SDD updated for REPA alignment (explicit omissions, no fabricated metrics, and SecureLogger scope).

## 2026-02-16
- Initial GitHub-ready subset published with Track A (Python) and Track B (C++/ONNX) pipelines.

