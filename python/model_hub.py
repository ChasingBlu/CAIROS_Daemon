#!/usr/bin/env python3
"""Central model orchestration utilities for CAIROS Daemon Mirror.

This module keeps every embedding/encoder model offline, registered, and
observable.  Each adapter verifies its own assets, exposes diagnostics, and
streams embeddings through a common export + SecureExperimentLogger pathway.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


class SecureExperimentLogger:
    """Minimal tamper-evident logger (ready for SecureExperimentLogger v3 swap)."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_file(self, model_name: str, artifact: Path, metadata: Dict[str, str]) -> Path:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": model_name,
            "artifact": str(artifact.resolve()),
            "sha256": digest,
            "metadata": metadata,
        }
        log_path = self.log_dir / f"secure_log_{datetime.utcnow().date():%Y%m%d}.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry))
            handle.write("\n")
        return log_path


class EmbeddingModelAdapter(ABC):
    def __init__(self, model_name: str, model_dir: Path) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.loaded = False

    @abstractmethod
    def required_files(self) -> List[str]:
        ...

    @abstractmethod
    def load(self) -> bool:
        ...

    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, object]]:
        ...

    @abstractmethod
    def diagnostics(self) -> Dict[str, object]:
        ...

    def verify_assets(self) -> None:
        missing = [rel for rel in self.required_files() if not (self.model_dir / rel).exists()]
        if missing:
            raise FileNotFoundError(f"{self.model_name}: missing files {missing}")


class ModelHub:
    def __init__(self, exports_root: Path) -> None:
        self.exports_root = exports_root
        self.embeddings_dir = exports_root / "embeddings"
        self.logs_dir = exports_root / "logs"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.logger = SecureExperimentLogger(self.logs_dir)
        self._adapters: Dict[str, EmbeddingModelAdapter] = {}

    def register(self, adapter: EmbeddingModelAdapter) -> None:
        self._adapters[adapter.model_name] = adapter

    def activate(self, model_name: str) -> EmbeddingModelAdapter:
        adapter = self._adapters[model_name]
        adapter.verify_assets()
        if not adapter.load():
            raise RuntimeError(f"{model_name}: activation aborted by operator")
        return adapter

    def run_diagnostics(self, model_name: str) -> Dict[str, object]:
        adapter = self._adapters[model_name]
        diag = adapter.diagnostics()
        diag_path = self.logs_dir / f"diagnostics_{model_name}_{datetime.utcnow():%Y%m%dT%H%M%SZ}.json"
        with diag_path.open("w", encoding="utf-8") as handle:
            json.dump(diag, handle, indent=2)
        self.logger.record_file(model_name, diag_path, {"type": "diagnostics"})
        return diag

    def export_embeddings(self, model_name: str, rows: Iterable[Dict[str, object]]) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = self.embeddings_dir / model_name / f"embeddings_{timestamp}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row))
                handle.write("\n")
        self.logger.record_file(model_name, out_path, {"type": "embedding_batch"})
        return out_path


def enforce_offline() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
