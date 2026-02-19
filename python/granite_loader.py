#!/usr/bin/env python3
"""Offline-only loader for IBM Granite-Embedding-278M (CAIROS Daemon voice)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer

from model_hub import EmbeddingModelAdapter, ModelHub, enforce_offline

TOOLS_DIR = Path(__file__).resolve().parent
MIRROR_ROOT = TOOLS_DIR.parent.parent
DEFAULT_MODEL_PATH = MIRROR_ROOT / "models" / "granite-embedding-278m-multilingual"
DEFAULT_INPUT_DIR = MIRROR_ROOT / "operations" / "granite" / "input"
DEFAULT_EXPORT_ROOT = MIRROR_ROOT / "operations" / "granite" / "exports"
DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def speak(message: str) -> None:
    print(f"[CAIROS DAEMON] {message}")


class GraniteEmbeddingAdapter(EmbeddingModelAdapter):
    def __init__(self, model_dir: Path, interactive: bool = True) -> None:
        super().__init__("granite-embedding-278m-multilingual", model_dir)
        self.interactive = interactive
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._pooling_mode = "mean"
        self._load_pooling_config()

    def required_files(self) -> List[str]:
        return [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "sentencepiece.bpe.model",
            "modules.json",
        ]

    def _prompt_action(self, error_msg: str) -> str:
        speak("Model load failed. Choose next action.")
        speak(error_msg)
        while True:
            choice = input("[reload/verbose/abort]: ").strip().lower()
            if choice in {"reload", "verbose", "abort"}:
                return choice
            speak("Please type reload, verbose, or abort.")

    def load(self) -> bool:
        enforce_offline()
        while True:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                self._model = AutoModel.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(device)
                self._model.eval()
                self.loaded = True
                speak(f"Granite ready on {device}.")
                return True
            except Exception as exc:  # pragma: no cover
                action = self._prompt_action(str(exc))
                if action == "verbose":
                    speak(repr(exc))
                if action == "abort":
                    return False

    def _load_pooling_config(self) -> None:
        config_path = self.model_dir / "1_Pooling" / "config.json"
        if not config_path.exists():
            return
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if data.get("pooling_mode_cls_token"):
            self._pooling_mode = "cls"
        elif data.get("pooling_mode_mean_tokens"):
            self._pooling_mode = "mean"

    def _l2_normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.norm(vectors, dim=1, keepdim=True).clamp(min=1e-12)
        return vectors / norms

    def _pool_hidden_states(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self._pooling_mode == "cls":
            return hidden[:, 0, :]
        mask = mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return summed / denom

    def encode(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, object]]:
        if not self.loaded or self._tokenizer is None or self._model is None:
            raise RuntimeError("Model not loaded")
        device = next(self._model.parameters()).device
        rows: List[Dict[str, object]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._tokenizer.model_max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
                pooled = self._pool_hidden_states(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = self._l2_normalize(pooled).cpu()
            for idx, emb in enumerate(pooled):
                rows.append(
                    {
                        "idx": start + idx,
                        "text": batch[idx],
                        "embedding": emb.tolist(),
                    }
                )
        return rows

    def diagnostics(self) -> Dict[str, object]:
        stats = {}
        for rel in self.required_files():
            path = self.model_dir / rel
            stats[rel] = {
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        stats["interactive"] = self.interactive
        stats["loaded"] = self.loaded
        return stats


def _prompt_for_path(question: str) -> Path:
    while True:
        resp = input(question).strip()
        if not resp:
            speak("Input cannot be empty.")
            continue
        if resp.lower() in {"q", "quit"}:
            speak("Operator aborted operation.")
            raise SystemExit(1)
        candidate = Path(resp).expanduser()
        if candidate.exists():
            return candidate
        speak("Path not found. Try again or type 'q' to abort.")


def _select_model_dir(cli_path: Path | None) -> Path:
    candidates = [cli_path, DEFAULT_MODEL_PATH]
    for cand in candidates:
        if cand and cand.is_dir():
            speak(f"Using model directory: {cand}")
            return cand
    speak("Model directory missing. Please enter a valid path.")
    return _prompt_for_path("Model directory: ")


def _list_input_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.txt"))


def _select_input_file(cli_path: Path | None) -> Path:
    if cli_path and cli_path.exists():
        speak(f"Using input file: {cli_path}")
        return cli_path
    files = _list_input_files(DEFAULT_INPUT_DIR)
    if files:
        speak("Available input files:")
        for idx, file in enumerate(files, start=1):
            print(f"    [{idx}] {file.name}")
        while True:
            choice = input("Select file number or enter custom path: ").strip()
            if choice.isdigit():
                index = int(choice)
                if 1 <= index <= len(files):
                    selection = files[index - 1]
                    speak(f"Selected input file: {selection}")
                    return selection
            elif choice:
                candidate = Path(choice).expanduser()
                if candidate.exists():
                    speak(f"Using input file: {candidate}")
                    return candidate
            speak("Invalid selection. Try again.")
    else:
        speak(f"No .txt files found in {DEFAULT_INPUT_DIR}.")
    selection = _prompt_for_path("Enter path to a .txt file: ")
    if not selection.exists():
        speak("Input file not found, aborting.")
        raise SystemExit(1)
    speak(f"Using input file: {selection}")
    return selection


def _select_export_root(cli_path: Path | None) -> Path:
    path = cli_path or DEFAULT_EXPORT_ROOT
    path.mkdir(parents=True, exist_ok=True)
    speak(f"Embeddings will be written under: {path}")
    return path


def _read_texts(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Granite embedding runner")
    parser.add_argument("--model-path", type=Path, help="Override model directory", default=None)
    parser.add_argument("--input", type=Path, help="Override input file path", default=None)
    parser.add_argument("--exports", type=Path, help="Override export root", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--non-interactive", action="store_true", help="Disable reload prompts")
    parser.add_argument("--queue", action="store_true", help="Emit READY/DONE markers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = _select_model_dir(args.model_path)
    input_file = _select_input_file(args.input)
    exports_root = _select_export_root(args.exports)
    texts = _read_texts(input_file)
    if not texts:
        speak("Input file contained no usable lines. Aborting.")
        return
    hub = ModelHub(exports_root)
    adapter = GraniteEmbeddingAdapter(model_dir, interactive=not args.non_interactive)
    hub.register(adapter)
    if args.queue:
        print("[GRANITE] LOADING", file=sys.stderr)
    active = hub.activate(adapter.model_name)
    hub.run_diagnostics(adapter.model_name)
    if args.queue:
        print("[GRANITE] READY", file=sys.stderr)
    rows = active.encode(texts, batch_size=args.batch_size)
    export_path = hub.export_embeddings(adapter.model_name, rows)
    if args.queue:
        print("[GRANITE] DONE", file=sys.stderr)
    speak(f"Embeddings written to {export_path}")


if __name__ == "__main__":
    main()
