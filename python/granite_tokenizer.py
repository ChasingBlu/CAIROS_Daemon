#!/usr/bin/env python3
"""Granite tokenizer (model-consistent) using SentencePiece.
Outputs JSONL with token_count per line for TCDM.
"""
import argparse
import json
from pathlib import Path

try:
    import sentencepiece as spm
except Exception as exc:
    raise SystemExit("sentencepiece is required for this tokenizer. Install it and retry.") from exc


def word_count(text: str) -> int:
    return len(text.split())


def map_sp_id(sp_id: int, sp) -> int:
    bos = sp.bos_id()
    eos = sp.eos_id()
    unk = sp.unk_id()
    if sp_id == bos and bos >= 0:
        return 0
    if sp_id == eos and eos >= 0:
        return 2
    if sp_id == unk and unk >= 0:
        return 3
    if sp_id < 0:
        return sp_id
    return sp_id + 1


def tokenize_line(sp, text: str, add_bos_eos: bool) -> list[int]:
    ids = sp.encode(text, out_type=int)
    out = []
    if add_bos_eos and sp.bos_id() >= 0:
        out.append(map_sp_id(sp.bos_id(), sp))
    for i in ids:
        out.append(map_sp_id(i, sp))
    if add_bos_eos and sp.eos_id() >= 0:
        out.append(map_sp_id(sp.eos_id(), sp))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Granite tokenizer (SentencePiece)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-bos-eos", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    spm_path = model_dir / "sentencepiece.bpe.model"
    if not spm_path.exists():
        raise SystemExit(f"sentencepiece.bpe.model not found in {model_dir}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))

    add_bos_eos = not args.no_bos_eos
    input_path = Path(args.input)
    output_path = Path(args.output)

    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    with output_path.open("w", encoding="utf-8") as f:
        for idx, line in enumerate(lines):
            if not line:
                continue
            token_ids = tokenize_line(sp, line, add_bos_eos)
            payload = {
                "idx": idx,
                "token_count": len(token_ids),
                "word_count": word_count(line),
            }
            f.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
