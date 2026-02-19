#!/usr/bin/env python3
"""
Track A (Python) metrics runner.
Outputs legacy RECP metrics JSON + optional SecureLogger metadata.

Notes:
- SecureLogger v2.0 is private and not included in this repo.
- When SecureLogger is unavailable, the pipeline still computes metrics and
  marks security fields as unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRANITE_LOADER = Path(__file__).resolve().parent / "granite_loader.py"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "granite-embedding-278m-multilingual"


def now_local() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_iso() -> str:
    return datetime.now().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_conversation(path: Path, max_turns: Optional[int] = None) -> List[Tuple[str, str, str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    turns: List[Tuple[str, str, str]] = []
    speaker = None
    buf: List[str] = []
    tid = 0
    for line in lines:
        tag = line.strip().lower()
        if tag in ("user", "chatgpt"):
            if speaker and buf:
                tid += 1
                turns.append((f"turn_{tid:03d}", speaker, "\n".join(buf).strip()))
                if max_turns and len(turns) >= max_turns:
                    break
                buf = []
            speaker = tag
        else:
            buf.append(line)
    if speaker and buf and (not max_turns or len(turns) < max_turns):
        tid += 1
        turns.append((f"turn_{tid:03d}", speaker, "\n".join(buf).strip()))
    return turns


def prepare_turns(convo_file: Path, out_dir: Path, label: str, max_turns: Optional[int]) -> Tuple[Path, Path, List[str]]:
    turns = parse_conversation(convo_file, max_turns=max_turns)
    turns_file = out_dir / f"turns_{label}.txt"
    meta_file = out_dir / f"turns_{label}_meta.json"
    with turns_file.open("w", encoding="utf-8") as f:
        for tid, speaker, text in turns:
            line = f"{tid}|{speaker}|" + collapse_whitespace(text)
            f.write(line + "\n")
    meta = {
        "source": str(convo_file),
        "count": len(turns),
        "turns": [{"turn_id": tid, "speaker": speaker, "text": text} for tid, speaker, text in turns],
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    texts = [t[2] for t in turns]
    return turns_file, meta_file, texts


def run_granite(input_file: Path, export_root: Path, logs_dir: Path, model_dir: Path, granite_loader: Path) -> Path:
    export_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"granite_{input_file.stem}.log"
    cmd = [
        sys.executable,
        str(granite_loader),
        "--model-path",
        str(model_dir),
        "--input",
        str(input_file),
        "--exports",
        str(export_root),
        "--non-interactive",
        "--queue",
    ]
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] CMD: {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"\n[{now_iso()}] Exit code: {proc.returncode}\n")
    embed_dir = export_root / "embeddings" / "granite-embedding-278m-multilingual"
    jsonls = sorted(embed_dir.glob("embeddings_*.jsonl"))
    if not jsonls:
        raise RuntimeError(f"No embeddings jsonl found in {embed_dir}")
    return jsonls[-1]




def load_token_counts(path: Optional[Path]) -> Optional[List[int]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Token counts file not found: {path}")
    counts: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "token_count" not in obj:
                raise ValueError(f"Missing token_count in {path}")
            counts.append(int(obj["token_count"]))
    return counts


def token_to_word_variance_counts(turns: List[str], token_counts: List[int]) -> float:
    ratios = []
    for line, tcount in zip(turns, token_counts):
        words = line.split()
        if len(words) > 0:
            ratios.append(float(tcount) / float(len(words)))
    return float(np.var(ratios)) if ratios else 0.0


def t_test_ratio_variance_counts(turns: List[str], token_counts: List[int]) -> Tuple[Optional[float], Optional[float]]:
    ratios = []
    for line, tcount in zip(turns, token_counts):
        words = line.split()
        if len(words) > 0:
            ratios.append(float(tcount) / float(len(words)))
    if len(ratios) < 4:
        return None, None
    mid = len(ratios) // 2
    a = np.asarray(ratios[:mid], dtype=float)
    b = np.asarray(ratios[mid:], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None, None
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    denom = math.sqrt(var_a / len(a) + var_b / len(b))
    if denom == 0:
        return 0.0, None
    t_stat = (mean_a - mean_b) / denom
    try:
        from scipy import stats  # type: ignore
        p_val = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        p_val = None
    return float(t_stat), p_val


def load_embeddings(jsonl_path: Path) -> List[Dict]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def compute_weighted_anchor_centroid(anchor_embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """May-style weighted centroid: weight by semantic uniqueness."""
    if not anchor_embeddings:
        return None
    if len(anchor_embeddings) == 1:
        return anchor_embeddings[0]
    weights = []
    for i, emb in enumerate(anchor_embeddings):
        similarities = [
            cosine(emb, other)
            for j, other in enumerate(anchor_embeddings)
            if i != j
        ]
        uniqueness = 1.0 - max(similarities) if similarities else 1.0
        weights.append(uniqueness)
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum < 1e-12:
        weights = np.ones(len(anchor_embeddings), dtype=np.float64) / float(len(anchor_embeddings))
    else:
        weights = weights / weight_sum
    return np.average(np.vstack(anchor_embeddings), axis=0, weights=weights)


def anchor_persistence_index(turns: List[str], anchors: List[str]) -> float:
    k = len(anchors)
    n = len(turns)
    if k == 0 or n == 0:
        return 0.0
    count = 0
    for a in anchors:
        for m in turns:
            if a.lower() in m.lower():
                count += 1
    return count / (k * n)


def entropy(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = float(sum(counts.values()))
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p + 1e-9) for p in probs)


def entropy_collapse_rate(turns: List[str]) -> float:
    if len(turns) < 2:
        return 0.0
    h1 = entropy(turns[0])
    hn = entropy(turns[-1])
    if h1 == 0:
        return 0.0
    return 1.0 - (hn / h1)


def identity_consistency_pairwise(embeddings: list[np.ndarray]) -> float:
    n = len(embeddings)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine(embeddings[i], embeddings[j])
            count += 1
    return total / count if count else 0.0


def loop_divergence_index(embeddings: List[np.ndarray]) -> float:
    n = len(embeddings)
    if n < 2:
        return 0.0
    total_div = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_div += (1.0 - cosine(embeddings[i], embeddings[j]))
            count += 1
    return (2.0 / (n * (n - 1))) * total_div


def signal_recursion_variance(embeddings: List[np.ndarray], k: int = 1) -> float:
    if len(embeddings) <= k:
        return 0.0
    drift_values = []
    for i in range(k, len(embeddings)):
        drift_values.append(1.0 - cosine(embeddings[i], embeddings[i - k]))
    return float(np.var(drift_values))


def token_to_word_variance(turns: List[str]) -> float:
    ratios = []
    for line in turns:
        words = line.split()
        tokens = list(line)
        if len(words) > 0:
            ratios.append(len(tokens) / len(words))
    return float(np.var(ratios)) if ratios else 0.0


def t_test_ratio_variance(turns: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """Welch t-test between first and second half token/word ratios."""
    ratios = []
    for line in turns:
        words = line.split()
        tokens = list(line)
        if len(words) > 0:
            ratios.append(len(tokens) / len(words))
    if len(ratios) < 4:
        return None, None
    mid = len(ratios) // 2
    a = np.asarray(ratios[:mid], dtype=float)
    b = np.asarray(ratios[mid:], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None, None
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    denom = math.sqrt(var_a / len(a) + var_b / len(b))
    if denom == 0:
        return 0.0, None
    t_stat = (mean_a - mean_b) / denom
    try:
        from scipy import stats  # type: ignore
        p_val = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        p_val = None
    return float(t_stat), p_val


def load_anchors(path: Optional[Path]) -> List[str]:
    if not path:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Anchors file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        anchors = data.get("anchors", []) if isinstance(data, dict) else []
        return [str(a) for a in anchors]
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def load_master_key() -> bytes:
    key_file = os.environ.get("SECURE_LOGGER_KEY_FILE")
    key_hex = os.environ.get("SECURE_LOGGER_KEY")
    if key_file:
        return Path(key_file).read_bytes().strip()
    if key_hex:
        return bytes.fromhex(key_hex.strip())
    raise RuntimeError("SECURE_LOGGER_KEY or SECURE_LOGGER_KEY_FILE must be set")


def init_secure_logger(log_dir: Path, system_id: str) -> Tuple[object | None, str]:
    if os.environ.get("CAIROS_SECURE_LOGGER", "0") not in {"1", "true", "TRUE"}:
        return None, "disabled"
    try:
        from secure_exp_logger import SecureExperimentLogger  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, f"unavailable: {exc}"
    key = load_master_key()
    return SecureExperimentLogger(log_dir=log_dir, master_key=key, system_id=system_id), "enabled"


def load_logger_state(log_dir: Path) -> Optional[Dict]:
    state_file = log_dir / "logger_state.json"
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    data["state_file"] = str(state_file)
    try:
        data["state_file_sha256"] = hashlib.sha256(state_file.read_bytes()).hexdigest()
    except Exception:
        pass
    return data


def compute_metrics(
    turn_rows: List[Dict],
    turn_texts: List[str],
    anchors: List[str],
    anchor_embeddings: Optional[List[np.ndarray]],
) -> Tuple[Dict, Dict]:
    embeddings = [np.asarray(row["embedding"], dtype=np.float64) for row in turn_rows]
    metrics_notes: Dict[str, str] = {}

    ics_pairwise = identity_consistency_pairwise(embeddings)
    centroid = compute_weighted_anchor_centroid(anchor_embeddings or [])
    if centroid is None:
        ics = float(ics_pairwise)
        metrics_notes["ICS"] = "Pairwise ICS (legacy). Anchor centroid unavailable (no anchors)."
    else:
        ics_anchor_vals = [cosine(emb, centroid) for emb in embeddings] if embeddings else [0.0]
        ics_anchor = float(np.nanmean(ics_anchor_vals)) if ics_anchor_vals else 0.0
        ics = ics_anchor
        metrics_notes["ICS_pairwise"] = f"{ics_pairwise:.6f}"
        metrics_notes["ICS_anchor_centroid"] = f"{ics_anchor:.6f}"
    api = anchor_persistence_index(turn_texts, anchors)
    ecr = entropy_collapse_rate(turn_texts)
    ldi = loop_divergence_index(embeddings)
    srv = signal_recursion_variance(embeddings)
    if token_counts is not None:
        if len(token_counts) != len(turn_texts):
            raise ValueError("Token counts length does not match turns length")
        tcdm_var = token_to_word_variance_counts(turn_texts, token_counts)
        tcdm_t, tcdm_p = t_test_ratio_variance_counts(turn_texts, token_counts)
        metrics_notes["TCDM_source"] = "token_counts_jsonl"
    else:
        tcdm_var = token_to_word_variance(turn_texts)
        tcdm_t, tcdm_p = t_test_ratio_variance(turn_texts)
        metrics_notes["TCDM_source"] = "legacy_char_word"
        metrics_notes["TCDM_note"] = "Legacy char/word variance (non-REPA for evidence). Provide token_counts_jsonl for canonical TCDM."

    sda = None
    hmv = None
    metrics_notes["SDA"] = "Unavailable: requires logits/token probabilities."
    metrics_notes["HMV"] = "Unavailable: requires hook embeddings + token weights."
    if tcdm_p is None:
        metrics_notes["TCDM_p"] = "Unavailable: scipy not installed (Welch t-test p-value)."

    payload = {
        "timestamp": now_local(),
        "ICS": round(float(ics), 6),
        "API": round(float(api), 6),
        "ECR": round(float(ecr), 6),
        "LDI": round(float(ldi), 6),
        "SDA": sda,
        "SRV": round(float(srv), 6),
        "HMV": hmv,
        "TCDM_var": round(float(tcdm_var), 6),
        "TCDM_t": None if tcdm_t is None else round(float(tcdm_t), 6),
        "TCDM_p": None if tcdm_p is None else round(float(tcdm_p), 6),
    }
    return payload, metrics_notes


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ctx-on", required=True, type=Path)
    parser.add_argument("--ctx-off", required=True, type=Path)
    parser.add_argument("--anchors", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--granite-loader", type=Path, default=DEFAULT_GRANITE_LOADER)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--token-counts-ctxon", type=Path, default=None)
    parser.add_argument("--token-counts-ctxoff", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    logs = run_dir / "logs"
    artifacts = run_dir / "artifacts"
    for d in [inputs, outputs, logs, artifacts]:
        d.mkdir(parents=True, exist_ok=True)

    logger, logger_state = init_secure_logger(logs / "secure_logger", system_id="track_a_ctxonoff")

    start = now_iso()
    if logger:
        logger.log_event(
            "track_a_start",
            {
                "run_id": run_dir.name,
                "timestamp_start": start,
                "ctx_on_file": str(args.ctx_on),
                "ctx_off_file": str(args.ctx_off),
                "granite_loader": str(args.granite_loader),
                "model_dir": str(args.model_dir),
            },
        )

    anchors = load_anchors(args.anchors)

    ctxon_file, ctxon_meta, ctxon_texts = prepare_turns(args.ctx_on, inputs, "ctxon", args.max_turns)
    ctxoff_file, ctxoff_meta, ctxoff_texts = prepare_turns(args.ctx_off, inputs, "ctxoff", args.max_turns)

    token_counts_ctxon = load_token_counts(args.token_counts_ctxon)
    token_counts_ctxoff = load_token_counts(args.token_counts_ctxoff)

    anchors_file = inputs / "anchors.txt"
    with anchors_file.open("w", encoding="utf-8") as f:
        for a in anchors:
            f.write(collapse_whitespace(a) + "\n")

    anchors_jsonl = run_granite(anchors_file, outputs / "anchors", logs, args.model_dir, args.granite_loader)
    ctxon_jsonl = run_granite(ctxon_file, outputs / "ctxon", logs, args.model_dir, args.granite_loader)
    ctxoff_jsonl = run_granite(ctxoff_file, outputs / "ctxoff", logs, args.model_dir, args.granite_loader)

    ctxon_rows = load_embeddings(ctxon_jsonl)
    ctxoff_rows = load_embeddings(ctxoff_jsonl)

    anchor_rows = load_embeddings(anchors_jsonl)
    anchor_embs = [np.asarray(row["embedding"], dtype=np.float64) for row in anchor_rows]

    ctxon_metrics, ctxon_notes = compute_metrics(ctxon_rows, ctxon_texts, anchors, anchor_embs, token_counts_ctxon)
    ctxoff_metrics, ctxoff_notes = compute_metrics(ctxoff_rows, ctxoff_texts, anchors, anchor_embs, token_counts_ctxoff)

    combined = {
        "contextual": ctxon_metrics,
        "non_contextual": ctxoff_metrics,
        "timestamp": now_local(),
    }

    security = {
        "logger_version": "SecureExperimentLogger v2" if logger else "unavailable",
        "logger_state": logger_state,
        "logger_status": "enabled" if logger else logger_state,
        "sha256_inputs": {
            "ctxon_source": sha256_file(args.ctx_on),
            "ctxoff_source": sha256_file(args.ctx_off),
            "ctxon_turns_file": sha256_file(ctxon_file),
            "ctxoff_turns_file": sha256_file(ctxoff_file),
            "anchors_file": sha256_file(anchors_file),
        },
        "sha256_outputs": {
            "anchors_embeddings_jsonl": sha256_file(anchors_jsonl),
            "ctxon_embeddings_jsonl": sha256_file(ctxon_jsonl),
            "ctxoff_embeddings_jsonl": sha256_file(ctxoff_jsonl),
        },
        "metric_notes": {
            "contextual": ctxon_notes,
            "non_contextual": ctxoff_notes,
        },
        "tamper_detected": False,
    }

    combined["security"] = security

    output_file = outputs / f"RECP-EXP-{run_dir.name}_metrics.json"
    output_file.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    if logger:
        logger.log_event(
            "track_a_metrics_written",
            {
                "run_id": run_dir.name,
                "metrics_file": str(output_file),
                "metrics_sha256": sha256_file(output_file),
                "timestamp_end": now_iso(),
            },
        )

    summary = {
        "run_id": run_dir.name,
        "metrics_file": str(output_file),
        "metrics_sha256": sha256_file(output_file),
        "ctxon_embeddings_jsonl": str(ctxon_jsonl),
        "ctxoff_embeddings_jsonl": str(ctxoff_jsonl),
        "logger_status": "enabled" if logger else "unavailable",
    }
    (artifacts / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
