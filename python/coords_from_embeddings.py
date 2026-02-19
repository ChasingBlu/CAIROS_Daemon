#!/usr/bin/env python3
"""Convert RECP JSONL embeddings into 2D/3D PCA coords.
C/C++ pipeline is primary; this Python tool is for parity and quick checks.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np


def read_jsonl_embeddings(path):
    idxs = []
    embeddings = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "embedding" not in obj:
                raise ValueError(f"Missing 'embedding' in {path} line {line_no}")
            idx = obj.get("idx", line_no - 1)
            idxs.append(idx)
            embeddings.append(obj["embedding"])
    if not embeddings:
        raise ValueError(f"No embeddings found in {path}")
    return idxs, np.asarray(embeddings, dtype=np.float64)


def compute_pca_basis(embeddings, n_components=3):
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    return mean, components


def project(embeddings, mean, components):
    return (embeddings - mean) @ components


def minmax_params(coords):
    minv = coords.min(axis=0)
    maxv = coords.max(axis=0)
    return minv, maxv


def scale_minmax(coords, minv, maxv):
    denom = maxv - minv
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return (coords - minv) / denom


def write_csv(path, idxs, coords):
    with open(path, "w", encoding="utf-8") as handle:
        dims = coords.shape[1]
        if dims == 2:
            handle.write("idx,x,y\n")
            for idx, (x, y) in zip(idxs, coords):
                handle.write(f"{idx},{x:.8f},{y:.8f}\n")
        elif dims == 3:
            handle.write("idx,x,y,z\n")
            for idx, (x, y, z) in zip(idxs, coords):
                handle.write(f"{idx},{x:.8f},{y:.8f},{z:.8f}\n")
        else:
            raise ValueError(f"Unsupported coord dimension for CSV output: {dims}")


def write_json_coords(path, coords):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(coords.tolist(), handle, indent=2)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def process_mode(mode, may_path, feb_path, out_dir, dims=3, anchor_path=None):
    may_idxs, may_embeddings = read_jsonl_embeddings(may_path)
    feb_idxs, feb_embeddings = read_jsonl_embeddings(feb_path)
    anchor_idxs = None
    anchor_embeddings = None
    if anchor_path:
        anchor_idxs, anchor_embeddings = read_jsonl_embeddings(anchor_path)

    if may_embeddings.shape[1] != feb_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch for {mode}: "
            f"May={may_embeddings.shape[1]} Feb={feb_embeddings.shape[1]}"
        )

    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")

    mean, components = compute_pca_basis(may_embeddings, n_components=dims)
    may_coords = project(may_embeddings, mean, components)
    feb_coords = project(feb_embeddings, mean, components)

    minv, maxv = minmax_params(may_coords)
    may_scaled = scale_minmax(may_coords, minv, maxv)
    feb_scaled = scale_minmax(feb_coords, minv, maxv)

    prefix = f"{mode}_pca_{dims}d"
    save_json(
        os.path.join(out_dir, f"{prefix}_meta.json"),
        {
            "mode": mode,
            "created": datetime.now(timezone.utc).isoformat(),
            "may_source": os.path.abspath(may_path),
            "feb_source": os.path.abspath(feb_path),
            "anchor_source": os.path.abspath(anchor_path) if anchor_path else None,
            "embedding_dim": int(may_embeddings.shape[1]),
            "components_shape": [int(x) for x in components.shape],
            "coords_dims": dims,
        },
    )
    save_json(
        os.path.join(out_dir, f"{prefix}_minmax.json"),
        {
            "mode": mode,
            "min": minv.tolist(),
            "max": maxv.tolist(),
        },
    )
    save_json(os.path.join(out_dir, f"{prefix}_mean.json"), mean.tolist())
    save_json(os.path.join(out_dir, f"{prefix}_components.json"), components.tolist())

    write_csv(os.path.join(out_dir, f"may_{mode}_coords_{dims}d.csv"), may_idxs, may_scaled)
    write_csv(os.path.join(out_dir, f"feb_{mode}_coords_{dims}d.csv"), feb_idxs, feb_scaled)
    write_json_coords(os.path.join(out_dir, f"may_{mode}_coords_{dims}d.json"), may_scaled)
    write_json_coords(os.path.join(out_dir, f"feb_{mode}_coords_{dims}d.json"), feb_scaled)

    if anchor_embeddings is not None:
        anchor_coords = project(anchor_embeddings, mean, components)
        anchor_scaled = scale_minmax(anchor_coords, minv, maxv)
        write_csv(os.path.join(out_dir, f"anchors_{mode}_coords_{dims}d.csv"), anchor_idxs, anchor_scaled)
        write_json_coords(os.path.join(out_dir, f"anchors_{mode}_coords_{dims}d.json"), anchor_scaled)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RECP JSONL embeddings into coords using fixed PCA basis."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing MAY/FEB_CTXON/CTXOFF_EMBEDDINGS.jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <input-root>/coords_out_3d for dims=3).",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=3,
        help="Number of PCA dims for coords (2 or 3). 3 uses PC3 as Z.",
    )
    parser.add_argument(
        "--anchors",
        default=None,
        help="Optional anchors embeddings JSONL; projected using May PCA basis.",
    )
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    default_out = "coords_out_3d" if args.dims == 3 else "coords_out"
    out_dir = os.path.abspath(args.out_dir or os.path.join(input_root, default_out))
    os.makedirs(out_dir, exist_ok=True)

    file_map = {
        "ctxon": (
            os.path.join(input_root, "MAY_CTXON_EMBEDDINGS.jsonl"),
            os.path.join(input_root, "FEB_CTXON_EMBEDDINGS.jsonl"),
        ),
        "ctxoff": (
            os.path.join(input_root, "MAY_CTXOFF_EMBEDDINGS.jsonl"),
            os.path.join(input_root, "FEB_CTXOFF_EMBEDDINGS.jsonl"),
        ),
    }

    for mode, (may_path, feb_path) in file_map.items():
        if not os.path.exists(may_path):
            raise FileNotFoundError(f"Missing {mode} May embeddings: {may_path}")
        if not os.path.exists(feb_path):
            raise FileNotFoundError(f"Missing {mode} Feb embeddings: {feb_path}")
        process_mode(mode, may_path, feb_path, out_dir, dims=args.dims, anchor_path=args.anchors)

    print(f"[OK] Wrote coords + PCA basis to: {out_dir}")


if __name__ == "__main__":
    main()
