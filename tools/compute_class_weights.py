#!/usr/bin/env python3
"""Compute STSD class counts and class-balanced weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gssnet.utils.config import load_config
from gssnet.utils.losses.class_weight import (
    effective_number_weights,
    median_frequency_weights,
)
from gssnet.utils.run_settings import resolve_config_path


def _default_output_dir(cfg: dict, project_root: Path) -> Path:
    loss_cfg = ((cfg.get("model") or {}).get("loss") or {})
    ce_cfg = loss_cfg.get("ce") or {}
    cw_cfg = ce_cfg.get("class_weight") or {}
    weight_file = cw_cfg.get("weight_file")
    if weight_file:
        p = Path(weight_file)
        if not p.is_absolute():
            p = project_root / p
        return p.parent
    return project_root / "outputs" / "stsd_train"


def count_one_split_dir(
    split_root: Path,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    if not split_root.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_root}")
    scenes = sorted(p for p in split_root.iterdir() if p.is_dir())
    if not scenes:
        raise FileNotFoundError(f"no scene directories in {split_root}")
    for scene in scenes:
        seg_path = scene / "segment.npy"
        if not seg_path.is_file():
            continue
        seg = np.load(seg_path).reshape(-1).astype(np.int64)
        valid = seg != int(ignore_index)
        seg = seg[valid]
        seg = seg[(seg >= 0) & (seg < num_classes)]
        counts += np.bincount(seg, minlength=num_classes)[:num_classes]
    return counts


def count_split(
    data_root: Path,
    split: str | list[str],
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    splits = [split] if isinstance(split, str) else list(split)
    counts = np.zeros(num_classes, dtype=np.int64)
    for sp in splits:
        counts += count_one_split_dir(
            data_root / sp,
            num_classes,
            ignore_index,
        )
    return counts


def main():
    parser = argparse.ArgumentParser(description="Compute class weights for STSD")
    parser.add_argument("--config", default="configs/stsd/gssnet.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--ignore-index", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_config_path(args.config))
    project_root = Path(cfg.get("_project_root", ".")).resolve()
    model_cfg = cfg.get("model") or {}
    loss_cfg = model_cfg.get("loss") or {}
    ce_cfg = loss_cfg.get("ce") or {}
    cw_cfg = ce_cfg.get("class_weight") or {}

    split = args.split or (cfg.get("train") or {}).get("split", "Area_1")
    splits = [split] if isinstance(split, str) else list(split)
    num_classes = int(args.num_classes or model_cfg.get("num_classes"))
    ignore_index = int(args.ignore_index if args.ignore_index is not None else model_cfg.get("ignore_index", -1))
    beta = float(args.beta if args.beta is not None else cw_cfg.get("beta", 0.999))
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(cfg, project_root)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = count_split(
        Path(cfg["data_root"]),
        splits,
        num_classes,
        ignore_index,
    )
    eff = effective_number_weights(
        counts,
        beta=beta,
        normalize=str(cw_cfg.get("normalize", "mean")),
        min_weight=float(cw_cfg.get("min_weight", 0.2)),
        max_weight=float(cw_cfg.get("max_weight", 5.0)),
    )
    med = median_frequency_weights(
        counts,
        normalize=str(cw_cfg.get("normalize", "mean")),
        min_weight=float(cw_cfg.get("min_weight", 0.2)),
        max_weight=float(cw_cfg.get("max_weight", 5.0)),
    )
    total = max(int(counts.sum()), 1)
    freq = counts.astype(np.float64) / float(total)

    with (out_dir / "class_counts.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": splits[0] if len(splits) == 1 else None,
                "splits": splits,
                "ignore_index": ignore_index,
                "num_classes": num_classes,
                "counts": counts.tolist(),
                "frequency": freq.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    np.save(out_dir / "class_weights_effective_number.npy", eff)
    med_path = out_dir / "class_weights_median_frequency.npy"
    weight_file = cw_cfg.get("weight_file")
    if weight_file:
        wf = Path(weight_file)
        if not wf.is_absolute():
            wf = project_root / wf
        med_path = wf
    med_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(med_path, med)

    eff_range = float(eff.max() - eff.min()) if eff.size else 0.0
    if eff_range < 1e-4:
        print(
            "WARNING: effective_number weights are nearly uniform (all ~1.0 after mean normalize). "
            "This happens when every class has a large point count and beta is close to 1. "
            "For STSD, prefer weight_file: outputs/stsd_train/class_weights_median_frequency.npy in your yaml."
        )

    print(f"saved class weights to {out_dir}")
    for c in range(num_classes):
        print(
            f"class {c:02d}: count={int(counts[c])} freq={freq[c]:.8f} "
            f"effective={float(eff[c]):.6f} median_freq={float(med[c]):.6f}"
        )


if __name__ == "__main__":
    main()
