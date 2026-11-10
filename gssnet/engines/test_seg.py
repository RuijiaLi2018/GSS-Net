#!/usr/bin/env python3
"""Test / evaluate STSD semantic segmentation."""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path

from gssnet.engines.launch import launch
from gssnet.engines.semseg_tester import SemSegTester
from gssnet.utils import comm
from gssnet.utils.config import load_config
from gssnet.utils.logger import setup_logger
from gssnet.utils.run_settings import resolve_config_path, resolve_project_path


@dataclass
class TestSegArgs:
    config_file: str = "configs/stsd/gssnet.yaml"
    save_path: str | None = None
    weight_name: str = "best_miou.pth"
    split: str = "test"


def _resolve_weight(args: TestSegArgs, cfg: dict) -> str:
    w = args.weight_name
    if w and os.path.isfile(w):
        return w
    save = args.save_path or cfg.get("save_path")
    cand = Path(save) / w
    if cand.is_file():
        return str(cand)
    ec = cfg.get("eval_checkpoint")
    if ec and os.path.isfile(ec):
        return ec
    raise FileNotFoundError(f"Checkpoint not found: {w} or {cand}")


def test_worker(cfg: dict, weight_path: str, eval_split_key: str):
    from gssnet.utils.train_report import attach_file_logger

    logger = setup_logger()
    if eval_split_key == "val":
        test_dir = cfg.get("eval_valid_save_path") or str(Path(cfg["save_path"]) / "eval_valid")
    else:
        test_dir = cfg.get("test_save_path") or cfg["save_path"]
    if comm.is_main_process():
        attach_file_logger(str(test_dir), filename=f"test_{eval_split_key}.log")
    tester = SemSegTester(cfg, weight_path, logger=logger, eval_split_key=eval_split_key)
    tester.test()


def main():
    args = TestSegArgs()
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation")
    parser.add_argument("--config", default=args.config_file, help="Path to YAML config")
    parser.add_argument("--save-path", default=args.save_path)
    parser.add_argument("--weight-name", default=args.weight_name)
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default=args.split,
        help="Evaluate test or valid split (both use test_mode inference)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override GPU count for evaluation (default: num_gpus_test or num_gpus in yaml)",
    )
    ns = parser.parse_args()
    args.config_file = ns.config
    args.save_path = ns.save_path
    args.weight_name = ns.weight_name
    args.split = ns.split

    config_path = resolve_config_path(args.config_file)
    cfg_probe = load_config(config_path)
    if ns.num_gpus is not None:
        num_gpus = int(ns.num_gpus)
    else:
        num_gpus = int(cfg_probe.get("num_gpus_test", cfg_probe.get("num_gpus", 1)))

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        if num_gpus == 1:
            cvd = cfg_probe.get("cuda_visible_devices_test") or "0"
        else:
            cvd = cfg_probe.get("cuda_visible_devices")
        if cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    cfg = load_config(config_path)
    if args.save_path:
        cfg["save_path"] = resolve_project_path(args.save_path, cfg["_project_root"])
    weight = _resolve_weight(args, cfg)
    launch(test_worker, num_gpus_per_machine=num_gpus, cfg=(cfg, weight, args.split))


if __name__ == "__main__":
    main()
