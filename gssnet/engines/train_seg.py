#!/usr/bin/env python3
"""STSD semantic segmentation training (GSS-Net)."""

from __future__ import annotations

import logging
import os
import argparse
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gssnet.datasets.collate import stsd_collate_fn
from gssnet.datasets.stsd_dataset import STSDDataset
from gssnet.engines.launch import launch
from gssnet.models.segmentor import GSSNet
from gssnet.utils import comm
from gssnet.utils.checkpoint import load_checkpoint, save_checkpoint
from gssnet.utils.config import load_config
from gssnet.utils.eval_utils import apply_inverse_mapping, intersection_and_union, summarize_iou
from gssnet.utils.logger import setup_logger
from gssnet.utils.optimizer import build_optimizer, resolve_onecycle_max_lr
from gssnet.utils.paths import resolve_resume
from gssnet.utils.run_settings import resolve_config_path
from gssnet.utils.train_report import (
    attach_file_logger,
    log_config_summary,
    log_dataset_summary,
    log_epoch_seg,
    log_model_summary,
)
from gssnet.utils.training_history import append_epoch_record, plot_training_curves

CONFIG_FILE = "configs/stsd/gssnet.yaml"


def _check_sample_feature_dim(sample: dict, expected: int, where: str) -> None:
    feat = sample.get("feat")
    actual = None if feat is None or getattr(feat, "ndim", 0) < 2 else int(feat.shape[1])
    if actual != int(expected):
        name = sample.get("name") or sample.get("scene") or where
        raise RuntimeError(
            f"feature dim mismatch at {where}: expected in_channels={int(expected)}, "
            f"actual feat dim={actual}, file/batch name={name}"
        )


def _attach_auto_class_counts(cfg: dict, train_ds, num_classes: int, ignore_index: int, logger) -> None:
    loss_cfg = (cfg.get("model") or {}).get("loss") or {}
    ce_cfg = loss_cfg.get("ce") or {}
    cw_cfg = ce_cfg.get("class_weight") or {}
    if str(cw_cfg.get("source", "")).lower() != "auto" or cw_cfg.get("class_counts") is not None:
        return
    counts = np.zeros(int(num_classes), dtype=np.int64)
    for scene_dir in getattr(train_ds, "data_paths", []):
        seg_path = Path(scene_dir) / "segment.npy"
        if not seg_path.is_file():
            continue
        seg = np.load(seg_path).reshape(-1).astype(np.int64)
        valid = seg != int(ignore_index)
        seg = seg[valid]
        seg = seg[(seg >= 0) & (seg < int(num_classes))]
        counts += np.bincount(seg, minlength=int(num_classes))[: int(num_classes)]
    if counts.sum() <= 0:
        raise RuntimeError("class_weight.source=auto could not find any valid class labels")
    cw_cfg["class_counts"] = counts.tolist()
    logger.info("auto class counts from train dataset: %s", counts.tolist())


def _peek_resume_checkpoint(path: str) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    ck = load_checkpoint(path, map_location="cpu")
    if not isinstance(ck, dict):
        return None
    return ck


def _load_best_miou_from_dir(save_dir: str) -> float:
    best_path = os.path.join(save_dir, "best_miou.pth")
    if not os.path.isfile(best_path):
        return -1.0
    ck = load_checkpoint(best_path, map_location="cpu")
    if isinstance(ck, dict) and "miou" in ck:
        return float(ck["miou"])
    return -1.0


def _load_resume_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ck: dict,
    *,
    steps_per_epoch: int,
    save_dir: str,
    logger: logging.Logger,
) -> tuple[int, int, float, list[str]]:
    """Restore training state. Returns (start_epoch, global_step, best_miou, restored_keys)."""
    mod = model.module if hasattr(model, "module") else model
    restored: list[str] = []

    sd = ck.get("model")
    if isinstance(sd, dict):
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        restored.append("model")
        if comm.is_main_process():
            logger.info(
                "resume: loaded model weights (missing=%d unexpected=%d)",
                len(missing),
                len(unexpected),
            )

    completed_epoch = int(ck.get("epoch", 0))
    start_epoch = max(0, completed_epoch)
    if "global_step" in ck:
        global_step = max(0, int(ck["global_step"]))
    else:
        global_step = max(0, start_epoch * max(steps_per_epoch, 1))
        if comm.is_main_process() and start_epoch > 0:
            logger.warning(
                "resume: checkpoint has no global_step; estimated global_step=%d from epoch=%d",
                global_step,
                start_epoch,
            )

    if "best_miou" in ck:
        best_miou = float(ck["best_miou"])
    else:
        best_miou = _load_best_miou_from_dir(save_dir)
        if comm.is_main_process() and best_miou >= 0:
            logger.info("resume: best_miou=%.4f loaded from best_miou.pth", best_miou)

    partial = False
    if ck.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ck["optimizer"])
            restored.append("optimizer")
        except Exception as e:
            partial = True
            if comm.is_main_process():
                logger.warning("resume: optimizer state not restored (%s)", e)
    else:
        partial = True

    if ck.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ck["scheduler"])
            restored.append("scheduler")
        except Exception as e:
            partial = True
            if comm.is_main_process():
                logger.warning("resume: scheduler state not restored (%s)", e)
    else:
        partial = True

    if "scheduler" not in restored and global_step > 0:
        _fast_forward_scheduler(scheduler, global_step)
        restored.append(f"scheduler_fast_forward({global_step})")
        if comm.is_main_process():
            logger.warning(
                "resume: scheduler fast-forwarded %d steps (checkpoint had no scheduler state)",
                global_step,
            )

    if ck.get("scaler") is not None and scaler.is_enabled():
        try:
            scaler.load_state_dict(ck["scaler"])
            restored.append("scaler")
        except Exception as e:
            partial = True
            if comm.is_main_process():
                logger.warning("resume: GradScaler state not restored (%s)", e)

    if partial and comm.is_main_process():
        logger.warning(
            "resume: partial restore (model epoch=%d global_step=%d). "
            "Optimizer/scheduler/scaler may re-init; LR schedule can differ slightly.",
            start_epoch,
            global_step,
        )

    if comm.is_main_process():
        logger.info(
            "resume: continue from epoch %d/%d (next loop index %d), global_step=%d, restored=%s",
            completed_epoch,
            start_epoch,
            start_epoch,
            global_step,
            restored or ["none"],
        )
    return start_epoch, global_step, best_miou, restored


def _build_last_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    *,
    epoch: int,
    train_loss: float,
    global_step: int,
    best_miou: float,
    amp: bool,
) -> dict:
    mod = model.module if hasattr(model, "module") else model
    payload = {
        "model": mod.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "global_step": global_step,
        "best_miou": best_miou,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if amp:
        payload["scaler"] = scaler.state_dict()
    return payload


def _fast_forward_scheduler(scheduler: torch.optim.lr_scheduler.LRScheduler, steps: int) -> None:
    if steps <= 0:
        return
    for _ in range(steps):
        scheduler.step()


def create_ddp_model(model):
    if comm.get_world_size() == 1:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[comm.get_local_rank()],
        broadcast_buffers=False,
    )


@torch.no_grad()
def evaluate_val(model, loader, device, nc, ignore, amp, logger):
    model.eval()
    intersection_sum = torch.zeros(nc, device=device, dtype=torch.float64)
    union_sum = torch.zeros(nc, device=device, dtype=torch.float64)
    target_sum = torch.zeros(nc, device=device, dtype=torch.float64)
    loss_sum = 0.0
    loss_count = 0

    for batch in loader:
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(b)
        lv = float(out["loss"].item())
        loss_sum += lv
        loss_count += 1
        pred, segment = apply_inverse_mapping(out["seg_logits"], b)
        inter, uni, tgt = intersection_and_union(pred, segment, nc, ignore)
        intersection_sum += inter.double()
        union_sum += uni.double()
        target_sum += tgt.double()

    if comm.get_world_size() > 1:
        pack = torch.cat([intersection_sum, union_sum, target_sum])
        dist.all_reduce(pack)
        intersection_sum = pack[:nc]
        union_sum = pack[nc : 2 * nc]
        target_sum = pack[2 * nc :]
        loss_stats = torch.tensor([loss_sum, float(loss_count)], dtype=torch.float64, device=device)
        dist.all_reduce(loss_stats)
        loss_sum = float(loss_stats[0].item())
        loss_count = int(loss_stats[1].item())

    if comm.is_main_process():
        metrics = summarize_iou(
            intersection_sum.cpu().numpy(),
            union_sum.cpu().numpy(),
            target_sum.cpu().numpy(),
        )
        avg_loss = loss_sum / max(loss_count, 1)
        logger.info(
            "evaluate [val]: avg_loss=%.6f mIoU=%.4f OA=%.4f",
            avg_loss,
            metrics["mIoU"],
            metrics["OA"],
        )
        return avg_loss, metrics
    return None, None


def train_worker(cfg: dict):
    rank = comm.get_local_rank()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()
    if not comm.is_main_process():
        logger.setLevel(logging.WARNING)

    save_dir = cfg["save_path"]
    if comm.is_main_process():
        os.makedirs(save_dir, exist_ok=True)
        attach_file_logger(save_dir)

    log_iv = int(cfg.get("log_interval", 0) or 0)
    mc = cfg["model"]
    nc = int(mc["num_classes"])
    ignore_index = int(mc.get("ignore_index", -1))
    amp = bool(cfg.get("amp", True) and device.type == "cuda")
    clip_grad = float(cfg.get("clip_grad", 0.5))
    mix_prob = float(cfg.get("mix_prob", 0.8))
    epochs = max(1, int(cfg["epochs"]))
    eval_iv = max(1, int(cfg.get("eval_interval", 1)))
    world = comm.get_world_size()
    bs_global = int(cfg["batch_size"])
    bs = max(1, bs_global // world)
    expected_in = int(mc["backbone"]["in_channels"])

    train_ds = STSDDataset(
        cfg["train"]["split"],
        cfg["data_root"],
        pipeline="train",
        loop=int(cfg["train"].get("loop", 10)),
    )
    val_ds = STSDDataset(
        cfg["val"]["split"],
        cfg["data_root"],
        pipeline="val",
        loop=int(cfg["val"].get("loop", 1)),
    )
    _check_sample_feature_dim(train_ds[0], expected_in, "train dataset")
    _check_sample_feature_dim(val_ds[0], expected_in, "val dataset")
    _attach_auto_class_counts(cfg, train_ds, nc, ignore_index, logger)
    if comm.is_main_process():
        log_config_summary(logger, cfg, cfg.get("_config_path", ""))
        log_dataset_summary(logger, train_ds=train_ds, val_ds=val_ds)

    resume_path = cfg.get("resume")
    resume_ck = _peek_resume_checkpoint(resume_path) if resume_path else None
    if resume_path and resume_ck is None and comm.is_main_process():
        logger.warning("resume path not found or invalid, training from scratch: %s", resume_path)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world > 1 else None
    nw = max(int(cfg.get("num_workers", 0)), 0)

    collate_train = partial(stsd_collate_fn, mix_prob=mix_prob)
    drop_last_train = bool(cfg.get("drop_last_train", False))
    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=collate_train,
        drop_last=drop_last_train,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=max(1, int(cfg.get("batch_size_eval", bs_global)) // world),
        shuffle=False,
        sampler=val_sampler,
        num_workers=nw,
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=partial(stsd_collate_fn, mix_prob=0),
    )

    model = GSSNet(mc).to(device)
    if comm.is_main_process():
        log_model_summary(logger, model)

    model = create_ddp_model(model)
    optimizer = build_optimizer(model.module if hasattr(model, "module") else model, cfg)
    sched_cfg = cfg.get("scheduler", {})
    max_lr = resolve_onecycle_max_lr(sched_cfg, optimizer)
    if comm.is_main_process():
        max_lr_cfg = sched_cfg.get("max_lr")
        if max_lr_cfg is not None and len(max_lr_cfg) != len(optimizer.param_groups):
            logger.info(
                "scheduler.max_lr length %d != param_groups %d; OneCycleLR uses %s",
                len(max_lr_cfg),
                len(optimizer.param_groups),
                max_lr,
            )
    total_steps = max(1, len(train_dl)) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=float(sched_cfg.get("pct_start", 0.05)),
        div_factor=float(sched_cfg.get("div_factor", 10.0)),
        final_div_factor=float(sched_cfg.get("final_div_factor", 1000.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_path = os.path.join(save_dir, "best_miou.pth")
    last_path = os.path.join(save_dir, "last.pth")
    best_miou = -1.0
    global_step = 0
    start_epoch = 0
    shape_logged = False

    if resume_ck is not None and resume_path:
        start_epoch, global_step, best_miou, _ = _load_resume_state(
            model,
            optimizer,
            scheduler,
            scaler,
            resume_ck,
            steps_per_epoch=len(train_dl),
            save_dir=save_dir,
            logger=logger,
        )

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        tr_loss_sum = 0.0
        tr_loss_count = 0

        for batch in train_dl:
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.cuda.amp.autocast(enabled=amp) if amp else nullcontext()
            with ctx:
                out = model(b)
                loss = out["loss"]
            if not shape_logged and comm.is_main_process():
                logger.info(
                    "first batch shapes: coord=%s feat=%s segment=%s seg_logits=%s sparse_inverse_restored=%s",
                    tuple(b["coord"].shape),
                    tuple(b["feat"].shape),
                    tuple(b["segment"].shape) if "segment" in b else None,
                    tuple(out["seg_logits"].shape) if "seg_logits" in out else None,
                    out.get("sparse_inverse_restored", False),
                )
                shape_logged = True
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            scheduler.step()
            global_step += 1
            lv = float(loss.item())
            tr_loss_sum += lv
            tr_loss_count += 1
            if log_iv > 0 and global_step % log_iv == 0 and comm.is_main_process():
                log_epoch_seg(
                    logger,
                    epoch=epoch + 1,
                    epochs=epochs,
                    train_loss=lv,
                    lr=scheduler.get_last_lr()[0],
                    elapsed_s=0.0,
                    step=global_step,
                    step_loss=lv,
                )

        if world > 1:
            train_stats = torch.tensor([tr_loss_sum, float(tr_loss_count)], dtype=torch.float64, device=device)
            dist.all_reduce(train_stats)
            tr_loss_sum = float(train_stats[0].item())
            tr_loss_count = int(train_stats[1].item())
        tl = tr_loss_sum / max(tr_loss_count, 1) if tr_loss_count else float("nan")
        val_loss_m = val_metrics = None
        if (epoch + 1) % eval_iv == 0 or epoch + 1 == epochs:
            val_loss_m, val_metrics = evaluate_val(model, val_dl, device, nc, ignore_index, amp, logger)

        if comm.is_main_process():
            record = {"epoch": epoch + 1, "split": "train", "loss": tl, "lr": scheduler.get_last_lr()[0]}
            append_epoch_record(save_dir, record)
            if val_metrics is not None:
                append_epoch_record(
                    save_dir,
                    {
                        "epoch": epoch + 1,
                        "split": "val",
                        "loss": val_loss_m,
                        "mIoU": val_metrics["mIoU"],
                        "OA": val_metrics["OA"],
                        "per_class_IoU": val_metrics.get("per_class_IoU"),
                        "per_class_Recall": val_metrics.get("per_class_Recall"),
                        "per_class_Precision": val_metrics.get("per_class_Precision"),
                    },
                )
                if val_metrics["mIoU"] >= best_miou:
                    best_miou = val_metrics["mIoU"]
                    mod = model.module if hasattr(model, "module") else model
                    save_checkpoint(best_path, {"model": mod.state_dict(), "epoch": epoch + 1, "miou": best_miou})
                    logger.info("new best val mIoU %.4f -> %s", best_miou, best_path)
            mod = model.module if hasattr(model, "module") else model
            save_checkpoint(
                last_path,
                _build_last_checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=epoch + 1,
                    train_loss=tl,
                    global_step=global_step,
                    best_miou=best_miou,
                    amp=amp,
                ),
            )
            log_epoch_seg(
                logger,
                epoch=epoch + 1,
                epochs=epochs,
                train_loss=tl,
                lr=scheduler.get_last_lr()[0],
                elapsed_s=0.0,
                val_loss=val_loss_m,
                metrics=val_metrics,
                num_classes=nc,
                best_miou=best_miou if val_metrics else None,
                saved_best=val_metrics is not None and val_metrics["mIoU"] >= best_miou,
            )

    if comm.is_main_process():
        try:
            plot_training_curves(save_dir, num_classes=nc)
        except Exception as e:
            logger.warning("plot_training_curves failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Train STSD semantic segmentation (GSS-Net)")
    parser.add_argument("--config", default=CONFIG_FILE, help="Path to YAML config")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume checkpoint path, or 'auto' for save_path/last.pth (overrides yaml resume)",
    )
    args = parser.parse_args()
    config_file = args.config or CONFIG_FILE
    if config_file and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        cfg_probe = load_config(resolve_config_path(config_file))
        cvd = cfg_probe.get("cuda_visible_devices")
        if cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    config_path = resolve_config_path(config_file)
    cfg = load_config(config_path)
    if args.resume is not None:
        cfg["resume"] = resolve_resume(args.resume, Path(cfg["_config_dir"]), cfg.get("save_path"))
    num_gpus = int(cfg.get("num_gpus", 1))
    launch(train_worker, num_gpus_per_machine=num_gpus, cfg=(cfg,))


if __name__ == "__main__":
    main()
