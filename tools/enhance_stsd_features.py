#!/usr/bin/env python3
"""Offline geometry feature enhancement for STSD Pointcept-format scenes (5-D feat.npy)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import threading
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stsd" / "feature_enhance.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - dependency check path
    raise ImportError("scipy is required for offline KNN. Install scipy first.") from exc


EPS = 1.0e-6
_Z_ORDER_MOD = None
_HILBERT_MOD = None
FEATURE_NAMES = [
    "curvature",
    "linearity",
    "planarity",
    "scattering",
    "density",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _as_float_2d(arr: np.ndarray, width: int | None = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if width is not None and arr.shape[1] != width:
        raise ValueError(f"expected width {width}, got shape {arr.shape}")
    return arr


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _z_order_module():
    global _Z_ORDER_MOD
    if _Z_ORDER_MOD is None:
        _Z_ORDER_MOD = _load_module_from_path(
            "gssnet_offline_z_order",
            PROJECT_ROOT / "gssnet" / "models" / "serialization" / "z_order.py",
        )
    return _Z_ORDER_MOD


def _hilbert_module():
    global _HILBERT_MOD
    if _HILBERT_MOD is None:
        _HILBERT_MOD = _load_module_from_path(
            "gssnet_offline_hilbert",
            PROJECT_ROOT / "gssnet" / "models" / "serialization" / "hilbert.py",
        )
    return _HILBERT_MOD


def _read_scene_dir(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for fp in sorted(path.glob("*.npy")):
        out[fp.stem] = np.load(fp, allow_pickle=True)
    if "coord" not in out:
        raise KeyError(f"{path}: missing coord.npy")
    return out


def _read_dict_file(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        return {k: npz[k] for k in npz.files}
    if path.suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise TypeError(f"{path}: expected dict checkpoint/data")
        return {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
            for k, v in obj.items()
        }
    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        if getattr(obj, "shape", None) == ():
            item = obj.item()
            if isinstance(item, dict):
                return dict(item)
    raise ValueError(f"Unsupported file format: {path}")


def _find_inputs(input_root: Path, patterns: list[str]) -> list[Path]:
    scene_dirs = [p for p in input_root.rglob("*") if p.is_dir() and (p / "coord.npy").is_file()]
    if scene_dirs:
        return sorted(scene_dirs)
    found: list[Path] = []
    for patt in patterns:
        found.extend(input_root.rglob(patt))
    return sorted({p for p in found if p.is_file()})


def _copy_scene(src: Path, dst: Path, overwrite: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if src.is_dir():
        for fp in src.iterdir():
            if fp.is_file():
                out = dst / fp.name
                if overwrite or not out.exists():
                    shutil.copy2(fp, out)


def _grid_for_serialization(coord: np.ndarray, depth: int = 16) -> np.ndarray:
    mn = coord.min(axis=0, keepdims=True)
    mx = coord.max(axis=0, keepdims=True)
    scale = float((1 << depth) - 1)
    grid = np.floor((coord - mn) / (mx - mn + EPS) * scale).astype(np.int64)
    return np.clip(grid, 0, int(scale))


def _serialized_order(coord: np.ndarray, serialization: str) -> np.ndarray:
    import torch

    order_map = {
        "morton": "z",
        "z": "z",
        "z-trans": "z-trans",
        "morton-trans": "z-trans",
        "hilbert": "hilbert",
        "hilbert-trans": "hilbert-trans",
    }
    key = order_map.get(serialization)
    if key is None:
        raise ValueError(f"serialization={serialization}")
    grid = torch.from_numpy(_grid_for_serialization(coord)).long()
    if key in ("z", "z-trans"):
        if key == "z-trans":
            grid = grid[:, [1, 0, 2]]
        z_order = _z_order_module()
        code = z_order.xyz2key(
            grid[:, 0],
            grid[:, 1],
            grid[:, 2],
            b=None,
            depth=16,
        ).cpu().numpy()
    elif key in ("hilbert", "hilbert-trans"):
        if key == "hilbert-trans":
            grid = grid[:, [1, 0, 2]]
        hilbert = _hilbert_module()
        code = hilbert.encode(grid, num_dims=3, num_bits=16).cpu().numpy()
    else:
        raise ValueError(f"serialization={serialization}")
    return np.argsort(code, kind="stable")


def _query_knn(coord: np.ndarray, k: int, chunk_size: int) -> np.ndarray:
    n = coord.shape[0]
    kk = max(1, min(int(k) + 1, n))
    tree = cKDTree(coord)
    inds = []
    for start in range(0, n, max(int(chunk_size), 1)):
        stop = min(start + max(int(chunk_size), 1), n)
        _, idx = tree.query(coord[start:stop], k=kk, workers=-1)
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx.reshape(-1, 1)
        inds.append(idx[:, 1:] if idx.shape[1] > 1 else idx[:, :0])
    return np.concatenate(inds, axis=0)


def _serialized_window_neighbors(
    coord: np.ndarray,
    k: int,
    serialization: str,
    max_radius: float | None,
    min_neighbors: int,
    fallback: str,
    chunk_size: int,
) -> np.ndarray:
    n = coord.shape[0]
    order = _serialized_order(coord, serialization)
    pos_to_idx = order
    idx_to_pos = np.empty(n, dtype=np.int64)
    idx_to_pos[order] = np.arange(n, dtype=np.int64)
    out = np.empty((n, min(k, max(n - 1, 0))), dtype=np.int64)
    knn = None
    for i in range(n):
        pos = int(idx_to_pos[i])
        half = max(k // 2, 1)
        lo = max(0, pos - half)
        hi = min(n, pos + half + 1)
        cand = pos_to_idx[lo:hi]
        cand = cand[cand != i]
        while cand.size < min(k, n - 1):
            if fallback == "expand_window":
                half *= 2
                lo = max(0, pos - half)
                hi = min(n, pos + half + 1)
                cand = pos_to_idx[lo:hi]
                cand = cand[cand != i]
                if lo == 0 and hi == n:
                    break
            else:
                if knn is None:
                    knn = _query_knn(coord, k, chunk_size)
                cand = knn[i]
                break
        if max_radius is not None and cand.size:
            dist = np.linalg.norm(coord[cand] - coord[i], axis=1)
            cand = cand[dist <= float(max_radius)]
        if cand.size < min_neighbors:
            if fallback == "knn":
                if knn is None:
                    knn = _query_knn(coord, k, chunk_size)
                cand = knn[i]
        cand = cand[: out.shape[1]]
        if cand.size < out.shape[1]:
            pad = np.full(out.shape[1] - cand.size, i, dtype=np.int64)
            cand = np.concatenate([cand, pad])
        out[i] = cand
    return out


def _serialized_candidate_knn(
    coord: np.ndarray,
    k: int,
    serialization: str,
    candidate_multiplier: int,
    max_radius: float | None,
    min_neighbors: int,
    fallback: str,
    chunk_size: int,
) -> np.ndarray:
    n = coord.shape[0]
    order = _serialized_order(coord, serialization)
    idx_to_pos = np.empty(n, dtype=np.int64)
    idx_to_pos[order] = np.arange(n, dtype=np.int64)
    width = min(max(k * int(candidate_multiplier), k), max(n - 1, 0))
    out = np.empty((n, min(k, max(n - 1, 0))), dtype=np.int64)
    knn = None
    for i in range(n):
        pos = int(idx_to_pos[i])
        half = max(width // 2, 1)
        cand = order[max(0, pos - half) : min(n, pos + half + 1)]
        cand = cand[cand != i]
        if cand.size:
            dist = np.linalg.norm(coord[cand] - coord[i], axis=1)
            if max_radius is not None:
                keep = dist <= float(max_radius)
                cand, dist = cand[keep], dist[keep]
            nn = cand[np.argsort(dist)[: out.shape[1]]]
        else:
            nn = cand
        if nn.size < min_neighbors and fallback == "knn":
            if knn is None:
                knn = _query_knn(coord, k, chunk_size)
            nn = knn[i]
        if nn.size < out.shape[1]:
            nn = np.concatenate([nn, np.full(out.shape[1] - nn.size, i, dtype=np.int64)])
        out[i] = nn[: out.shape[1]]
    return out


def _neighbors(coord: np.ndarray, cfg: dict[str, Any], runtime: dict[str, Any]) -> np.ndarray:
    n = coord.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    k = min(int(cfg.get("k", 16)), n - 1)
    mode = str(cfg.get("mode", "knn")).lower()
    chunk_size = int(runtime.get("chunk_size", 200000))
    if mode == "knn":
        return _query_knn(coord, k, chunk_size)
    max_radius = cfg.get("max_radius")
    min_neighbors = int(cfg.get("min_neighbors", 8))
    fallback = str(cfg.get("serialized_fallback", cfg.get("fallback", "knn")))
    serialization = str(cfg.get("serialization", "hilbert"))
    if mode == "serialized_window":
        return _serialized_window_neighbors(coord, k, serialization, max_radius, min_neighbors, fallback, chunk_size)
    if mode == "serialized_candidate_knn":
        return _serialized_candidate_knn(
            coord,
            k,
            serialization,
            int(cfg.get("candidate_multiplier", 4)),
            max_radius,
            min_neighbors,
            fallback,
            chunk_size,
        )
    raise ValueError(f"neighbor.mode={mode}")


def _pca_geometry(coord: np.ndarray, neigh: np.ndarray, normal_abs: bool) -> tuple[np.ndarray, np.ndarray]:
    n = coord.shape[0]
    normal = np.zeros((n, 3), dtype=np.float32)
    normal[:, 2] = 1.0
    geom = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        idx = neigh[i] if neigh.size else np.empty(0, dtype=np.int64)
        idx = idx[idx != i]
        if idx.size < 3:
            continue
        pts = coord[idx]
        pts = pts - pts.mean(axis=0, keepdims=True)
        try:
            cov = (pts.T @ pts) / max(pts.shape[0] - 1, 1)
            vals, vecs = np.linalg.eigh(cov.astype(np.float64))
            vals = np.maximum(vals.astype(np.float32), 0.0)
            l0, l1, l2 = vals
            nvec = vecs[:, 0].astype(np.float32)
            normal[i] = np.abs(nvec) if normal_abs else nvec
            geom[i, 0] = l0 / (l0 + l1 + l2 + EPS)
            geom[i, 1] = (l2 - l1) / (l2 + EPS)
            geom[i, 2] = (l1 - l0) / (l2 + EPS)
            geom[i, 3] = l0 / (l2 + EPS)
        except np.linalg.LinAlgError:
            continue
    return normal, geom


def _density(coord: np.ndarray, neigh: np.ndarray, density_type: str, norm_mode: str) -> np.ndarray:
    if neigh.size == 0:
        return np.zeros((coord.shape[0], 1), dtype=np.float32)
    dens = np.zeros((coord.shape[0],), dtype=np.float32)
    for i in range(coord.shape[0]):
        idx = neigh[i]
        idx = idx[idx != i]
        if idx.size == 0:
            continue
        md = float(np.linalg.norm(coord[idx] - coord[i], axis=1).mean())
        dens[i] = 1.0 / (md + EPS) if density_type == "inverse_mean_distance" else md
    if norm_mode == "minmax":
        dens = 2.0 * (dens - dens.min()) / (dens.max() - dens.min() + EPS) - 1.0
    else:
        dens = (dens - dens.mean()) / (dens.std() + EPS)
    return dens.reshape(-1, 1).astype(np.float32)


def build_feature(
    scene: dict[str, np.ndarray],
    feature_cfg: dict[str, Any],
    neighbor_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    *,
    log_prefix: str = "",
):
    coord = _as_float_2d(scene["coord"], 3)
    print(f"{log_prefix}coord points={coord.shape[0]}: build geometry features", flush=True)
    has_i = 1.0 if scene.get("strength", scene.get("intensity")) is not None else 0.0
    has_rgb = 1.0 if scene.get("color", scene.get("rgb")) is not None else 0.0
    print(
        f"{log_prefix}neighbor search mode={(neighbor_cfg or {}).get('mode', 'knn')} "
        f"k={(neighbor_cfg or {}).get('k', 16)}",
        flush=True,
    )
    neigh = _neighbors(coord, neighbor_cfg, runtime_cfg)
    print(f"{log_prefix}PCA geometry: normals/curvature/linearity/planarity/scattering", flush=True)
    normal, geom = _pca_geometry(coord, neigh, bool(feature_cfg.get("normal_abs", False)))
    print(f"{log_prefix}density feature", flush=True)
    dens = _density(
        coord,
        neigh,
        str(feature_cfg.get("density_type", "mean_distance")),
        str(feature_cfg.get("density_norm_mode", "standard")),
    )
    feat = np.concatenate(
        [
            geom[:, 0:1],
            geom[:, 1:2],
            geom[:, 2:3],
            geom[:, 3:4],
            dens,
        ],
        axis=1,
    ).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    if feat.shape[1] != int(feature_cfg.get("output_dim", 5)):
        raise RuntimeError(f"feature dim {feat.shape[1]} != configured output_dim")
    print(f"{log_prefix}assembled feat shape={feat.shape}", flush=True)
    return feat, normal.astype(np.float32), has_i, has_rgb


def _relative_output_path(src: Path, input_root: Path, output_root: Path) -> Path:
    rel = src.relative_to(input_root)
    return output_root / rel


def process_one(src: Path, input_root: Path, output_root: Path, cfg: dict[str, Any]) -> tuple[int, int, int]:
    overwrite = bool(cfg.get("overwrite", False))
    feature_cfg = dict(cfg.get("feature") or {})
    neighbor_cfg = dict(cfg.get("neighbor") or {})
    runtime_cfg = dict(cfg.get("runtime") or {})
    if src.is_dir():
        print(f"  read scene dir: {src}", flush=True)
        scene = _read_scene_dir(src)
        dst = _relative_output_path(src, input_root, output_root)
        feat_path = dst / "feat.npy"
        if feat_path.exists() and not overwrite:
            return 0, 0, 0
        _copy_scene(src, dst, overwrite=overwrite)
        feat, normal, has_i, has_rgb = build_feature(
            scene,
            feature_cfg,
            neighbor_cfg,
            runtime_cfg,
            log_prefix="  ",
        )
        print(f"  save feat -> {feat_path}", flush=True)
        np.save(feat_path, feat.astype(runtime_cfg.get("save_dtype", "float32")))
        np.save(dst / "normal.npy", normal.astype(runtime_cfg.get("save_dtype", "float32")))
        return int(scene["coord"].shape[0]), int(has_i == 0.0), int(has_rgb == 0.0)

    print(f"  read dict file: {src}", flush=True)
    scene = _read_dict_file(src)
    feat, normal, has_i, has_rgb = build_feature(
        scene,
        feature_cfg,
        neighbor_cfg,
        runtime_cfg,
        log_prefix="  ",
    )
    dst = _relative_output_path(src, input_root, output_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return 0, 0, 0
    scene["feat"] = feat.astype(runtime_cfg.get("save_dtype", "float32"))
    scene["normal"] = normal.astype(runtime_cfg.get("save_dtype", "float32"))
    if src.suffix == ".npz":
        print(f"  save npz -> {dst}", flush=True)
        np.savez_compressed(dst, **scene)
    elif src.suffix == ".npy":
        print(f"  save npy -> {dst}", flush=True)
        np.save(dst, scene, allow_pickle=True)
    else:
        import torch

        print(f"  save torch -> {dst}", flush=True)
        torch.save(scene, dst)
    return int(scene["coord"].shape[0]), int(has_i == 0.0), int(has_rgb == 0.0)


def _resolve_num_workers(cli_value: int | None, runtime_cfg: dict[str, Any]) -> int:
    if cli_value is not None:
        return max(1, int(cli_value))
    cfg_val = (runtime_cfg or {}).get("num_workers")
    if cfg_val is not None:
        return max(1, int(cfg_val))
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        return max(1, int(env))
    return max(1, min(8, os.cpu_count() or 1))


def _estimate_scene_points(src: Path) -> int:
    coord_path = src / "coord.npy" if src.is_dir() else None
    if coord_path is not None and coord_path.is_file():
        return max(1, coord_path.stat().st_size // 12)
    return 1


def _scene_peak_bytes(points: int, runtime_cfg: dict[str, Any]) -> int:
    bytes_per_point = int(runtime_cfg.get("peak_bytes_per_point", 1400))
    overhead_gb = float(runtime_cfg.get("scene_memory_overhead_gb", 2.0))
    return points * bytes_per_point + int(overhead_gb * (1024**3))


def _run_parallel_jobs(
    inputs: list[Path],
    input_root: Path,
    output_root: Path,
    cfg_copy: dict[str, Any],
    num_workers: int,
    consume_result,
) -> None:
    runtime_cfg = dict(cfg_copy.get("runtime") or {})
    memory_budget_gb = float(runtime_cfg.get("memory_budget_gb", 0) or 0)
    tasks = [(str(src), str(input_root), str(output_root), cfg_copy) for src in inputs]
    peak_bytes = [_scene_peak_bytes(_estimate_scene_points(src), runtime_cfg) for src in inputs]

    if memory_budget_gb > 0:
        budget_bytes = int(memory_budget_gb * (1024**3))
        print(
            f"[parallel] memory_budget_gb={memory_budget_gb} "
            f"peak_bytes_per_point={runtime_cfg.get('peak_bytes_per_point', 1400)}",
            flush=True,
        )
        lock = threading.Lock()
        in_flight_bytes = 0
        pending = list(enumerate(tasks, start=1))
        futures: dict = {}

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            while pending or futures:
                while pending:
                    idx, task = pending[0]
                    need = peak_bytes[idx - 1]
                    with lock:
                        at_worker_cap = len(futures) >= num_workers
                        over_budget = in_flight_bytes + need > budget_bytes and bool(futures)
                    if at_worker_cap or over_budget:
                        break
                    pending.pop(0)
                    fut = pool.submit(_process_one_worker, task)
                    with lock:
                        in_flight_bytes += need
                    futures[fut] = (idx, need, task[0])

                if not futures:
                    if not pending:
                        break
                    idx, task = pending.pop(0)
                    need = peak_bytes[idx - 1]
                    print(
                        f"[parallel] scene exceeds memory budget alone, running serially: {task[0]}",
                        flush=True,
                    )
                    fut = pool.submit(_process_one_worker, task)
                    with lock:
                        in_flight_bytes += need
                    futures[fut] = (idx, need, task[0])

                done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    idx, need, src_s = futures.pop(fut)
                    with lock:
                        in_flight_bytes -= need
                    src_s, n, mi, mr, err = fut.result()
                    print(f"[{idx}/{len(inputs)}] finished {src_s}", flush=True)
                    consume_result(idx, src_s, n, mi, mr, err)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_one_worker, task): idx for idx, task in enumerate(tasks, start=1)}
        while futures:
            done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done_set:
                idx = futures.pop(fut)
                src_s, n, mi, mr, err = fut.result()
                print(f"[{idx}/{len(inputs)}] finished {src_s}", flush=True)
                consume_result(idx, src_s, n, mi, mr, err)


def _process_one_worker(task: tuple[str, str, str, dict[str, Any]]) -> tuple[str, int, int, int, str | None]:
    src_s, input_root_s, output_root_s, cfg = task
    try:
        n, mi, mr = process_one(Path(src_s), Path(input_root_s), Path(output_root_s), cfg)
        return src_s, n, mi, mr, None
    except Exception as exc:
        return src_s, 0, 0, 0, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML config path. Default: configs/stsd/feature_enhance.yaml",
    )
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers; default runtime.num_workers or SLURM_CPUS_PER_TASK",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    print(f"[config] {cfg_path}", flush=True)
    cfg = _load_yaml(cfg_path)
    config_base = cfg_path.parent
    runtime_cfg = dict(cfg.get("runtime") or {})
    input_root = _resolve_path(cfg["input_root"], config_base)
    output_root = _resolve_path(cfg["output_root"], config_base)
    patterns = list(cfg.get("file_patterns") or ["*.pth", "*.pt", "*.npy", "*.npz"])
    num_workers = _resolve_num_workers(args.num_workers, runtime_cfg)
    print(f"[scan] input_root={input_root}", flush=True)
    print(f"[scan] output_root={output_root}", flush=True)
    print(f"[scan] patterns={patterns}", flush=True)
    print(f"[parallel] num_workers={num_workers}", flush=True)
    inputs = _find_inputs(input_root, patterns)
    print(f"[scan] found {len(inputs)} scene/file inputs", flush=True)
    if inputs:
        est_points = [_estimate_scene_points(src) for src in inputs]
        largest = max(est_points)
        print(
            f"[scan] estimated points/scene: min={min(est_points):,} "
            f"median={int(np.median(est_points)):,} max={largest:,}",
            flush=True,
        )
    if args.limit_files is not None:
        inputs = inputs[: max(int(args.limit_files), 0)]
        print(f"[scan] limit_files={args.limit_files}, processing {len(inputs)} inputs", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    processed = 0
    total_points = 0
    missing_i = 0
    missing_rgb = 0
    cfg_copy = dict(cfg)

    def _consume_result(i: int, src_label: str, n: int, mi: int, mr: int, err: str | None) -> None:
        nonlocal processed, total_points, missing_i, missing_rgb
        if err is not None:
            print(f"[{i}/{len(inputs)}] failed {src_label}: {err}", file=sys.stderr)
            raise RuntimeError(err)
        if n > 0:
            processed += 1
            total_points += n
            missing_i += mi
            missing_rgb += mr
            print(f"[{i}/{len(inputs)}] processed {src_label} points={n}", flush=True)
        else:
            print(f"[{i}/{len(inputs)}] skipped {src_label}", flush=True)

    if num_workers <= 1 or len(inputs) <= 1:
        for i, src in enumerate(inputs, start=1):
            print(f"[{i}/{len(inputs)}] start {src}", flush=True)
            n, mi, mr = process_one(src, input_root, output_root, cfg_copy)
            _consume_result(i, str(src), n, mi, mr, None)
    else:
        _run_parallel_jobs(inputs, input_root, output_root, cfg_copy, num_workers, _consume_result)
    elapsed = time.time() - t0
    avg_points = float(total_points) / max(processed, 1)
    print("===== offline feature enhancement summary =====")
    print(f"processed files: {processed}")
    print(f"feature dim: {len(FEATURE_NAMES)}")
    print(f"missing intensity count: {missing_i}")
    print(f"missing rgb count: {missing_rgb}")
    print(f"average points per scene: {avg_points:.1f}")
    print(f"neighbor mode: {(cfg.get('neighbor') or {}).get('mode', 'knn')}")
    print(f"num_workers: {num_workers}")
    print(f"time cost: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
