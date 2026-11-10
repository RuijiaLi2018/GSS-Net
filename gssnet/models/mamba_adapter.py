from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from timm.layers import DropPath

from gssnet.models.offset_utils import offset2batch
from gssnet.models.serialization_codec import encode


def _grid_depth(grid: torch.Tensor) -> int:
    m = int(grid.max().clamp(min=0).item())
    return min(max(1, m.bit_length() if m > 0 else 1), 16)


class MambaGlobalAdapter(nn.Module):
    """Apply two independent Mamba branches along Hilbert paths and fuse them."""

    PATHS = ("hilbert", "hilbert-trans")

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path: float = 0.0,
        alpha_init: float = 1.0e-3,
    ):
        super().__init__()
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except ImportError as e:
            raise ImportError("Install causal-conv1d and mamba-ssm (see README).") from e

        self.paths = self.PATHS
        self.norm = nn.LayerNorm(dim)
        self.mamba = nn.ModuleList(
            [
                Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in self.paths
            ]
        )
        self.proj = nn.Linear(dim * len(self.paths), dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _order_for_path(
        self,
        path: str,
        grid_coord: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if path not in self.PATHS:
            raise ValueError(f"Unsupported Mamba path {path!r}")
        depth = _grid_depth(grid_coord)
        batch_index = torch.zeros(grid_coord.shape[0], dtype=torch.long, device=device)
        code = encode(grid_coord.long(), batch_index, depth=depth, order=path)
        return torch.argsort(code, stable=True)

    def _run_branch(
        self,
        branch: nn.Module,
        features: torch.Tensor,
        order: torch.Tensor,
    ) -> torch.Tensor:
        inverse = torch.empty_like(order)
        inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
        sequence = self.norm(features)[order].unsqueeze(0).float()
        with torch.amp.autocast("cuda", enabled=False):
            output = branch(sequence).squeeze(0)
        return output[inverse]

    def forward(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        batch_or_offset: Optional[torch.Tensor] = None,
        *,
        grid_coord: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if coord.shape[0] != feat.shape[0]:
            raise ValueError(
                f"coord/feat length mismatch: coord={coord.shape[0]}, feat={feat.shape[0]}"
            )
        if grid_coord is None:
            raise ValueError("grid_coord is required for Hilbert Mamba paths.")
        if batch is None:
            if offset is None:
                if batch_or_offset is None:
                    raise ValueError("Need batch or offset")
                offset = batch_or_offset
            batch = offset2batch(offset.long())

        out_feat = feat.clone()
        batch_count = int(batch.max().item()) + 1
        for batch_index in range(batch_count):
            selected = torch.nonzero(batch == batch_index, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            features = feat[selected]
            grids = grid_coord[selected].long()
            branch_outputs = [
                self._run_branch(
                    self.mamba[path_index],
                    features,
                    self._order_for_path(path, grids, feat.device),
                )
                for path_index, path in enumerate(self.paths)
            ]
            with torch.amp.autocast("cuda", enabled=False):
                fused = torch.cat(branch_outputs, dim=1)
                residual = self.drop_path(self.proj(fused))
            out_feat[selected] = features + self.alpha.to(feat.dtype) * residual.to(feat.dtype)
        return out_feat


def yaml_stage_to_enc(stage: str) -> str:
    if stage.startswith("enc"):
        return stage
    if stage.startswith("stage"):
        return "enc" + stage[len("stage") :]
    return stage
