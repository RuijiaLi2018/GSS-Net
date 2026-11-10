from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from gssnet.models.sparse_backbone import GSSBackbone
from gssnet.models.point_structure import Point
from gssnet.models.mamba_adapter import MambaGlobalAdapter, yaml_stage_to_enc
from gssnet.utils.losses import build_seg_criteria

INPUT_CHANNELS = 17
NUM_CLASSES = 12
MAMBA_STAGE = "stage4"


def unravel_point(point: Point) -> Point:
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


def resolve_point_feats_for_head(point: Point, input_dict: Mapping | None = None) -> tuple[torch.Tensor, Point]:
    point = unravel_point(point)
    feat = point.feat
    if input_dict is not None and "segment" in input_dict:
        n_feat = int(feat.shape[0])
        n_seg = int(input_dict["segment"].shape[0])
        if n_feat != n_seg and not ("inverse" in input_dict and "origin_segment" in input_dict):
            raise RuntimeError(
                f"feature/label length mismatch before head: feat={n_feat}, segment={n_seg}"
            )
    return feat, point


def _check_inverse(inv: torch.Tensor, logits_n: int, what: str) -> None:
    if inv.numel() == 0:
        return
    inv_min = int(inv.min().item())
    inv_max = int(inv.max().item())
    if inv_min < 0 or inv_max >= logits_n:
        raise RuntimeError(f"{what}: inverse out of range for logits N={logits_n}")


def resolve_logits_for_loss_and_output(
    seg_logits: torch.Tensor,
    input_dict: Mapping,
    point: Point,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    restored = False
    target = None
    if "origin_segment" in input_dict:
        target = input_dict["origin_segment"].long()
        if int(seg_logits.shape[0]) == int(target.shape[0]):
            restored = True
        elif "inverse" in input_dict:
            inv = input_dict["inverse"].long()
            _check_inverse(inv, int(seg_logits.shape[0]), "resolve_logits")
            seg_logits = seg_logits[inv]
            restored = True
    elif "segment" in input_dict:
        target = input_dict["segment"].long()

    if target is not None and int(seg_logits.shape[0]) != int(target.shape[0]):
        raise RuntimeError(
            f"logits/target mismatch: logits={int(seg_logits.shape[0])}, target={int(target.shape[0])}"
        )
    return seg_logits, target, restored


def _build_mamba_adapters(enc_channels: tuple[int, ...]) -> nn.ModuleDict:
    enc_name = yaml_stage_to_enc(MAMBA_STAGE)
    idx = int(enc_name.replace("enc", ""))
    dim = int(enc_channels[idx])
    adapters = nn.ModuleDict(
        {
            enc_name: MambaGlobalAdapter(
                dim=dim,
                d_state=16,
                d_conv=4,
                expand=2,
                drop_path=0.0,
                alpha_init=1.0e-3,
            )
        }
    )
    return adapters


class GSSNet(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        backbone_cfg = dict(cfg["backbone"])
        ec = tuple(backbone_cfg["enc_channels"])
        mamba = _build_mamba_adapters(ec)
        m_stages = {yaml_stage_to_enc(MAMBA_STAGE)}
        dilated = dict(backbone_cfg.pop("dilated_attention", {}) or {})
        self.backbone = GSSBackbone(
            mamba_adapters=mamba,
            mamba_after_enc_stages=m_stages,
            dilated_attention=dilated,
            **backbone_cfg,
        )
        self.out_channels = int(cfg["backbone_out_channels"])
        self.num_classes = int(cfg.get("num_classes", NUM_CLASSES))
        self.ignore_index = int(cfg.get("ignore_index", -1))
        self.seg_head = nn.Linear(self.out_channels, self.num_classes)
        self.criteria = build_seg_criteria(cfg.get("loss", {}), default_ignore_index=self.ignore_index)

    def forward(self, input_dict, return_point=False):
        point = self.backbone(Point(input_dict))
        feat, point = resolve_point_feats_for_head(point, input_dict)
        seg_logits = self.seg_head(feat)
        seg_logits, target, restored = resolve_logits_for_loss_and_output(
            seg_logits, input_dict, point
        )
        return_dict = {"sparse_inverse_restored": restored}
        if return_point:
            return_dict["point"] = point
        if target is not None:
            return_dict["loss"] = self.criteria(seg_logits, target)
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict
