from __future__ import annotations

import math
from functools import partial
from typing import Iterable, Mapping, Optional

import addict
import spconv.pytorch as spconv
import torch

_SPCONV_ALGO = spconv.ConvAlgo.Native
import torch.nn as nn
import torch_scatter
from timm.layers import DropPath

from gssnet.models.modules import PointModule, PointSequential
from gssnet.models.offset_utils import batch2offset, offset2bincount
from gssnet.models.point_structure import Point


def clear_point_attn_cache(point: Point) -> None:
    stale = [
        k
        for k in list(point.keys())
        if k in ("pad", "unpad", "cu_seqlens_key")
        or k.startswith("rel_pos_")
        or k.startswith("pad_")
        or k.startswith("unpad_")
        or k.startswith("cu_seqlens_")
    ]
    for k in stale:
        point.pop(k)


def _check_gather_indices(index: torch.Tensor, size: int, what: str) -> None:
    if index.numel() == 0:
        return
    mx = int(index.max())
    mn = int(index.min())
    if mn < 0 or mx >= size:
        raise RuntimeError(
            f"{what}: index out of range (min={mn}, max={mx}, size={size})"
        )


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        upcast_attention=True,
        upcast_softmax=True,
        dilations=None,
        dilation_fusion="mean",
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.dilations = tuple(int(d) for d in (dilations or [1]))
        self.dilation_fusion = str(dilation_fusion)
        if self.dilation_fusion not in {"mean", "sum", "concat_linear"}:
            raise ValueError(
                f"Unsupported dilation_fusion={self.dilation_fusion!r}; "
                "expected mean, sum, or concat_linear"
            )

        self.patch_size_max = patch_size
        self.patch_size = 0
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        if self.dilation_fusion == "concat_linear" and len(self.dilations) > 1:
            self.dilation_proj = nn.Linear(channels * len(self.dilations), channels)
        else:
            self.dilation_proj = None
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        n = int(point.feat.shape[0])
        ps = max(int(self.patch_size), 1)
        pad_key = f"pad_{self.order_index}_{n}_{ps}"
        unpad_key = f"unpad_{self.order_index}_{n}_{ps}"
        cu_key = f"cu_seqlens_{self.order_index}_{n}_{ps}"
        if pad_key not in point or unpad_key not in point or cu_key not in point:
            offset = point.offset
            bincount = offset2bincount(offset)
            ps = max(self.patch_size, 1)
            bincount_pad = torch.div(bincount + ps - 1, ps, rounding_mode="trunc") * ps
            mask_pad = bincount > ps
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - ps
                        + (bincount[i] % ps) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * ps
                        + (bincount[i] % ps) : _offset_pad[i + 1]
                        - ps
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=ps,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=int(_offset_pad[-1])
            )
        return point[pad_key], point[unpad_key], point[cu_key]

    def _standard_attention_branch(self, point, qkv_all: torch.Tensor) -> torch.Tensor:
        H = self.num_heads
        K = self.patch_size
        C = self.channels
        n_ord = int(point.serialized_order.shape[1])
        pad, unpad, _ = self.get_padding_and_inverse(point)
        padded_n = int(pad.shape[0])
        _check_gather_indices(pad, n_ord, "dilated attention pad")
        order = point.serialized_order[self.order_index][pad]
        inv_map = point.serialized_inverse[self.order_index]
        _check_gather_indices(inv_map, int(unpad.shape[0]), "dilated serialized_inverse")
        inverse = unpad[inv_map]
        _check_gather_indices(inverse, padded_n, "dilated attention unpad")
        qkv = qkv_all[order]
        q, k, v = (
            qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        )
        if self.upcast_attention:
            q = q.float()
            k = k.float()
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.upcast_softmax:
            attn = attn.float()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        return feat[inverse]

    def _manual_attention_for_order(
        self,
        point,
        order: torch.Tensor,
        dilation: int,
        qkv_all: torch.Tensor,
    ) -> torch.Tensor:

        H = self.num_heads
        C = self.channels
        K = max(int(self.patch_size), 1)
        out = torch.zeros_like(point.feat)
        bmax = int(point.batch.max().item()) + 1 if point.batch.numel() else 0
        for bidx in range(bmax):
            seq = order[point.batch[order] == bidx]
            if seq.numel() == 0:
                continue
            pieces = [seq] if dilation <= 1 else [seq[r::dilation] for r in range(dilation)]
            for piece in pieces:
                if piece.numel() == 0:
                    continue
                full = int(piece.numel()) // K * K
                if full > 0:
                    idx_full = piece[:full].reshape(-1, K)
                    qkv = qkv_all[idx_full]
                    q, k, v = (
                        qkv.reshape(-1, K, 3, H, C // H)
                        .permute(2, 0, 3, 1, 4)
                        .unbind(dim=0)
                    )
                    if self.upcast_attention:
                        q = q.float()
                        k = k.float()
                    attn = (q * self.scale) @ k.transpose(-2, -1)
                    if self.upcast_softmax:
                        attn = attn.float()
                    attn = self.softmax(attn)
                    if isinstance(self.attn_drop, nn.Module):
                        attn = self.attn_drop(attn).to(qkv.dtype)
                    feat = (attn @ v).transpose(1, 2).reshape(-1, C)
                    out[idx_full.reshape(-1)] = feat.to(out.dtype)
                if full == int(piece.numel()):
                    continue
                idx = piece[full:]
                if idx.numel() > 0:
                    qkv = qkv_all[idx]
                    q, k, v = (
                        qkv.reshape(-1, 3, H, C // H).permute(1, 2, 0, 3).unbind(dim=0)
                    )
                    if self.upcast_attention:
                        q = q.float()
                        k = k.float()
                    attn = (q * self.scale) @ k.transpose(-2, -1)
                    if self.upcast_softmax:
                        attn = attn.float()
                    attn = self.softmax(attn)
                    if isinstance(self.attn_drop, nn.Module):
                        attn = self.attn_drop(attn).to(qkv.dtype)
                    feat = (attn @ v).transpose(0, 1).reshape(-1, C)
                    out[idx] = feat.to(out.dtype)
        return out

    def _forward_dilated(self, point):
        n_feat = int(point.feat.shape[0])
        n_ord = int(point.serialized_order.shape[1])
        if n_feat != n_ord:
            raise RuntimeError(
                f"attention: feat N={n_feat} != serialized N={n_ord} "
                "(feat N changed after spconv; check Point / grid_coord alignment)"
            )
        order = point.serialized_order[self.order_index]
        qkv_all = self.qkv(point.feat)
        branches = [
            self._standard_attention_branch(point, qkv_all)
            if int(d) <= 1
            else self._manual_attention_for_order(point, order, int(d), qkv_all)
            for d in self.dilations
        ]
        if len(branches) == 1:
            feat = branches[0]
        elif self.dilation_fusion == "concat_linear":
            feat = self.dilation_proj(torch.cat(branches, dim=1))
        elif self.dilation_fusion == "mean":
            feat = torch.stack(branches, dim=0).mean(dim=0)
        else:
            # Kept for ablation only; sum can amplify features as branch count grows.
            feat = torch.stack(branches, dim=0).sum(dim=0)
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point

    def forward(self, point):
        self.patch_size = min(
            int(offset2bincount(point.offset).min()),
            self.patch_size_max,
        )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        if len(self.dilations) > 1 or self.dilations != (1,):
            return self._forward_dilated(point)

        n_feat = int(point.feat.shape[0])
        n_ord = int(point.serialized_order.shape[1])
        if n_feat != n_ord:
            raise RuntimeError(
                f"attention: feat N={n_feat} != serialized N={n_ord} "
                "(feat N changed after spconv; check Point / grid_coord alignment)"
            )

        pad, unpad, _ = self.get_padding_and_inverse(point)
        padded_n = int(pad.shape[0])
        _check_gather_indices(pad, n_ord, "attention pad")
        order = point.serialized_order[self.order_index][pad]
        inv_map = point.serialized_inverse[self.order_index]
        _check_gather_indices(inv_map, int(unpad.shape[0]), "serialized_inverse")
        inverse = unpad[inv_map]
        _check_gather_indices(inverse, padded_n, "attention unpad")
        qkv = self.qkv(point.feat)[order]

        q, k, v = (
            qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        )
        if self.upcast_attention:
            q = q.float()
            k = k.float()
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.upcast_softmax:
            attn = attn.float()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class BackboneMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        upcast_attention=True,
        upcast_softmax=True,
        dilations=None,
        dilation_fusion="mean",
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
                algo=_SPCONV_ALGO,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )
        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            dilations=dilations,
            dilation_fusion=dilation_fusion,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            BackboneMLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = None
        self.act = None
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        code = point.serialized_code >> pooling_depth * 3
        _, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        point_dict = addict.Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        out = Point(point_dict)
        clear_point_attn_cache(out)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        out.sparsify()
        return out


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        _check_gather_indices(inverse, int(point.feat.shape[0]), "pooling_inverse")
        clear_point_attn_cache(parent)
        parent.feat = parent.feat + point.feat[inverse]
        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
                algo=_SPCONV_ALGO,
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        return self.stem(point)


class GSSBackbone(PointModule):

    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48,) * 5,
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48,) * 4,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        mamba_adapters=None,
        mamba_after_enc_stages: Optional[Iterable[str]] = None,
        dilated_attention: Optional[Mapping] = None,
        **_: Mapping,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        default_stage_dilations = {
            "stage0": [1],
            "stage1": [1],
            "stage2": [1],
            "stage3": [1, 2],
            "stage4": [1, 2, 4],
        }
        dilated_attention = dict(dilated_attention or {})
        stage_dilations = dict(dilated_attention.get("stage_dilations") or default_stage_dilations)
        dilation_fusion = str(dilated_attention.get("fusion", "mean"))

        assert self.num_stages == len(stride) + 1
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                dilations = stage_dilations.get(f"stage{s}", [1])
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        dilations=dilations,
                        dilation_fusion=dilation_fusion,
                    ),
                    name=f"block{i}",
                )
            if len(enc):
                self.enc.add(module=enc, name=f"enc{s}")

        self.dec = None
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_ch = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_ch[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_ch[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_ch[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

        if mamba_adapters is None:
            self.mamba_adapters = None
        elif isinstance(mamba_adapters, nn.ModuleDict):
            self.mamba_adapters = mamba_adapters
        else:
            self.mamba_adapters = nn.ModuleDict(mamba_adapters)
        self.mamba_after_enc_stages = set(mamba_after_enc_stages or ())

    def forward(self, data_dict_or_point: Mapping | Point) -> Point:
        point = (
            data_dict_or_point
            if isinstance(data_dict_or_point, Point)
            else Point(data_dict_or_point)
        )
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        point = self.embedding(point)
        for enc_name, enc_stage in self.enc.named_children():
            point = enc_stage(point)
            if (
                self.mamba_adapters is not None
                and enc_name in self.mamba_after_enc_stages
                and enc_name in self.mamba_adapters
            ):
                mod = self.mamba_adapters[enc_name]
                if mod is not None:
                    point.feat = mod(
                        point.feat,
                        point.coord,
                        batch=point.batch,
                        grid_coord=point.grid_coord,
                    )
                    if "sparse_conv_feat" in point.keys():
                        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(
                            point.feat
                        )
        if not self.enc_mode and self.dec is not None:
            point = self.dec(point)
        return point
