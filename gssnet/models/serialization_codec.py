import torch

from gssnet.models.serialization.z_order import key2xyz, xyz2key
from gssnet.models.serialization import hilbert as hilbert_impl


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x = grid_coord[:, 0].long()
    y = grid_coord[:, 1].long()
    z = grid_coord[:, 2].long()
    return xyz2key(x, y, z, b=None, depth=depth)


def z_order_decode(code: torch.Tensor, depth):
    x, y, z, _unused = key2xyz(code, depth=depth)
    del _unused  # unused batch residue from auxiliary MSB packing
    return torch.stack([x, y, z], dim=-1)


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_impl.encode(grid_coord.long(), num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_impl.decode(code, num_dims=3, num_bits=depth)
