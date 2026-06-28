"""Helpers for moving the nested-dict scene representation across devices."""
from typing import Any

import torch


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move every tensor in a (possibly nested) dict/list to ``device``.
    Non-tensor leaves (ints, strings, id lists) are left untouched."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj
