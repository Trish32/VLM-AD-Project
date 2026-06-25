# List helpers (ported verbatim from official QCNet utils/list.py).
from typing import Any, List, Optional


def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None
