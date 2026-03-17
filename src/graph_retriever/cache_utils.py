import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GraphCacheSpec:
    method_name: str
    data_type: str
    data_mode: str
    fast_build: bool


def _safe_slug(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(s))


def csr_graph_cache_path(storage_dir_base: str, spec: GraphCacheSpec) -> str:
    """
    CSR NetworkX 圖 cache：用方法名做隔離，避免不同方法互相覆蓋。
    """
    base = _safe_slug(storage_dir_base)
    method = _safe_slug(spec.method_name)
    data_type = _safe_slug(spec.data_type)
    data_mode = _safe_slug(spec.data_mode)
    suffix = "_fast_test" if spec.fast_build else ""
    return f"{base}_{data_type}_{data_mode}_{method}{suffix}.pkl"


def load_pickle(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

