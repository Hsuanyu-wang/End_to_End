import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional
from src.storage import get_csr_graph_path


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
    CSR NetworkX 圖 cache：使用 StorageManager 統一管理路徑
    """
    # 使用 StorageManager 取得路徑
    return get_csr_graph_path(
        data_type=spec.data_type,
        data_mode=spec.data_mode,
        method=spec.method_name,
        fast_test=spec.fast_build
    )


def load_pickle(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

