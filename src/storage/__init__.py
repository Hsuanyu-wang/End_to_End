"""
Storage 管理模組

提供統一的 storage 路徑管理機制
"""

from .storage_manager import (
    StorageManager,
    get_storage_path,
    get_csr_graph_path,
    get_autoschema_output_dir,
)

__all__ = [
    "StorageManager",
    "get_storage_path",
    "get_csr_graph_path",
    "get_autoschema_output_dir",
]
