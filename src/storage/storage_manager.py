"""
Storage Manager 模組

統一管理所有 storage 路徑的生成邏輯
"""

import os
from typing import Literal, Optional


StorageType = Literal["vector_index", "graph_index", "lightrag", "csr_graph", "cache"]


def _safe_slug(s: str) -> str:
    """將字串轉成適合目錄／檔名的片段（與 graph_retriever.cache_utils 邏輯一致）。"""
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(s))


class StorageManager:
    """
    Storage 路徑管理器
    
    負責統一管理所有 storage 路徑的生成邏輯，確保不同模式、dataset、測試類型使用獨立的 storage
    """
    
    def __init__(self, root_dir: str = "/home/End_to_End_RAG/storage"):
        """
        初始化 StorageManager
        
        Args:
            root_dir: Storage 根目錄路徑
        """
        self.root_dir = root_dir
        self.storage_dirs = {
            "vector_index": os.path.join(root_dir, "vector_index"),
            "graph_index": os.path.join(root_dir, "graph_index"),
            "lightrag": os.path.join(root_dir, "lightrag"),
            "csr_graph": os.path.join(root_dir, "csr_graph"),
            "cache": os.path.join(root_dir, "cache"),
        }
        
        # 確保根目錄存在
        os.makedirs(self.root_dir, exist_ok=True)
        for storage_dir in self.storage_dirs.values():
            os.makedirs(storage_dir, exist_ok=True)
    
    def get_storage_path(
        self,
        storage_type: StorageType,
        data_type: str,
        method: str,
        fast_test: bool = False,
        custom_tag: str = "",
        mode: str = "",
        top_k: int = 0
    ) -> str:
        """
        取得 storage 路徑
        
        Args:
            storage_type: Storage 類型（vector_index, graph_index, lightrag, csr_graph, cache）
            data_type: 資料類型（DI, GEN）
            method: 方法名稱（hybrid, propertyindex, lightrag_default 等）
            fast_test: 是否為快速測試模式
            custom_tag: 自訂標籤（用於額外區分不同實驗）
            mode: 檢索模式（LightRAG: local/global/hybrid/mix 等）
            top_k: 檢索數量（用於區分不同 top_k 設定）
        
        Returns:
            完整的 storage 路徑
        
        Examples:
            >>> manager = StorageManager()
            >>> manager.get_storage_path("vector_index", "DI", "hybrid")
            '/home/End_to_End_RAG/storage/vector_index/DI_hybrid'
            >>> manager.get_storage_path("lightrag", "DI", "lightrag_default", mode="hybrid")
            '/home/End_to_End_RAG/storage/lightrag/DI_lightrag_default_hybrid'
            >>> manager.get_storage_path("vector_index", "DI", "hybrid", top_k=5)
            '/home/End_to_End_RAG/storage/vector_index/DI_hybrid_k5'
        """
        if storage_type not in self.storage_dirs:
            raise ValueError(f"不支援的 storage 類型: {storage_type}")
        
        base_dir = self.storage_dirs[storage_type]
        
        # 建立路徑名稱
        path_parts = [data_type, method]
        
        # 加入 mode（如果提供）
        if mode:
            path_parts.append(mode)
        
        # 加入 top_k（如果提供且 > 0）
        if top_k > 0:
            path_parts.append(f"k{top_k}")
        
        # 加入自訂標籤
        if custom_tag:
            path_parts.append(custom_tag)
        
        # 加入 fast_test 標記
        if fast_test:
            path_parts.append("fast_test")
        
        path_name = "_".join(path_parts)
        
        # CSR Graph 特殊處理（使用 .pkl 檔案而非目錄）
        if storage_type == "csr_graph":
            full_path = os.path.join(base_dir, f"{path_name}.pkl")
        else:
            full_path = os.path.join(base_dir, path_name)
            os.makedirs(full_path, exist_ok=True)
        
        return full_path
    
    def get_csr_graph_path(
        self,
        data_type: str,
        data_mode: str,
        method: str,
        fast_test: bool = False
    ) -> str:
        """
        取得 CSR Graph cache 路徑（向後相容）
        
        Args:
            data_type: 資料類型（DI, GEN）
            data_mode: 資料模式（natural_text, markdown 等）
            method: 方法名稱（khop, bridge 等）
            fast_test: 是否為快速測試
        
        Returns:
            完整的 CSR Graph cache 檔案路徑
        """
        csr_dir = self.storage_dirs["csr_graph"]
        
        path_parts = [data_type, data_mode, method]
        if fast_test:
            path_parts.append("fast_test")
        
        filename = "_".join(path_parts) + ".pkl"
        return os.path.join(csr_dir, filename)

    def get_autoschema_output_dir(
        self,
        data_type: str,
        data_mode: str = "",
        sup: str = "",
        fast_test: bool = False,
    ) -> str:
        """
        AutoSchemaKG 建圖輸出目錄：{root}/autoschema/{slug}

        slug 由 data_type、可選的 data_mode、sup、fast_test 組成，以區分不同資料集與實驗。
        """
        parts = [_safe_slug(data_type or "DI")]
        if data_mode:
            parts.append(_safe_slug(data_mode))
        if sup:
            parts.append(_safe_slug(sup))
        if fast_test:
            parts.append("fast_test")
        slug = "_".join(parts)
        full_path = os.path.join(self.root_dir, "autoschema", slug)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def migrate_old_storage(self, old_paths: dict) -> dict:
        """
        遷移舊的 storage 到新結構
        
        Args:
            old_paths: 舊路徑字典，格式 {"type": "old_path"}
        
        Returns:
            遷移結果字典，格式 {"old_path": "new_path"}
        """
        migration_map = {}
        
        # 此方法由 migrate_storage.py 腳本實作
        # 這裡僅提供介面
        
        return migration_map


# 全域單例
_storage_manager = None


def get_storage_path(
    storage_type: StorageType,
    data_type: str,
    method: str,
    fast_test: bool = False,
    custom_tag: str = "",
    mode: str = "",
    top_k: int = 0,
    root_dir: Optional[str] = None
) -> str:
    """
    取得 storage 路徑的便捷函數
    
    Args:
        storage_type: Storage 類型
        data_type: 資料類型
        method: 方法名稱
        fast_test: 是否為快速測試
        custom_tag: 自訂標籤
        mode: 檢索模式（LightRAG: local/global/hybrid/mix 等）
        top_k: 檢索數量（用於區分不同 top_k 設定）
        root_dir: Storage 根目錄（None 則使用預設值）
    
    Returns:
        完整的 storage 路徑
    """
    global _storage_manager
    
    if _storage_manager is None or (root_dir and root_dir != _storage_manager.root_dir):
        _storage_manager = StorageManager(root_dir or "/home/End_to_End_RAG/storage")
    
    return _storage_manager.get_storage_path(
        storage_type=storage_type,
        data_type=data_type,
        method=method,
        fast_test=fast_test,
        custom_tag=custom_tag,
        mode=mode,
        top_k=top_k
    )


def get_csr_graph_path(
    data_type: str,
    data_mode: str,
    method: str,
    fast_test: bool = False,
    root_dir: Optional[str] = None
) -> str:
    """
    取得 CSR Graph cache 路徑的便捷函數
    
    Args:
        data_type: 資料類型
        data_mode: 資料模式
        method: 方法名稱
        fast_test: 是否為快速測試
        root_dir: Storage 根目錄
    
    Returns:
        CSR Graph cache 檔案路徑
    """
    global _storage_manager
    
    if _storage_manager is None or (root_dir and root_dir != _storage_manager.root_dir):
        _storage_manager = StorageManager(root_dir or "/home/End_to_End_RAG/storage")
    
    return _storage_manager.get_csr_graph_path(
        data_type=data_type,
        data_mode=data_mode,
        method=method,
        fast_test=fast_test
    )


def get_autoschema_output_dir(
    data_type: str,
    data_mode: str = "",
    sup: str = "",
    fast_test: bool = False,
    root_dir: Optional[str] = None,
) -> str:
    """
    取得 AutoSchemaKG 輸出目錄（並建立目錄）。

    Args:
        data_type: 資料集類型（如 DI、GEN）
        data_mode: 資料模式（如 natural_text）；空字串則不納入 slug
        sup: 實驗用附加標籤；空字串則不納入 slug
        fast_test: 快速測試時於 slug 尾端加上 fast_test
        root_dir: Storage 根目錄；None 則使用預設
    """
    global _storage_manager

    if _storage_manager is None or (root_dir and root_dir != _storage_manager.root_dir):
        _storage_manager = StorageManager(root_dir or "/home/End_to_End_RAG/storage")

    return _storage_manager.get_autoschema_output_dir(
        data_type=data_type,
        data_mode=data_mode,
        sup=sup,
        fast_test=fast_test,
    )
