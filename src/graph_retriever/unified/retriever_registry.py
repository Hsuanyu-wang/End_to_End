"""
Graph Retriever 註冊器

提供 Plugin Registry System 讓 Retriever 可動態註冊和發現
"""

from typing import Dict, Type, List
from src.graph_retriever.base_retriever import BaseGraphRetriever


class GraphRetrieverRegistry:
    """
    Graph Retriever 註冊器
    
    支援動態註冊和發現不同的檢索方法
    """
    
    _retrievers: Dict[str, Type[BaseGraphRetriever]] = {}
    
    @classmethod
    def register(cls, name: str, retriever_class: Type[BaseGraphRetriever]):
        """
        註冊 retriever
        
        Args:
            name: Retriever 名稱（用於識別）
            retriever_class: Retriever 類別（須繼承 BaseGraphRetriever）
        """
        if not issubclass(retriever_class, BaseGraphRetriever):
            raise ValueError(f"Retriever 類別必須繼承 BaseGraphRetriever，收到: {retriever_class}")
        
        cls._retrievers[name] = retriever_class
        print(f"✅ 註冊 Retriever: {name} -> {retriever_class.__name__}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseGraphRetriever:
        """
        根據名稱建立 retriever 實例
        
        Args:
            name: Retriever 名稱
            **kwargs: Retriever 初始化參數
        
        Returns:
            Retriever 實例
        """
        if name not in cls._retrievers:
            available = list(cls._retrievers.keys())
            raise ValueError(f"未知的 Retriever: {name}。可用: {available}")
        
        retriever_class = cls._retrievers[name]
        return retriever_class(**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的 retriever"""
        return list(cls._retrievers.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """檢查 retriever 是否已註冊"""
        return name in cls._retrievers


# 自動註冊內建 retrievers
def _register_builtin_retrievers():
    """註冊內建的 retrievers"""
    try:
        from src.graph_retriever.lightrag_retriever import LightRAGRetriever
        GraphRetrieverRegistry.register("lightrag", LightRAGRetriever)
    except ImportError:
        print("⚠️  LightRAG Retriever 註冊失敗（可能缺少依賴）")

    try:
        from src.graph_retriever.lightrag_graph_retriever import LightRAGGraphRetriever

        # 以不同預設 strategy 註冊多個名稱，方便 config / CLI 直接使用
        GraphRetrieverRegistry.register("lightrag_ppr", LightRAGGraphRetriever)
        GraphRetrieverRegistry.register("lightrag_pcst", LightRAGGraphRetriever)
        GraphRetrieverRegistry.register("lightrag_tog", LightRAGGraphRetriever)
        GraphRetrieverRegistry.register("lightrag_k_hop", LightRAGGraphRetriever)
        GraphRetrieverRegistry.register("lightrag_anchor_hybrid_khop", LightRAGGraphRetriever)
    except ImportError:
        print("⚠️  LightRAGGraphRetriever 註冊失敗（可能缺少依賴）")


# 初始化時自動註冊
_register_builtin_retrievers()
