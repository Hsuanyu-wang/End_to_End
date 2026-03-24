"""
GraphRetriever 基類

定義 Graph Retrieval 階段的統一介面
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from llama_index.core.schema import NodeWithScore


class BaseGraphRetriever(ABC):
    """
    Graph Retriever 基類
    
    所有 Graph Retriever 都應該繼承此類並實作必要的方法
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        取得 Retriever 名稱
        
        Returns:
            Retriever 名稱
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        從圖譜中檢索相關內容
        
        Args:
            query: 使用者查詢
            graph_data: 圖譜資料(可選,某些 Retriever 可能已內建圖譜)
            top_k: 檢索數量
            **kwargs: 額外參數
        
        Returns:
            檢索結果字典,包含:
            - contexts: List[str] - 檢索到的文本內容
            - nodes: List[NodeWithScore] - 檢索到的節點(可選)
            - metadata: Dict[str, Any] - 額外元資訊
        """
        pass
    
    async def aretrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        非同步檢索（預設 fallback 到同步 retrieve）

        子類別可覆寫此方法以提供真正的非同步實作。
        """
        return self.retrieve(query, graph_data=graph_data, top_k=top_k, **kwargs)

    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Retriever
        
        Args:
            config: 配置參數
        """
        pass
    
    def cleanup(self):
        """清理 Retriever 資源"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}')"


class GraphData:
    """
    標準化的圖譜資料結構
    
    用於在 Builder 和 Retriever 之間傳遞圖譜資訊
    """
    
    def __init__(
        self,
        nodes: List[Dict[str, Any]] = None,
        edges: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
        schema_info: Dict[str, Any] = None,
        storage_path: str = None,
        graph_format: str = "custom"
    ):
        """
        初始化圖譜資料
        
        Args:
            nodes: 節點列表
            edges: 邊列表
            metadata: 元資訊
            schema_info: Schema 資訊(實體類型、關係類型等)
            storage_path: 儲存路徑(用於持久化的圖譜)
            graph_format: 圖譜格式("custom", "networkx", "neo4j", "lightrag")
        """
        self.nodes = nodes or []
        self.edges = edges or []
        self.metadata = metadata or {}
        self.schema_info = schema_info or {}
        self.storage_path = storage_path
        self.graph_format = graph_format
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": self.metadata,
            "schema_info": self.schema_info,
            "storage_path": self.storage_path,
            "graph_format": self.graph_format
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphData":
        """從字典建立 GraphData 實例"""
        return cls(**data)
