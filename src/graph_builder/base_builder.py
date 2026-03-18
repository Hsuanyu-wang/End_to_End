# graph_builder/base_builder.py
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from llama_index.core import Document


class BaseGraphBuilder(ABC):
    """
    Graph Builder 基類
    
    定義 Graph Building 階段的統一介面
    """
    
    def __init__(self, graph_store: Any = None, settings: Any = None):
        """
        使用依賴注入 (Dependency Injection) 將共用資源傳入 Builder
        
        Args:
            graph_store: 圖譜儲存後端(可選)
            settings: 配置設定(可選)
        """
        self.graph_store = graph_store
        self.settings = settings

    @abstractmethod
    def build(self, documents: List[Document]) -> Dict[str, Any]:
        """
        所有建圖插件都必須實作此方法
        
        Args:
            documents: 文檔列表
            
        Returns:
            標準化的圖譜資料字典,包含:
            - nodes: List[Dict] - 節點列表
            - edges: List[Dict] - 邊列表
            - metadata: Dict - 元資訊
            - schema_info: Dict - Schema 資訊(entities, relations)
            - storage_path: str - 儲存路徑(可選)
            - graph_format: str - 圖譜格式標識
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        取得 Builder 名稱
        
        Returns:
            Builder 名稱
        """
        pass
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Builder
        
        Args:
            config: 配置參數
        """
        pass
    
    def cleanup(self):
        """清理 Builder 資源"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}')"
