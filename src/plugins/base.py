"""
插件基礎介面

定義 Knowledge Graph 增強插件的標準介面
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseKGPlugin(ABC):
    """
    Knowledge Graph 增強插件基類
    
    所有 KG 插件都應該繼承此類並實作必要的方法
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        取得插件名稱
        
        Returns:
            插件名稱
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        取得插件描述
        
        Returns:
            插件功能描述
        """
        pass
    
    def enhance_schema(
        self,
        text_corpus: List[str],
        base_schema: List[str],
        **kwargs
    ) -> List[str]:
        """
        增強 Schema 定義
        
        Args:
            text_corpus: 文本語料庫
            base_schema: 基礎 Schema（實體類型列表）
            **kwargs: 額外參數
        
        Returns:
            增強後的 Schema
        """
        # 預設實作：不修改 Schema
        return base_schema
    
    def enhance_extraction(
        self,
        text: str,
        base_triples: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        增強三元組提取
        
        Args:
            text: 輸入文本
            base_triples: 基礎提取的三元組列表
            **kwargs: 額外參數
        
        Returns:
            增強後的三元組列表
        """
        # 預設實作：不修改三元組
        return base_triples
    
    def post_process_graph(
        self,
        graph_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        後處理圖譜資料
        
        Args:
            graph_data: 圖譜資料
            **kwargs: 額外參數
        
        Returns:
            後處理後的圖譜資料
        """
        # 預設實作：不修改圖譜
        return graph_data
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化插件
        
        Args:
            config: 插件配置
        """
        pass
    
    def cleanup(self):
        """清理插件資源"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}')"
