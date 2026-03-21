"""
檢索組件抽象基類

定義了檢索器的統一介面，所有檢索實作都應繼承此基類。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple

# 從頂層 formats 模組導入
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from formats import GraphData


class RetrievalResult:
    """
    檢索結果
    
    Attributes:
        contexts: 檢索到的上下文列表
        scores: 對應的相關性分數
        metadata: 額外的元資料（如檢索路徑、耗時等）
    """
    
    def __init__(
        self,
        contexts: List[str],
        scores: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.contexts = contexts
        self.scores = scores or [1.0] * len(contexts)
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "contexts": self.contexts,
            "scores": self.scores,
            "metadata": self.metadata,
        }


class BaseRetriever(ABC):
    """
    檢索器抽象基類
    
    所有檢索組件（LightRAG、CSR、ToG、Adaptive）都應實作此介面。
    
    關鍵設計原則：
    1. 輸入：查詢 + 圖譜資料
    2. 輸出：相關上下文 + 分數
    3. 支援多種檢索策略
    
    Examples:
        >>> class LightRAGRetriever(BaseRetriever):
        ...     def retrieve(self, query, graph_data, top_k=5, mode="hybrid"):
        ...         # LightRAG 的檢索邏輯
        ...         return RetrievalResult(...)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化檢索器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._initialize()
    
    def _initialize(self):
        """子類可覆寫此方法進行初始化"""
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        graph_data: GraphData,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """
        從圖譜中檢索相關資訊
        
        這是核心抽象方法，所有子類必須實作。
        
        Args:
            query: 查詢文本
            graph_data: 圖譜資料
            top_k: 返回的上下文數量
            **kwargs: 額外參數（如 mode、k_hop、threshold 等）
        
        Returns:
            RetrievalResult: 檢索結果
        
        Notes:
            - 應支援不同的檢索模式（如 local、global、hybrid）
            - 可以使用圖譜結構進行遍歷
            - 返回的上下文應該是有用的文本片段
        """
        pass
    
    def retrieve_batch(
        self,
        queries: List[str],
        graph_data: GraphData,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        批次檢索
        
        預設實作為逐一處理，子類可覆寫以提供更高效的批次處理。
        
        Args:
            queries: 查詢列表
            graph_data: 圖譜資料
            top_k: 每個查詢返回的上下文數量
            **kwargs: 額外參數
        
        Returns:
            List[RetrievalResult]: 檢索結果列表
        """
        results = []
        for query in queries:
            result = self.retrieve(query, graph_data, top_k, **kwargs)
            results.append(result)
        return results
    
    def get_name(self) -> str:
        """
        獲取檢索器名稱
        
        Returns:
            str: 檢索器名稱
        """
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """獲取當前配置"""
        return self.config.copy()
    
    def supports_mode(self, mode: str) -> bool:
        """
        檢查是否支援特定檢索模式
        
        Args:
            mode: 檢索模式（如 "local", "global", "hybrid"）
        
        Returns:
            bool: 是否支援
        """
        # 預設實作：假設支援所有模式
        # 子類應覆寫以提供準確的資訊
        return True
    
    def get_supported_modes(self) -> List[str]:
        """
        獲取支援的檢索模式列表
        
        Returns:
            List[str]: 支援的模式
        """
        # 預設實作：返回常見模式
        # 子類應覆寫以提供準確的資訊
        return ["default"]
    
    def preprocess_query(self, query: str) -> str:
        """
        預處理查詢
        
        Args:
            query: 原始查詢
        
        Returns:
            str: 處理後的查詢
        """
        # 預設實作：基本清理
        return query.strip()
    
    def postprocess_contexts(self, contexts: List[str]) -> List[str]:
        """
        後處理檢索到的上下文
        
        Args:
            contexts: 原始上下文列表
        
        Returns:
            List[str]: 處理後的上下文
        """
        # 預設實作：去重和清理
        seen = set()
        result = []
        for context in contexts:
            context = context.strip()
            if context and context not in seen:
                result.append(context)
                seen.add(context)
        return result
    
    def rank_contexts(
        self,
        query: str,
        contexts: List[str],
        scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        """
        重排檢索到的上下文
        
        Args:
            query: 查詢
            contexts: 上下文列表
            scores: 分數列表
        
        Returns:
            Tuple[List[str], List[float]]: 排序後的上下文和分數
        """
        # 預設實作：按分數降序排序
        paired = list(zip(contexts, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        sorted_contexts, sorted_scores = zip(*paired) if paired else ([], [])
        return list(sorted_contexts), list(sorted_scores)
