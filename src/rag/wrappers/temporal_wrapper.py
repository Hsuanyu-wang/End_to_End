"""
Temporal LightRAG Wrapper

封裝帶有時間權重的 LightRAG Pipeline
"""

import time
from typing import Dict, Any
from .base_wrapper import BaseRAGWrapper


class TemporalLightRAGWrapper(BaseRAGWrapper):
    """
    Temporal LightRAG 封裝器
    
    封裝帶有時間權重的 LightRAG，檢索結果會根據時序排序
    
    Attributes:
        name: Wrapper 名稱
        rag: TemporalLightRAG 實例
        mode: 檢索模式
        time_weighting: 是否啟用時間權重
    """
    
    def __init__(
        self,
        name: str,
        rag_instance,
        mode: str = "hybrid",
        time_weighting: bool = True,
        schema_info: Dict[str, Any] = None
    ):
        """
        初始化 Temporal LightRAG Wrapper
        
        Args:
            name: Wrapper 名稱
            rag_instance: TemporalLightRAG 實例
            mode: 檢索模式
            time_weighting: 是否啟用時間權重
            schema_info: Schema 資訊字典
        """
        super().__init__(name, schema_info=schema_info)
        self.rag = rag_instance
        self.mode = mode
        self.time_weighting = time_weighting
    
    def _extract_contexts_from_response(self, response) -> list:
        """從 TemporalLightRAG 回應中提取 contexts"""
        response_str = str(response)
        if not response_str or response_str == "None":
            return []
        return [response_str]
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        執行 Temporal LightRAG 查詢
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果字典
        """
        response = self.rag.query(
            query,
            mode=self.mode,
            time_weighting=self.time_weighting
        )
        
        response_str = str(response) if response else ""
        contexts = self._extract_contexts_from_response(response)
        contexts = self._truncate_contexts_by_tokens(contexts)
        
        return {
            "generated_answer": response_str,
            "retrieved_contexts": contexts,
            "retrieved_ids": [],
            "source_nodes": [],
        }
    
    def query_and_log(self, user_query: str) -> Dict[str, Any]:
        """
        同步查詢並記錄執行時間
        
        Args:
            user_query: 使用者查詢
        
        Returns:
            包含查詢結果與執行時間的字典
        """
        start_time = time.time()
        
        response = self.rag.query(
            user_query,
            mode=self.mode,
            time_weighting=self.time_weighting
        )
        
        end_time = time.time()
        
        response_str = str(response) if response else ""
        contexts = self._extract_contexts_from_response(response)
        
        return {
            "generated_answer": response_str,
            "retrieved_contexts": contexts,
            "retrieved_ids": [],
            "source_nodes": [],
            "execution_time_sec": round(end_time - start_time, 4)
        }


# 向後兼容別名
TemporalWrapper = TemporalLightRAGWrapper
