"""
Vector RAG Wrapper

封裝基於向量檢索的 RAG Pipeline
"""

import time
from typing import Dict, Any
from .base_wrapper import BaseRAGWrapper


class VectorRAGWrapper(BaseRAGWrapper):
    """
    Vector RAG 封裝器
    
    用於封裝 LlamaIndex 的向量查詢引擎，提供標準化介面
    
    Attributes:
        name: Wrapper 名稱
        query_engine: LlamaIndex 查詢引擎實例
    """
    
    def __init__(self, name: str, query_engine, schema_info: Dict[str, Any] = None):
        """
        初始化 Vector RAG Wrapper
        
        Args:
            name: Wrapper 名稱
            query_engine: LlamaIndex 查詢引擎（支援 aquery 方法）
            schema_info: Schema 資訊字典（可選）
        """
        super().__init__(name, schema_info=schema_info)
        self.query_engine = query_engine
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        執行向量檢索與生成
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果字典
        """
        response = await self.query_engine.aquery(query)
        
        # 提取生成的答案
        generated_answer = str(response)
        
        # 提取檢索到的 Context 與 Document IDs
        retrieved_contexts = []
        retrieved_ids = []
        
        for node in response.source_nodes:
            retrieved_contexts.append(node.get_content())
            doc_id = node.metadata.get("NO", None)
            retrieved_ids.append(str(doc_id))
        
        # 應用 retrieval token 限制
        retrieved_contexts = self._truncate_contexts_by_tokens(retrieved_contexts)
        
        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": retrieved_ids,
            "source_nodes": response.source_nodes,
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
        
        response = self.query_engine.query(user_query)
        
        end_time = time.time()
        execution_time_sec = round(end_time - start_time, 4)
        
        # 提取生成的答案
        generated_answer = str(response)
        
        # 提取檢索到的 Context 與 Document IDs
        retrieved_contexts = []
        retrieved_ids = []
        
        for node in response.source_nodes:
            retrieved_contexts.append(node.get_content())
            doc_id = node.metadata.get("id", node.node_id)
            retrieved_ids.append(str(doc_id))
        
        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": retrieved_ids,
            "source_nodes": response.source_nodes,
            "execution_time_sec": execution_time_sec
        }


# 向後兼容別名
RAGPipelineWrapper = VectorRAGWrapper
