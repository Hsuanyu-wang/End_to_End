"""
RAG Wrapper 基底類別

統一所有 RAG Pipeline 封裝器的介面，消除重複代碼
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseRAGWrapper(ABC):
    """
    RAG Wrapper 抽象基底類別
    
    統一的 Wrapper 介面，提供：
    - 時間計算
    - 錯誤處理
    - 標準化輸出格式
    
    所有 Wrapper 都應繼承此類別並實作 _execute_query 方法
    
    Attributes:
        name: Wrapper 名稱
    """
    
    def __init__(self, name: str):
        """
        初始化 Wrapper
        
        Args:
            name: Wrapper 名稱
        """
        self.name = name
    
    @abstractmethod
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        執行實際查詢邏輯（子類別實作）
        
        Args:
            query: 使用者查詢
        
        Returns:
            包含以下鍵值的字典：
                - generated_answer: 生成的答案
                - retrieved_contexts: 檢索到的上下文列表
                - retrieved_ids: 檢索到的文件 ID 列表
                - source_nodes: 原始節點列表
        """
        pass
    
    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        處理查詢錯誤
        
        Args:
            error: 錯誤物件
        
        Returns:
            錯誤結果字典
        """
        print(f"❌ {self.name} 查詢發生錯誤: {error}")
        
        return {
            "generated_answer": f"Error: {error}",
            "retrieved_contexts": [],
            "retrieved_ids": [],
            "source_nodes": [],
        }
    
    async def aquery_and_log(self, user_query: str) -> Dict[str, Any]:
        """
        非同步查詢並記錄執行時間
        
        Args:
            user_query: 使用者查詢
        
        Returns:
            包含查詢結果與執行時間的字典
        """
        start_time = time.time()
        
        try:
            result = await self._execute_query(user_query)
        except Exception as e:
            result = self._handle_error(e)
        
        end_time = time.time()
        result["execution_time_sec"] = round(end_time - start_time, 4)
        
        return result
    
    def query_and_log(self, user_query: str) -> Dict[str, Any]:
        """
        同步查詢並記錄執行時間（預設實作）
        
        Args:
            user_query: 使用者查詢
        
        Returns:
            包含查詢結果與執行時間的字典
        
        Note:
            子類別可以覆寫此方法以提供同步實作
        """
        raise NotImplementedError(f"{self.name} 不支援同步查詢，請使用 aquery_and_log")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
