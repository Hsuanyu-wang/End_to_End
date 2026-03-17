"""
RAG Pipeline 基底類別

定義所有 RAG Pipeline 的共用介面
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseRAGPipeline(ABC):
    """
    RAG Pipeline 抽象基底類別
    
    定義所有 RAG Pipeline 必須實作的介面
    
    Attributes:
        name: Pipeline 名稱
    """
    
    def __init__(self, name: str):
        """
        初始化 RAG Pipeline
        
        Args:
            name: Pipeline 名稱
        """
        self.name = name
    
    @abstractmethod
    async def aquery(self, query: str) -> Any:
        """
        非同步查詢方法
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果
        """
        pass
    
    @abstractmethod
    def query(self, query: str) -> Any:
        """
        同步查詢方法
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
