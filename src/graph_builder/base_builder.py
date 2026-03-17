# graph_builder/base_builder.py
from abc import ABC, abstractmethod
from typing import List, Any
from llama_index.core import Document

class BaseGraphBuilder(ABC):
    def __init__(self, graph_store: Any, settings: Any):
        """
        使用依賴注入 (Dependency Injection) 將共用資源傳入 Plugin
        """
        self.graph_store = graph_store
        self.settings = settings

    @abstractmethod
    def build(self, documents: List[Document]):
        """所有建圖插件都必須實作此方法"""
        pass