"""
LightRAG Wrapper

封裝 LightRAG 框架的 RAG Pipeline
"""

import inspect
from typing import Dict, Any
from .base_wrapper import BaseRAGWrapper
from src.config.settings import get_settings


class LightRAGWrapper(BaseRAGWrapper):
    """
    LightRAG 封裝器
    
    支援 LightRAG 的多種檢索模式：
    - local: 關注特定實體與細節
    - global: 關注整體趨勢與總結
    - hybrid: 混合模式
    - mix: 知識圖譜 + 向量檢索
    - naive: 僅向量檢索
    - bypass: 直接查詢 LLM
    
    Attributes:
        name: Wrapper 名稱
        rag: LightRAG 實例
        mode: 檢索模式
        use_context: 是否使用 LightRAG 的 context（新版模式）
        _initialized: 是否已初始化（用於新版 LightRAG）
    """
    
    def __init__(
        self,
        name: str,
        rag_instance,
        mode: str = "hybrid",
        use_context: bool = True,
        model_type: str = "small"
    ):
        """
        初始化 LightRAG Wrapper
        
        Args:
            name: Wrapper 名稱
            rag_instance: LightRAG 實例
            mode: 檢索模式
            use_context: 是否使用 context 進行自訂生成（False 則使用 LightRAG 原生）
            model_type: 模型類型（用於取得 Settings）
        """
        super().__init__(name)
        self.rag = rag_instance
        self.mode = mode
        self.use_context = use_context
        self.model_type = model_type
        self._initialized = False
    
    def query(self, question: str) -> str:
        """
        同步查詢（僅用於簡單呼叫）
        
        Args:
            question: 使用者問題
        
        Returns:
            LightRAG 回應
        """
        from lightrag.lightrag import QueryParam
        return self.rag.query(question, param=QueryParam(mode=self.mode))
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        執行 LightRAG 查詢
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果字典
        """
        from lightrag.lightrag import QueryParam
        
        Settings = get_settings(model_type=self.model_type)
        
        # 解決 LightRAG 新版本的初始化問題
        if not self._initialized and hasattr(self.rag, "initialize_storages"):
            await self.rag.initialize_storages()
            self._initialized = True
        
        # 不使用自訂 context（原生模式）
        if not self.use_context:
            response = await self._execute_original_mode(query)
            return {
                "generated_answer": str(response) if response else "找不到答案",
                "retrieved_contexts": [],
                "retrieved_ids": [],
                "source_nodes": [],
            }
        
        # 使用自訂 context 模式（新版）
        response = await self._execute_context_mode(query, Settings)
        return response
    
    async def _execute_original_mode(self, query: str) -> str:
        """
        執行原生 LightRAG 模式（不使用自訂 context）
        
        Args:
            query: 使用者查詢
        
        Returns:
            LightRAG 回應
        """
        from lightrag.lightrag import QueryParam
        
        if hasattr(self.rag, "aquery"):
            response = await self.rag.aquery(query, param=QueryParam(mode=self.mode))
        elif inspect.iscoroutinefunction(self.rag.query):
            response = await self.rag.query(query, param=QueryParam(mode=self.mode))
        else:
            response = self.rag.query(query, param=QueryParam(mode=self.mode))
        
        # 相容處理：如果回傳的是 LlamaIndex 物件才取 .text
        if hasattr(response, "text"):
            response = response.text
        elif not isinstance(response, str):
            response = str(response)
        
        return response
    
    async def _execute_context_mode(self, query: str, Settings) -> Dict[str, Any]:
        """
        執行自訂 context 模式（使用 LightRAG context + 自訂 prompt）
        
        Args:
            query: 使用者查詢
            Settings: 模型設定物件
        
        Returns:
            查詢結果字典
        """
        from lightrag.lightrag import QueryParam
        from lightrag.prompt import PROMPTS
        
        # 取得 LightRAG context
        if hasattr(self.rag, "aquery"):
            raw_context = await self.rag.aquery_data(
                query,
                param=QueryParam(mode=self.mode, only_need_context=True)
            )
        elif inspect.iscoroutinefunction(self.rag.query):
            raw_context = self.rag.query_data(
                query,
                param=QueryParam(mode=self.mode, only_need_context=True)
            )
        else:
            raw_context = self.rag.query_data(
                query,
                param=QueryParam(mode=self.mode, only_need_context=True)
            )
        
        print(f"RAW_CONTEXT:\n\n{raw_context}\n\n")
        
        retrieved_contexts = [str(raw_context)] if raw_context else []
        context_str = "\n".join(retrieved_contexts) if retrieved_contexts else ""
        
        # 組裝 Prompt
        prompt = PROMPTS["rag_response"].format(
            context_data=context_str,
            response_type="Multiple Paragraphs",
            user_prompt=query,
            query=query,
            retrieved_contexts=context_str
        )
        
        # 使用 LLM 生成答案
        llm_response = await Settings.llm.acomplete(prompt)
        response = llm_response.text
        
        return {
            "generated_answer": str(response) if response else "找不到答案",
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": [],
            "source_nodes": [],
        }


# 向後兼容別名
class LightRAGWrapper_Original(LightRAGWrapper):
    """向後兼容：原生模式的 LightRAG Wrapper"""
    
    def __init__(self, name: str, rag_instance, mode: str = "hybrid"):
        super().__init__(
            name=name,
            rag_instance=rag_instance,
            mode=mode,
            use_context=False  # 使用原生模式
        )
