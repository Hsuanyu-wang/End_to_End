"""
RAG Wrapper 基底類別

統一所有 RAG Pipeline 封裝器的介面，消除重複代碼
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.utils.token_counter import get_token_counter


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
        schema_info: Schema 資訊（可選）
    """
    
    def __init__(self, name: str, schema_info: Dict[str, Any] = None):
        """
        初始化 Wrapper
        
        Args:
            name: Wrapper 名稱
            schema_info: Schema 資訊字典，包含 method、entities、relations、validation_schema
        """
        self.name = name
        self.schema_info = schema_info or {}
        self.retrieval_max_tokens = 0  # 0 表示不限制
    
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
    
    def set_retrieval_max_tokens(self, max_tokens: int) -> None:
        """
        設定檢索內容的最大 token 數
        
        Args:
            max_tokens: 最大 token 數（0 表示不限制）
        """
        self.retrieval_max_tokens = max(0, int(max_tokens or 0))
        if self.retrieval_max_tokens > 0:
            print(f"🎯 [{self.name}] 設定 retrieval_max_tokens={self.retrieval_max_tokens}")
    
    def _truncate_contexts_by_tokens(self, contexts: List[str]) -> List[str]:
        """
        根據 token 數量截斷檢索到的 contexts
        
        Args:
            contexts: 原始 context 列表
        
        Returns:
            截斷後的 context 列表
        """
        if not self.retrieval_max_tokens or self.retrieval_max_tokens <= 0:
            return contexts
        
        if not contexts:
            return contexts
        
        token_counter = get_token_counter()
        truncated_contexts = []
        total_tokens = 0
        
        for ctx in contexts:
            ctx_tokens = token_counter.count_tokens(ctx)
            if total_tokens + ctx_tokens <= self.retrieval_max_tokens:
                truncated_contexts.append(ctx)
                total_tokens += ctx_tokens
            else:
                # 計算剩餘可用 token 數
                remaining_tokens = self.retrieval_max_tokens - total_tokens
                if remaining_tokens > 50:  # 只有在剩餘 token 夠多時才部分截取
                    # 部分截取當前 context
                    truncated_ctx = token_counter.truncate_text_by_tokens(ctx, remaining_tokens)
                    truncated_contexts.append(truncated_ctx)
                break
        
        if len(truncated_contexts) < len(contexts):
            print(f"📊 [{self.name}] Context 截斷: {len(contexts)} → {len(truncated_contexts)} chunks (總 tokens ≤ {self.retrieval_max_tokens})")
        
        return truncated_contexts
    
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
        非同步查詢並記錄執行時間與token使用量
        
        Args:
            user_query: 使用者查詢
        
        Returns:
            包含查詢結果、執行時間與token統計的字典
        """
        start_time = time.time()
        
        try:
            result = await self._execute_query(user_query)
        except Exception as e:
            result = self._handle_error(e)
        
        end_time = time.time()
        result["execution_time_sec"] = round(end_time - start_time, 4)
        
        # 計算context tokens（如果子類別沒有提供）
        if "context_tokens" not in result:
            result = self._add_token_stats(result)
        
        return result
    
    def _add_token_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        計算並添加token統計資訊
        
        Args:
            result: 查詢結果字典
        
        Returns:
            添加token統計後的結果字典
        """
        retrieved_contexts = result.get("retrieved_contexts", [])
        
        if not retrieved_contexts:
            result["context_tokens"] = 0
            result["context_token_details"] = {}
            return result
        
        # 使用TokenCounter計算
        token_counter = get_token_counter()
        token_details = token_counter.count_tokens_with_details(retrieved_contexts)
        
        result["context_tokens"] = token_details["total_tokens"]
        result["context_token_details"] = {
            "total_tokens": token_details["total_tokens"],
            "num_chunks": token_details["num_texts"],
            "avg_tokens_per_chunk": round(token_details["avg_tokens_per_text"], 2),
            "min_tokens": token_details["min_tokens"],
            "max_tokens": token_details["max_tokens"]
        }
        
        return result

    async def _llm_acomplete_with_cap(self, llm, prompt: str, max_tokens: int = 0):
        """
        帶 max_tokens 上限的非同步 completion，若底層不支援參數則自動降級。
        """
        if max_tokens and max_tokens > 0:
            try:
                return await llm.acomplete(prompt, max_tokens=max_tokens)
            except TypeError:
                pass
            try:
                return await llm.acomplete(prompt, num_predict=max_tokens)
            except TypeError:
                pass
        return await llm.acomplete(prompt)

    def _llm_complete_with_cap(self, llm, prompt: str, max_tokens: int = 0):
        """
        帶 max_tokens 上限的同步 completion，若底層不支援參數則自動降級。
        """
        if max_tokens and max_tokens > 0:
            try:
                return llm.complete(prompt, max_tokens=max_tokens)
            except TypeError:
                pass
            try:
                return llm.complete(prompt, num_predict=max_tokens)
            except TypeError:
                pass
        return llm.complete(prompt)
    
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
