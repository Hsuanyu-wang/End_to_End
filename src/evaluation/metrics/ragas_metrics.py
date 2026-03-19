"""
RAGAS 評估指標

提供基於 RAGAS 框架的評估指標，包括：
- AnswerRelevancy: 評估答案與問題的相關性
- ContextPrecision: 評估檢索上下文的精準度
- ContextRecall: 評估檢索上下文的召回率
- RAGASFaithfulness: 使用 RAGAS 版本的忠實度評估

參考 eval_rag_quality.py 的實作，確保使用實際檢索到的上下文進行評估
"""

import warnings
from typing import List, Optional
from .base import BaseMetric

# 條件性導入 RAGAS 相關套件
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    Dataset = None
    evaluate = None
    AnswerRelevancy = None
    ContextPrecision = None
    ContextRecall = None
    Faithfulness = None
    LangchainLLMWrapper = None
    ChatOpenAI = None
    OpenAIEmbeddings = None


class RAGASMetricBase(BaseMetric):
    """
    RAGAS 指標基底類別
    
    處理 RAGAS 依賴檢查與 LLM/Embeddings 初始化
    """
    
    def __init__(self, name: str, description: str, llm=None, embeddings=None):
        """
        初始化 RAGAS 指標
        
        Args:
            name: 指標名稱
            description: 指標描述
            llm: 評估用 LLM（可選）
            embeddings: 評估用 Embeddings（可選）
        """
        super().__init__(name=name, description=description)
        
        if not RAGAS_AVAILABLE:
            warnings.warn(
                f"{name} 需要 RAGAS 套件。請執行: pip install ragas datasets",
                ImportWarning
            )
            self.available = False
            return
        
        self.available = True
        self.llm = llm
        self.embeddings = embeddings
        
        # 若未提供 LLM/Embeddings，嘗試從環境建立預設值
        if self.llm is None or self.embeddings is None:
            self._init_default_llm_embeddings()
    
    def _init_default_llm_embeddings(self):
        """初始化預設的 LLM 與 Embeddings"""
        try:
            import os
            
            # 從環境變數讀取配置
            eval_llm_api_key = os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
            eval_llm_base_url = os.getenv("EVAL_LLM_BINDING_HOST")
            
            eval_embedding_api_key = (
                os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
                or os.getenv("EVAL_LLM_BINDING_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            eval_embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-large")
            eval_embedding_base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or os.getenv("EVAL_LLM_BINDING_HOST")
            
            if not eval_llm_api_key:
                warnings.warn("RAGAS 指標需要 API Key，請設定環境變數")
                self.available = False
                return
            
            # 建立 LLM
            llm_kwargs = {
                "model": eval_model,
                "api_key": eval_llm_api_key,
                "max_retries": int(os.getenv("EVAL_LLM_MAX_RETRIES", "5")),
                "request_timeout": int(os.getenv("EVAL_LLM_TIMEOUT", "180")),
            }
            if eval_llm_base_url:
                llm_kwargs["base_url"] = eval_llm_base_url
            
            base_llm = ChatOpenAI(**llm_kwargs)
            
            # 使用 LangchainLLMWrapper 包裝，啟用 bypass_n 模式
            self.llm = LangchainLLMWrapper(
                langchain_llm=base_llm,
                bypass_n=True
            )
            
            # 建立 Embeddings
            embedding_kwargs = {
                "model": eval_embedding_model,
                "api_key": eval_embedding_api_key,
            }
            if eval_embedding_base_url:
                embedding_kwargs["base_url"] = eval_embedding_base_url
            
            self.embeddings = OpenAIEmbeddings(**embedding_kwargs)
            
        except Exception as e:
            warnings.warn(f"初始化 RAGAS LLM/Embeddings 失敗: {e}")
            self.available = False


class AnswerRelevancyMetric(RAGASMetricBase):
    """
    答案相關性評估指標
    
    定義：評估生成答案是否直接回答使用者問題
    評估維度：答案的針對性、完整性、是否偏題
    
    範圍：[0, 1]（越高越好）
    """
    
    def __init__(self, llm=None, embeddings=None):
        super().__init__(
            name="answer_relevancy",
            description="評估答案與問題的相關性",
            llm=llm,
            embeddings=embeddings
        )
        
        if self.available:
            self.metric = AnswerRelevancy()
    
    async def compute_async(
        self,
        query: str,
        response: str,
        ground_truth: str = None
    ) -> Optional[float]:
        """
        非同步計算答案相關性分數
        
        Args:
            query: 使用者問題
            response: 生成的答案
            ground_truth: 標準答案（可選，RAGAS AnswerRelevancy 不需要）
        
        Returns:
            答案相關性分數 (0-1)，失敗時返回 None
        """
        if not self.available:
            return None
        
        try:
            # 建立單筆資料的 Dataset
            eval_dataset = Dataset.from_dict({
                "question": [query],
                "answer": [response],
                "contexts": [[""]],  # AnswerRelevancy 不需要 contexts，但 RAGAS 要求提供
            })
            
            # 執行評估
            result = evaluate(
                dataset=eval_dataset,
                metrics=[self.metric],
                llm=self.llm,
                embeddings=self.embeddings,
            )
            
            # 轉換為 DataFrame 並提取分數
            df = result.to_pandas()
            score = df.iloc[0].get("answer_relevancy", None)
            
            return float(score) if score is not None else None
            
        except Exception as e:
            print(f"  ⚠️ AnswerRelevancy 評估失敗: {e}")
            return None
    
    def compute(
        self,
        query: str,
        response: str,
        ground_truth: str = None
    ) -> Optional[float]:
        """
        同步計算答案相關性分數（不建議使用，RAGAS 建議使用異步）
        
        Args:
            query: 使用者問題
            response: 生成的答案
            ground_truth: 標準答案（可選）
        
        Returns:
            答案相關性分數 (0-1)，失敗時返回 None
        """
        import asyncio
        try:
            return asyncio.run(self.compute_async(query, response, ground_truth))
        except Exception as e:
            print(f"  ⚠️ AnswerRelevancy 同步評估失敗: {e}")
            return None


class ContextPrecisionMetric(RAGASMetricBase):
    """
    上下文精準度評估指標
    
    定義：評估檢索到的上下文是否精準（少雜訊）
    評估維度：相關上下文佔比、是否包含無關資訊
    
    範圍：[0, 1]（越高越好）
    """
    
    def __init__(self, llm=None, embeddings=None):
        super().__init__(
            name="context_precision",
            description="評估檢索上下文的精準度",
            llm=llm,
            embeddings=embeddings
        )
        
        if self.available:
            self.metric = ContextPrecision()
    
    async def compute_async(
        self,
        query: str,
        contexts: List[str],
        ground_truth: str
    ) -> Optional[float]:
        """
        非同步計算上下文精準度分數
        
        Args:
            query: 使用者問題
            contexts: 檢索到的上下文列表（實際檢索結果）
            ground_truth: 標準答案
        
        Returns:
            上下文精準度分數 (0-1)，失敗時返回 None
        """
        if not self.available:
            return None
        
        if not contexts or not ground_truth:
            return None
        
        try:
            # 建立單筆資料的 Dataset
            eval_dataset = Dataset.from_dict({
                "question": [query],
                "answer": [""],  # ContextPrecision 不需要 answer，但 RAGAS 要求提供
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            })
            
            # 執行評估
            result = evaluate(
                dataset=eval_dataset,
                metrics=[self.metric],
                llm=self.llm,
                embeddings=self.embeddings,
            )
            
            # 轉換為 DataFrame 並提取分數
            df = result.to_pandas()
            score = df.iloc[0].get("context_precision", None)
            
            return float(score) if score is not None else None
            
        except Exception as e:
            print(f"  ⚠️ ContextPrecision 評估失敗: {e}")
            return None
    
    def compute(
        self,
        query: str,
        contexts: List[str],
        ground_truth: str
    ) -> Optional[float]:
        """
        同步計算上下文精準度分數（不建議使用）
        
        Args:
            query: 使用者問題
            contexts: 檢索到的上下文列表
            ground_truth: 標準答案
        
        Returns:
            上下文精準度分數 (0-1)，失敗時返回 None
        """
        import asyncio
        try:
            return asyncio.run(self.compute_async(query, contexts, ground_truth))
        except Exception as e:
            print(f"  ⚠️ ContextPrecision 同步評估失敗: {e}")
            return None


class ContextRecallMetric(RAGASMetricBase):
    """
    上下文召回率評估指標
    
    定義：評估檢索到的上下文是否包含所有必要資訊
    評估維度：是否遺漏重要上下文、資訊完整度
    
    範圍：[0, 1]（越高越好）
    """
    
    def __init__(self, llm=None, embeddings=None):
        super().__init__(
            name="context_recall",
            description="評估檢索上下文的召回率",
            llm=llm,
            embeddings=embeddings
        )
        
        if self.available:
            self.metric = ContextRecall()
    
    async def compute_async(
        self,
        query: str,
        contexts: List[str],
        ground_truth: str
    ) -> Optional[float]:
        """
        非同步計算上下文召回率分數
        
        Args:
            query: 使用者問題
            contexts: 檢索到的上下文列表（實際檢索結果）
            ground_truth: 標準答案
        
        Returns:
            上下文召回率分數 (0-1)，失敗時返回 None
        """
        if not self.available:
            return None
        
        if not contexts or not ground_truth:
            return None
        
        try:
            # 建立單筆資料的 Dataset
            eval_dataset = Dataset.from_dict({
                "question": [query],
                "answer": [""],  # ContextRecall 不需要 answer，但 RAGAS 要求提供
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            })
            
            # 執行評估
            result = evaluate(
                dataset=eval_dataset,
                metrics=[self.metric],
                llm=self.llm,
                embeddings=self.embeddings,
            )
            
            # 轉換為 DataFrame 並提取分數
            df = result.to_pandas()
            score = df.iloc[0].get("context_recall", None)
            
            return float(score) if score is not None else None
            
        except Exception as e:
            print(f"  ⚠️ ContextRecall 評估失敗: {e}")
            return None
    
    def compute(
        self,
        query: str,
        contexts: List[str],
        ground_truth: str
    ) -> Optional[float]:
        """
        同步計算上下文召回率分數（不建議使用）
        
        Args:
            query: 使用者問題
            contexts: 檢索到的上下文列表
            ground_truth: 標準答案
        
        Returns:
            上下文召回率分數 (0-1)，失敗時返回 None
        """
        import asyncio
        try:
            return asyncio.run(self.compute_async(query, contexts, ground_truth))
        except Exception as e:
            print(f"  ⚠️ ContextRecall 同步評估失敗: {e}")
            return None


class RAGASFaithfulnessMetric(RAGASMetricBase):
    """
    RAGAS 版本的忠實度評估指標
    
    定義：使用 RAGAS 框架評估生成答案是否忠實於檢索到的上下文
    評估維度：是否包含幻覺、是否基於檢索結果生成
    
    範圍：[0, 1]（越高越好）
    
    註：與 LlamaIndex 的 FaithfulnessMetric 互補使用
    """
    
    def __init__(self, llm=None, embeddings=None):
        super().__init__(
            name="ragas_faithfulness",
            description="使用 RAGAS 評估答案忠實度",
            llm=llm,
            embeddings=embeddings
        )
        
        if self.available:
            self.metric = Faithfulness()
    
    async def compute_async(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Optional[float]:
        """
        非同步計算忠實度分數
        
        Args:
            query: 使用者問題
            response: 生成的答案
            contexts: 檢索到的上下文列表（實際檢索結果）
        
        Returns:
            忠實度分數 (0-1)，失敗時返回 None
        """
        if not self.available:
            return None
        
        if not contexts:
            return None
        
        try:
            # 建立單筆資料的 Dataset
            eval_dataset = Dataset.from_dict({
                "question": [query],
                "answer": [response],
                "contexts": [contexts],
            })
            
            # 執行評估
            result = evaluate(
                dataset=eval_dataset,
                metrics=[self.metric],
                llm=self.llm,
                embeddings=self.embeddings,
            )
            
            # 轉換為 DataFrame 並提取分數
            df = result.to_pandas()
            score = df.iloc[0].get("faithfulness", None)
            
            return float(score) if score is not None else None
            
        except Exception as e:
            print(f"  ⚠️ RAGAS Faithfulness 評估失敗: {e}")
            return None
    
    def compute(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Optional[float]:
        """
        同步計算忠實度分數（不建議使用）
        
        Args:
            query: 使用者問題
            response: 生成的答案
            contexts: 檢索到的上下文列表
        
        Returns:
            忠實度分數 (0-1)，失敗時返回 None
        """
        import asyncio
        try:
            return asyncio.run(self.compute_async(query, response, contexts))
        except Exception as e:
            print(f"  ⚠️ RAGAS Faithfulness 同步評估失敗: {e}")
            return None
