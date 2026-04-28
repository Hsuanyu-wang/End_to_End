"""
RAGAS 評估指標

提供基於 RAGAS 框架的評估指標，包括：
- AnswerRelevancy: 評估答案與問題的相關性
- ContextPrecision: 評估檢索上下文的精準度
- ContextRecall: 評估檢索上下文的召回率
- RAGASFaithfulness: 使用 RAGAS 版本的忠實度評估

參考 eval_rag_quality.py 的實作，確保使用實際檢索到的上下文進行評估
"""

import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base import BaseMetric


def _normalize_ollama_openai_base_url(url: str) -> str:
    """Ollama OpenAI 相容 API 需使用 .../v1 作為 ChatOpenAI base_url。"""
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    if u.endswith("/v1"):
        return u
    return f"{u}/v1"


def _load_ragas_defaults_from_config() -> Optional[Dict[str, Any]]:
    """
    從 config.yml 讀取 eval_model / model，供無 EVAL_* 環境變數時與 LlamaIndex 評估後端對齊。
    可由環境變數 RAGAS_CONFIG_PATH 覆寫設定檔路徑。
    """
    import os

    path = os.getenv("RAGAS_CONFIG_PATH", "/home/End_to_End_RAG/config.yml")
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            return None
        eval_m = cfg.get("eval_model") or {}
        model_m = cfg.get("model") or {}
        eval_url = eval_m.get("ollama_url")
        eval_model = eval_m.get("llm_model")
        embed_url = model_m.get("ollama_url")
        embed_model = model_m.get("embed_model")
        if not eval_url or not eval_model:
            return None
        return {
            "llm_base_url": _normalize_ollama_openai_base_url(str(eval_url)),
            "llm_model": str(eval_model),
            "embedding_base_url": _normalize_ollama_openai_base_url(
                str(embed_url or eval_url)
            ),
            "embedding_model": str(embed_model or "nomic-embed-text"),
        }
    except Exception:
        return None


def _coerce_ragas_score(score: Any) -> Optional[float]:
    """將 RAGAS 輸出轉為 float；None / NaN / Inf 時回傳 None，避免寫入 CSV 為 nan。"""
    if score is None:
        return None
    try:
        f = float(score)
    except (TypeError, ValueError):
        return None

    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _normalize_text_input(value: Any) -> str:
    """將任意輸入安全轉為字串，並處理 None/NaN。"""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


# 條件性導入 RAGAS 相關套件
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        AnswerCorrectness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LiteLLMEmbeddings
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
    AnswerCorrectness = None
    LangchainLLMWrapper = None
    LiteLLMEmbeddings = None
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
        """初始化預設的 LLM 與 Embeddings（優先環境變數，其次 config.yml 與 LlamaIndex 評估後端對齊）"""
        try:
            import os

            # 從環境變數讀取配置
            eval_llm_api_key = os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            eval_model = os.getenv("EVAL_LLM_MODEL")
            eval_llm_base_url = os.getenv("EVAL_LLM_BINDING_HOST")

            eval_embedding_api_key = (
                os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
                or os.getenv("EVAL_LLM_BINDING_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            eval_embedding_model = os.getenv("EVAL_EMBEDDING_MODEL")
            eval_embedding_base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or os.getenv(
                "EVAL_LLM_BINDING_HOST"
            )
            eval_llm_base_url = _normalize_ollama_openai_base_url(eval_llm_base_url)
            eval_embedding_base_url = _normalize_ollama_openai_base_url(eval_embedding_base_url)

            # 無 EVAL_* 時自 config.yml 補齊（Ollama OpenAI 相容端點）
            fb = _load_ragas_defaults_from_config()
            if fb:
                if not eval_model:
                    eval_model = fb["llm_model"]
                if not eval_llm_base_url:
                    eval_llm_base_url = fb["llm_base_url"]
                if not eval_embedding_model:
                    eval_embedding_model = fb["embedding_model"]
                if not eval_embedding_base_url:
                    eval_embedding_base_url = fb["embedding_base_url"]

            if not eval_model:
                eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
            if not eval_embedding_model:
                eval_embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-large")

            # Ollama 等 OpenAI 相容端點不需真實 key，LangChain 仍需非空 api_key
            compat_key = os.getenv("EVAL_OPENAI_COMPAT_API_KEY", "ollama")
            if not eval_llm_api_key and eval_llm_base_url:
                eval_llm_api_key = compat_key
            if not eval_embedding_api_key and eval_embedding_base_url:
                eval_embedding_api_key = compat_key

            if not eval_llm_api_key:
                warnings.warn(
                    "RAGAS 指標需要 API Key、EVAL_LLM_BINDING_HOST，或可用的 config.yml（eval_model.ollama_url）",
                    stacklevel=2,
                )
                self.available = False
                return
            if not eval_embedding_api_key:
                warnings.warn(
                    "RAGAS 指標需要 Embedding API Key、EVAL_EMBEDDING_BINDING_HOST / EVAL_LLM_BINDING_HOST，"
                    "或可用的 config.yml（model.embed_model 與 ollama_url）",
                    stacklevel=2,
                )
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
            # 有自訂 base_url（Ollama 等本地端點）→ LiteLLMEmbeddings，避免 OpenAIEmbeddings
            # 對 Ollama /v1/embeddings 傳送不相容格式（400 invalid input type）
            if eval_embedding_base_url:
                # LiteLLMEmbeddings 的 api_base 不含 /v1 尾綴
                litellm_api_base = eval_embedding_base_url
                if litellm_api_base.endswith("/v1"):
                    litellm_api_base = litellm_api_base[:-3]
                litellm_api_base = litellm_api_base.rstrip("/")
                self.embeddings = LiteLLMEmbeddings(
                    model=f"ollama/{eval_embedding_model}",
                    api_base=litellm_api_base,
                    api_key=eval_embedding_api_key,
                )
            else:
                # 真實 OpenAI API
                self.embeddings = OpenAIEmbeddings(
                    model=eval_embedding_model,
                    api_key=eval_embedding_api_key,
                )
            
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
            
            return _coerce_ragas_score(score)
            
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
            
            return _coerce_ragas_score(score)
            
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
            
            return _coerce_ragas_score(score)
            
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


class AnswerCorrectnessMetric(RAGASMetricBase):
    """
    Ragas answer_correctness 評估指標

    定義：結合語義相似度（embedding）與 NLI 事實一致性，綜合評估生成答案與標準答案的正確程度。
    與 LlamaIndex CorrectnessMetric（0-5 分）互補，此指標範圍 [0, 1]。

    所需欄位：user_input（問題）、response（生成答案）、reference（標準答案）
    範圍：[0, 1]（越高越好）
    """

    def __init__(self, llm=None, embeddings=None):
        super().__init__(
            name="ragas_answer_correctness",
            description="使用 Ragas 評估答案綜合正確性（語義相似度 + 事實一致性）",
            llm=llm,
            embeddings=embeddings,
        )

        if self.available:
            self.metric = AnswerCorrectness()

    async def compute_async(
        self,
        query: str,
        response: str,
        reference: str,
    ) -> Optional[float]:
        """
        非同步計算 answer_correctness 分數

        Args:
            query: 使用者問題
            response: 生成的答案
            reference: 標準答案

        Returns:
            answer_correctness 分數 (0-1)，失敗時返回 None
        """
        if not self.available:
            return None
        query_text = _normalize_text_input(query)
        response_text = _normalize_text_input(response)
        reference_text = _normalize_text_input(reference)
        if not reference_text:
            return None

        try:
            eval_dataset = Dataset.from_dict({
                "user_input": [query_text],
                "response": [response_text],
                "reference": [reference_text],
            })

            result = evaluate(
                dataset=eval_dataset,
                metrics=[self.metric],
                llm=self.llm,
                embeddings=self.embeddings,
            )

            df = result.to_pandas()
            score = df.iloc[0].get("answer_correctness", None)
            parsed_score = _coerce_ragas_score(score)
            if parsed_score is not None:
                return parsed_score
            return None

        except Exception as e:
            print(
                "  ⚠️ Ragas AnswerCorrectness 評估失敗: "
                f"{e} | types(q/r/ref)=({type(query).__name__}/{type(response).__name__}/{type(reference).__name__}) "
                f"| lens(q/r/ref)=({len(query_text)}/{len(response_text)}/{len(reference_text)})"
            )
            return None

    def compute(
        self,
        query: str,
        response: str,
        reference: str,
    ) -> Optional[float]:
        """同步介面（不建議使用）"""
        import asyncio
        try:
            return asyncio.run(self.compute_async(query, response, reference))
        except Exception as e:
            print(f"  ⚠️ Ragas AnswerCorrectness 同步評估失敗: {e}")
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
            
            return _coerce_ragas_score(score)
            
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
