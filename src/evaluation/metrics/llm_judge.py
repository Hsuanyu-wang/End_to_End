"""
LLM-as-a-Judge 評估指標

使用 LLM 作為評估者，評估生成答案的正確性與忠實度
"""

from typing import List, Optional
from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator
from llama_index.core import Settings
from .base import BaseMetric


class CorrectnessMetric(BaseMetric):
    """
    正確性評估指標
    
    定義：使用 LLM 評估生成答案與標準答案的一致性
    評估維度：語義正確性、資訊完整性、準確度
    
    範圍：[0, 5] (LlamaIndex 預設評分範圍)
    """
    
    def __init__(self, llm=None):
        super().__init__(
            name="correctness",
            description="使用 LLM 評估答案正確性"
        )
        if llm is None:
            llm = Settings.eval_llm
        
        self.evaluator = CorrectnessEvaluator(llm=llm)
    
    async def compute_async(
        self,
        query: str,
        response: str,
        reference: str
    ) -> Optional[float]:
        """
        非同步計算正確性分數
        
        Args:
            query: 使用者問題
            response: 生成的答案
            reference: 標準答案
        
        Returns:
            正確性分數 (0-5)，失敗時返回 None
        """
        try:
            result = await self.evaluator.aevaluate(
                query=query,
                response=response,
                reference=reference
            )
            return result.score
        except Exception as e:
            print(f"  ⚠️ Correctness 評估失敗: {e}")
            return None
    
    def compute(
        self,
        query: str,
        response: str,
        reference: str
    ) -> Optional[float]:
        """
        同步計算正確性分數
        
        Args:
            query: 使用者問題
            response: 生成的答案
            reference: 標準答案
        
        Returns:
            正確性分數 (0-5)，失敗時返回 None
        """
        try:
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                reference=reference
            )
            return result.score
        except Exception as e:
            print(f"  ⚠️ Correctness 評估失敗: {e}")
            return None


class FaithfulnessMetric(BaseMetric):
    """
    忠實度評估指標
    
    定義：使用 LLM 評估生成答案是否忠實於檢索到的上下文
    評估維度：是否包含幻覺、是否基於檢索結果生成
    
    範圍：{0, 1} (通過/不通過)
    """
    
    def __init__(self, llm=None):
        super().__init__(
            name="faithfulness",
            description="使用 LLM 評估答案忠實度"
        )
        if llm is None:
            llm = Settings.eval_llm
        
        self.evaluator = FaithfulnessEvaluator(llm=llm)
    
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
            contexts: 檢索到的上下文列表
        
        Returns:
            忠實度分數 (1.0=忠實, 0.0=不忠實)，失敗時返回 None
        """
        try:
            result = await self.evaluator.aevaluate(
                query=query,
                response=response,
                contexts=contexts
            )
            return 1.0 if result.passing else 0.0
        except Exception as e:
            print(f"  ⚠️ Faithfulness 評估失敗: {e}")
            return None
    
    def compute(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Optional[float]:
        """
        同步計算忠實度分數
        
        Args:
            query: 使用者問題
            response: 生成的答案
            contexts: 檢索到的上下文列表
        
        Returns:
            忠實度分數 (1.0=忠實, 0.0=不忠實)，失敗時返回 None
        """
        try:
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts
            )
            return 1.0 if result.passing else 0.0
        except Exception as e:
            print(f"  ⚠️ Faithfulness 評估失敗: {e}")
            return None
