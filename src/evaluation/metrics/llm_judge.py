"""
LLM-as-a-Judge 評估指標

使用 LLM 作為評估者，評估生成答案的正確性與忠實度
"""

import json
import re
from typing import List, Optional, Tuple
from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator
from llama_index.core import Settings as LlamaSettings
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
            llm = LlamaSettings.eval_llm
        
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


class KeyFactCoverageMetric(BaseMetric):
    """
    關鍵事實覆蓋率指標

    兩步驟流程：
    1. 用 LLM 從 ground truth 提取原子性事實（人名、日期、系統元件、數字、事件等）
    2. 用 LLM 逐一核查生成答案是否覆蓋每個事實（允許語義等價的表述）

    輸出三個子分數（均為 [0, 1]）：
    - recall:    gt 事實被 gen_answer 覆蓋的比例
    - precision: gen_answer 的聲明被 gt 支持的比例
    - f1:        recall 與 precision 的調和平均
    另輸出 gt_fact_count / gen_claim_count 供診斷。

    此指標專門針對「多跳事實提取」類問題，可補足 ROUGE/BERTScore
    因表面字詞差異而失效的場景。
    """

    # Step-1: 從標準答案提取原子事實
    _EXTRACT_FACTS_PROMPT = """\
你是資訊萃取專家。請從「標準答案」中提取所有原子性事實。
每個事實須為獨立、可驗證的資訊片段（例如：人名、日期、系統/元件名稱、數字、具體事件）。
勿包含推測性或模糊敘述。

標準答案：
{reference}

請以如下 JSON 格式回傳（不要有任何其他文字）：
{{"facts": ["事實1", "事實2", ...]}}"""

    # Step-2: 從生成答案提取聲明（用於 precision）
    _EXTRACT_CLAIMS_PROMPT = """\
你是資訊萃取專家。請從「生成答案」中提取所有具體聲明。
每個聲明須為獨立、可驗證的資訊片段。

生成答案：
{generated}

請以如下 JSON 格式回傳（不要有任何其他文字）：
{{"claims": ["聲明1", "聲明2", ...]}}"""

    # Step-3a: 核查 gt facts 是否被 gen_answer 覆蓋（recall）
    _CHECK_RECALL_PROMPT = """\
你是事實核查專家。請逐一判斷下列「關鍵事實」是否在「生成答案」中被提到或覆蓋（允許語義等價的不同表述）。

關鍵事實列表（共 {n} 條）：
{facts_numbered}

生成答案：
{generated}

請以如下 JSON 格式回傳（布林值陣列，順序與事實列表一致，不要有任何其他文字）：
{{"covered": [true/false, ...]}}"""

    # Step-3b: 核查 gen_answer claims 是否被 gt 支持（precision）
    _CHECK_PRECISION_PROMPT = """\
你是事實核查專家。請逐一判斷下列「生成聲明」是否能在「標準答案」中找到支持或依據（允許語義等價的不同表述）。

生成聲明列表（共 {n} 條）：
{claims_numbered}

標準答案：
{reference}

請以如下 JSON 格式回傳（布林值陣列，順序與聲明列表一致，不要有任何其他文字）：
{{"supported": [true/false, ...]}}"""

    def __init__(self, llm=None):
        super().__init__(
            name="key_fact_coverage",
            description="用 LLM 評估生成答案對 ground truth 關鍵事實的覆蓋率"
        )
        if llm is None:
            llm = LlamaSettings.eval_llm
        self.llm = llm

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _numbered_list(items: List[str]) -> str:
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    @staticmethod
    def _parse_bool_list(text: str, key: str, n: int) -> List[bool]:
        """從 LLM 回傳的 JSON 文字中解析布林列表，失敗時回傳全 False。"""
        try:
            # 嘗試從回傳文字中找到 JSON 區塊
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                values = data.get(key, [])
                # 統一轉換為 bool（LLM 有時回傳字串）
                result = []
                for v in values[:n]:
                    if isinstance(v, bool):
                        result.append(v)
                    elif isinstance(v, str):
                        result.append(v.lower() in ("true", "yes", "1"))
                    else:
                        result.append(bool(v))
                # 若長度不足，補 False
                while len(result) < n:
                    result.append(False)
                return result
        except Exception:
            pass
        return [False] * n

    @staticmethod
    def _parse_str_list(text: str, key: str) -> List[str]:
        """從 LLM 回傳的 JSON 文字中解析字串列表，失敗時回傳空列表。"""
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [str(x) for x in data.get(key, [])]
        except Exception:
            pass
        return []

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    # 非同步實作
    # ------------------------------------------------------------------

    async def compute_async(
        self,
        query: str,
        response: str,
        reference: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[int]]:
        """
        非同步計算關鍵事實覆蓋率

        Returns:
            (recall, precision, f1, gt_fact_count, gen_claim_count)
            任一步驟失敗時對應值為 None
        """
        try:
            # Step 1: 提取 gt 事實
            facts_resp = await self.llm.acomplete(
                self._EXTRACT_FACTS_PROMPT.format(reference=reference)
            )
            facts = self._parse_str_list(str(facts_resp), "facts")
            if not facts:
                return None, None, None, 0, None

            # Step 2: 提取 gen 聲明
            claims_resp = await self.llm.acomplete(
                self._EXTRACT_CLAIMS_PROMPT.format(generated=response)
            )
            claims = self._parse_str_list(str(claims_resp), "claims")

            # Step 3a: 計算 recall（gt facts 被 gen 覆蓋）
            recall_resp = await self.llm.acomplete(
                self._CHECK_RECALL_PROMPT.format(
                    n=len(facts),
                    facts_numbered=self._numbered_list(facts),
                    generated=response,
                )
            )
            covered = self._parse_bool_list(str(recall_resp), "covered", len(facts))
            recall = sum(covered) / len(facts)

            # Step 3b: 計算 precision（gen claims 被 gt 支持）
            if claims:
                prec_resp = await self.llm.acomplete(
                    self._CHECK_PRECISION_PROMPT.format(
                        n=len(claims),
                        claims_numbered=self._numbered_list(claims),
                        reference=reference,
                    )
                )
                supported = self._parse_bool_list(str(prec_resp), "supported", len(claims))
                precision = sum(supported) / len(claims)
            else:
                precision = 0.0

            f1 = self._f1(precision, recall)
            return recall, precision, f1, len(facts), len(claims)

        except Exception as e:
            print(f"  ⚠️ KeyFactCoverage 評估失敗: {e}")
            return None, None, None, None, None

    def compute(self, **kwargs):
        """同步介面（預留，建議使用 compute_async）"""
        raise NotImplementedError("KeyFactCoverageMetric 請使用 compute_async")


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
            llm = LlamaSettings.eval_llm
        
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
