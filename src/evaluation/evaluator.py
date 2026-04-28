"""
RAG 評估引擎

提供完整的 RAG Pipeline 評估功能
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

from src.evaluation.metrics import (
    HitRateMetric,
    MRRMetric,
    RetrievalF1Metric,
    ROUGEMetric,
    BLEUMetric,
    METEORMetric,
    BERTScoreMetric,
    TokenF1Metric,
    JiebaF1Metric,
    CorrectnessMetric,
    FaithfulnessMetric,
    KeyFactCoverageMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    RAGASFaithfulnessMetric,
    AnswerCorrectnessMetric,
)


class RAGEvaluator:
    """
    RAG 評估器
    
    統籌評估流程，計算所有指標並生成報告
    
    Attributes:
        eval_llm: 用於 LLM-as-Judge 的語言模型
        base_eval_dir: 評估結果儲存目錄
    """
    
    def __init__(self, eval_llm, base_eval_dir: str = None, metrics_mode: str = "kfc_only"):
        """
        初始化評估器
        
        Args:
            eval_llm: 用於 LLM-as-Judge 的語言模型
            base_eval_dir: 評估結果儲存目錄，預設為 results/evaluation_results_{timestamp}
        """
        self.eval_llm = eval_llm
        self.metrics_mode = metrics_mode
        
        if base_eval_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_eval_dir = f"/home/End_to_End_RAG/results/evaluation_results_{timestamp}"
        
        self.base_eval_dir = base_eval_dir
        os.makedirs(self.base_eval_dir, exist_ok=True)
        
        # 初始化指標
        self._init_metrics()
    
    def _init_metrics(self):
        """初始化所有評估指標"""
        # KFC 永遠保留（目前預設模式）
        self.key_fact_coverage_metric = KeyFactCoverageMetric(llm=self.eval_llm)

        # 預設占位，避免 kfc_only 模式下誤用到未初始化屬性
        self.hit_rate_metric = None
        self.mrr_metric = None
        self.retrieval_f1_metric = None
        self.rouge_metric = None
        self.bleu_metric = None
        self.meteor_metric = None
        self.bertscore_metric = None
        self.token_f1_metric = None
        self.jieba_f1_metric = None
        self.correctness_metric = None
        self.faithfulness_metric = None
        self.answer_relevancy_metric = None
        self.context_precision_metric = None
        self.context_recall_metric = None
        self.ragas_faithfulness_metric = None
        self.answer_correctness_metric = None

        if self.metrics_mode != "full":
            return

        # 檢索指標
        self.hit_rate_metric = HitRateMetric()
        self.mrr_metric = MRRMetric()
        self.retrieval_f1_metric = RetrievalF1Metric()

        # 生成指標
        self.rouge_metric = ROUGEMetric()
        self.bleu_metric = BLEUMetric()
        self.meteor_metric = METEORMetric()
        self.bertscore_metric = BERTScoreMetric(lang="zh")
        self.token_f1_metric = TokenF1Metric()
        self.jieba_f1_metric = JiebaF1Metric()

        # LLM Judge 指標 (LlamaIndex)
        self.correctness_metric = CorrectnessMetric(llm=self.eval_llm)
        self.faithfulness_metric = FaithfulnessMetric(llm=self.eval_llm)
        self.key_fact_coverage_metric = KeyFactCoverageMetric(llm=self.eval_llm)

        # RAGAS 指標
        # 註：RAGAS 指標會自動從環境變數讀取配置，與 LlamaIndex 指標共用評估 LLM
        self.answer_relevancy_metric = AnswerRelevancyMetric()
        self.context_precision_metric = ContextPrecisionMetric()
        self.context_recall_metric = ContextRecallMetric()
        self.ragas_faithfulness_metric = RAGASFaithfulnessMetric()
        self.answer_correctness_metric = AnswerCorrectnessMetric()

    @staticmethod
    def _normalize_doc_ids(doc_ids: Any) -> List[str]:
        """將 ground truth/retrieved 的 ID 統一正規化為字串列表。"""
        if doc_ids is None:
            return []

        if isinstance(doc_ids, str):
            candidates = re.split(r"[\n,]+", doc_ids)
            return [c.strip() for c in candidates if c and c.strip()]

        normalized = []
        if isinstance(doc_ids, list):
            for item in doc_ids:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts = re.split(r"[\n,]+", item)
                    normalized.extend([p.strip() for p in parts if p and p.strip()])
                else:
                    normalized.append(str(item))
        return normalized

    @staticmethod
    def _clamp_retrieval_metric(value: float, metric_name: str, query: str, idx: int, pipeline_name: str) -> float:
        """將 retrieval 指標限制在 [0, 1]，超界時輸出 warning。"""
        if value is None:
            return 0.0
        clamped = max(0.0, min(1.0, float(value)))
        if clamped != value:
            print(
                f"⚠️ [RetrievalClamp] pipeline={pipeline_name}, idx={idx}, metric={metric_name}, "
                f"original={value}, clamped={clamped}, query={query[:80]}"
            )
        return clamped
    
    async def compute_metrics_for_sample(
        self,
        idx: int,
        source: str,
        query: str,
        gt_answer: str,
        gen_answer: str,
        gt_ids: List[str],
        retrieved_ids: List[str],
        retrieved_contexts: List[str],
        execution_time_sec: float,
        schema_info: Dict[str, Any] = None,
        context_tokens: int = 0,
        context_token_details: Dict[str, Any] = None,
        pipeline_name: str = ""
    ) -> Dict[str, Any]:
        """
        計算單一樣本的所有指標
        
        Args:
            idx: 樣本索引
            source: 資料來源
            query: 使用者問題
            gt_answer: 標準答案
            gen_answer: 生成的答案
            gt_ids: 標準答案的文件 ID
            retrieved_ids: 檢索到的文件 ID
            retrieved_contexts: 檢索到的上下文
            execution_time_sec: 執行時間
            schema_info: Schema 資訊（只在第一筆記錄）
            context_tokens: 總context token數
            context_token_details: 詳細token統計
        
        Returns:
            包含所有指標的字典
        """
        # 正規化 doc ids，避免 DI/GEN 格式差異影響評估
        gt_ids = self._normalize_doc_ids(gt_ids)
        retrieved_ids = self._normalize_doc_ids(retrieved_ids)

        is_full_metrics = self.metrics_mode == "full"

        # 檢索指標（僅 full）
        hit_rate = None
        mrr = None
        recall = None
        precision = None
        f1_score = None
        if is_full_metrics:
            hit_rate = self.hit_rate_metric.compute(retrieved_ids, gt_ids)
            mrr = self.mrr_metric.compute(retrieved_ids, gt_ids)
            recall, precision, f1_score = self.retrieval_f1_metric.compute(retrieved_ids, gt_ids)
            hit_rate = self._clamp_retrieval_metric(hit_rate, "hit_rate", query, idx, pipeline_name)
            mrr = self._clamp_retrieval_metric(mrr, "mrr", query, idx, pipeline_name)
            recall = self._clamp_retrieval_metric(recall, "retrieval_recall", query, idx, pipeline_name)
            precision = self._clamp_retrieval_metric(precision, "retrieval_precision", query, idx, pipeline_name)
            f1_score = self._clamp_retrieval_metric(f1_score, "retrieval_f1_score", query, idx, pipeline_name)

        # 生成指標（僅 full）
        rouge1 = rouge2 = rougeL = rougeLsum = None
        bleu = meteor = None
        bert_f1 = tok_f1 = jieba_f1 = None
        if is_full_metrics:
            _, _, jieba_f1 = self.jieba_f1_metric.compute(gen_answer, gt_answer)
            rouge1, rouge2, rougeL, rougeLsum = self.rouge_metric.compute(gen_answer, gt_answer)
            bleu = self.bleu_metric.compute(gen_answer, gt_answer)
            meteor = self.meteor_metric.compute(gen_answer, gt_answer)
            _, _, bert_f1 = self.bertscore_metric.compute(gen_answer, gt_answer)
            _, _, tok_f1 = self.token_f1_metric.compute(gen_answer, gt_answer)

        # LLM-as-Judge 指標（僅 full）
        correctness_score = None
        faithfulness_score = None
        if is_full_metrics:
            try:
                correctness_score = await self.correctness_metric.compute_async(
                    query=query,
                    response=gen_answer,
                    reference=gt_answer
                )

                faithfulness_score = await self.faithfulness_metric.compute_async(
                    query=query,
                    response=gen_answer,
                    contexts=retrieved_contexts
                )
            except Exception as e:
                print(f"  ⚠️ 評測 LLM API 呼叫失敗: {e}")
                correctness_score, faithfulness_score = None, None

        # 關鍵事實覆蓋率
        (
            kfc_recall,
            kfc_precision,
            kfc_f1,
            kfc_gt_fact_count,
            kfc_gen_claim_count,
        ) = await self.key_fact_coverage_metric.compute_async(
            query=query,
            response=gen_answer,
            reference=gt_answer,
        )
        
        # RAGAS 指標（僅 full）
        answer_relevancy_score = None
        context_precision_score = None
        context_recall_score = None
        ragas_faithfulness_score = None
        ragas_answer_correctness_score = None
        if is_full_metrics:
            try:
                # AnswerRelevancy: 評估答案與問題的相關性
                # answer_relevancy_score = await self.answer_relevancy_metric.compute_async(
                #     query=query,
                #     response=gen_answer,
                #     ground_truth=gt_answer
                # )

                # ContextPrecision: 評估檢索上下文的精準度
                # 註：使用實際檢索到的上下文 (retrieved_contexts)
                context_precision_score = await self.context_precision_metric.compute_async(
                    query=query,
                    contexts=retrieved_contexts,
                    ground_truth=gt_answer
                )

                # ContextRecall: 評估檢索上下文的召回率
                # 註：使用實際檢索到的上下文 (retrieved_contexts)
                context_recall_score = await self.context_recall_metric.compute_async(
                    query=query,
                    contexts=retrieved_contexts,
                    ground_truth=gt_answer
                )

                # RAGASFaithfulness: RAGAS 版本的忠實度評估
                # 註：使用實際檢索到的上下文 (retrieved_contexts)
                ragas_faithfulness_score = await self.ragas_faithfulness_metric.compute_async(
                    query=query,
                    response=gen_answer,
                    contexts=retrieved_contexts
                )
                # Ragas answer_correctness（結合語義相似度 + 事實一致性，範圍 [0, 1]）
                ragas_answer_correctness_score = await self.answer_correctness_metric.compute_async(
                    query=query,
                    response=gen_answer,
                    reference=gt_answer,
                )
            except Exception as e:
                print(f"  ⚠️ RAGAS 指標計算失敗: {e}")
        
        # 提取詳細token資訊
        entity_tokens = 0
        relation_tokens = 0
        chunk_tokens = 0
        
        if context_token_details:
            entity_tokens = context_token_details.get("entity_tokens", 0)
            relation_tokens = context_token_details.get("relation_tokens", 0)
            chunk_tokens = context_token_details.get("chunk_tokens", 0)
        
        # 統整回傳
        result = {
            "idx": idx,
            "dataset_source": source,
            "query": query,
            "ground_truth_answer": gt_answer,
            "generated_answer": gen_answer,
            "ground_truth_ids": ", ".join(gt_ids),
            "retrieved_ids": ", ".join(retrieved_ids),
            "execution_time_sec": execution_time_sec,
            # Token統計
            "context_tokens": context_tokens,
            "entity_tokens": entity_tokens,
            "relation_tokens": relation_tokens,
            "chunk_tokens": chunk_tokens,
            # 檢索指標
            "hit_rate": hit_rate,
            "mrr": mrr,
            "retrieval_recall": recall,
            "retrieval_precision": precision,
            "retrieval_f1_score": f1_score,
            # 生成指標（已使用 jieba 分詞，適用中英混合文本）
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "rougeLsum": rougeLsum,
            "bleu": bleu,
            "meteor": meteor,
            "bertscore_f1": bert_f1,
            "token_f1": tok_f1,
            "jieba_f1": jieba_f1,
            # _zh 欄位：與回溯重算結果欄名對齊，值與上方相同（底層指標已使用 jieba）
            "rouge1_zh": rouge1,
            "rouge2_zh": rouge2,
            "rougeL_zh": rougeL,
            "rougeLsum_zh": rougeLsum,
            "bleu_zh": bleu,
            "meteor_zh": meteor,
            # LLM Judge 指標 (LlamaIndex)
            "correctness_score": correctness_score,
            "faithfulness_score": faithfulness_score,
            # 關鍵事實覆蓋率
            "kfc_recall": kfc_recall,
            "kfc_precision": kfc_precision,
            "kfc_f1": kfc_f1,
            "kfc_gt_fact_count": kfc_gt_fact_count,
            "kfc_gen_claim_count": kfc_gen_claim_count,
            # RAGAS 指標
            "answer_relevancy": answer_relevancy_score,
            "context_precision": context_precision_score,
            "context_recall": context_recall_score,
            "ragas_faithfulness": ragas_faithfulness_score,
            "ragas_answer_correctness": ragas_answer_correctness_score,
        }
        
        # 只在第一筆資料（idx=1）時記錄 schema 資訊
        if idx == 1 and schema_info:
            result["schema_method"] = schema_info.get("method", "")
            result["schema_entities"] = json.dumps(schema_info.get("entities", []), ensure_ascii=False)
            result["schema_relations"] = json.dumps(schema_info.get("relations", []), ensure_ascii=False)
            result["schema_validation"] = json.dumps(schema_info.get("validation_schema", {}), ensure_ascii=False)
        else:
            result["schema_method"] = ""
            result["schema_entities"] = ""
            result["schema_relations"] = ""
            result["schema_validation"] = ""
        
        return result
    
    async def evaluate_pipeline(
        self,
        pipeline,
        qa_datasets: List[Dict[str, Any]],
        pipeline_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        評估單一 RAG Pipeline
        
        Args:
            pipeline: RAG Pipeline Wrapper
            qa_datasets: QA 資料集列表
            pipeline_name: Pipeline 名稱（可選，預設使用 pipeline.name）
        
        Returns:
            評估結果列表
        """
        if pipeline_name is None:
            pipeline_name = pipeline.name
        
        print(f"\n🚀 開始評測 Pipeline: {pipeline_name}")
        
        # 取得 pipeline 的 schema 資訊（如果有）
        schema_info = getattr(pipeline, "schema_info", None)
        if schema_info:
            print(f"  📋 Schema 方法: {schema_info.get('method', 'N/A')}")
            print(f"  📋 實體類型數量: {len(schema_info.get('entities', []))}")
        
        results = []
        
        for idx, qa in tqdm(enumerate(qa_datasets), total=len(qa_datasets), desc=f"評估 {pipeline_name} 進度"):
            query = qa["query"]
            gt_answer = qa["ground_truth_answer"]
            gt_ids = self._normalize_doc_ids(qa["ground_truth_doc_ids"])
            
            try:
                # 執行檢索與生成
                result = await pipeline.aquery_and_log(query)
                
                # 提取token資訊
                context_tokens = result.get("context_tokens", 0)
                context_token_details = result.get("context_token_details", {})
                
                # 計算所有指標
                metrics_row = await self.compute_metrics_for_sample(
                    idx=idx + 1,
                    source=qa["source"],
                    query=query,
                    gt_answer=gt_answer,
                    gen_answer=result["generated_answer"],
                    gt_ids=gt_ids,
                    retrieved_ids=result["retrieved_ids"],
                    retrieved_contexts=result["retrieved_contexts"],
                    execution_time_sec=result["execution_time_sec"],
                    schema_info=schema_info,
                    context_tokens=context_tokens,
                    context_token_details=context_token_details,
                    pipeline_name=pipeline_name
                )

                # 題型欄位（僅保留資料本身已存在的 Q_type；不存在則留空）
                metrics_row["Q_type"] = qa.get("Q_type", "") or ""

                # 保留完整 retrieval 內容供後續分析（不納入評估指標計算）
                metrics_row["retrieved_entities"] = result.get("retrieved_entities", [])
                metrics_row["retrieved_relations"] = result.get("retrieved_relations", [])
                metrics_row["retrieved_contexts"] = result.get("retrieved_contexts", [])

                results.append(metrics_row)
            
            except Exception as e:
                tqdm.write(f"⚠️ 第 {idx+1} 題評測崩潰，跳過。錯誤: {e}")
                continue
        
        return results


# 向後兼容函數
async def compute_and_format_metrics(
    idx: int,
    source: str,
    query: str,
    gt_answer: str,
    gen_answer: str,
    gt_ids: List[str],
    retrieved_ids: List[str],
    retrieved_contexts: List[str],
    execution_time_sec: float,
    correctness_evaluator,
    faithfulness_evaluator
) -> Dict[str, Any]:
    """
    計算並格式化指標（向後兼容函數）
    
    Args:
        參數說明見 RAGEvaluator.compute_metrics_for_sample
        correctness_evaluator: CorrectnessEvaluator 實例
        faithfulness_evaluator: FaithfulnessEvaluator 實例
    
    Returns:
        指標字典
    """
    # 建立臨時評估器
    evaluator = RAGEvaluator(eval_llm=correctness_evaluator.llm, metrics_mode="full")
    
    return await evaluator.compute_metrics_for_sample(
        idx=idx,
        source=source,
        query=query,
        gt_answer=gt_answer,
        gen_answer=gen_answer,
        gt_ids=gt_ids,
        retrieved_ids=retrieved_ids,
        retrieved_contexts=retrieved_contexts,
        execution_time_sec=execution_time_sec
    )
