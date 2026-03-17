"""
RAG 評估引擎

提供完整的 RAG Pipeline 評估功能
"""

import os
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
)


class RAGEvaluator:
    """
    RAG 評估器
    
    統籌評估流程，計算所有指標並生成報告
    
    Attributes:
        eval_llm: 用於 LLM-as-Judge 的語言模型
        base_eval_dir: 評估結果儲存目錄
    """
    
    def __init__(self, eval_llm, base_eval_dir: str = None):
        """
        初始化評估器
        
        Args:
            eval_llm: 用於 LLM-as-Judge 的語言模型
            base_eval_dir: 評估結果儲存目錄，預設為 results/evaluation_results_{timestamp}
        """
        self.eval_llm = eval_llm
        
        if base_eval_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_eval_dir = f"/home/End_to_End_RAG/results/evaluation_results_{timestamp}"
        
        self.base_eval_dir = base_eval_dir
        os.makedirs(self.base_eval_dir, exist_ok=True)
        
        # 初始化指標
        self._init_metrics()
    
    def _init_metrics(self):
        """初始化所有評估指標"""
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
        
        # LLM Judge 指標
        self.correctness_metric = CorrectnessMetric(llm=self.eval_llm)
        self.faithfulness_metric = FaithfulnessMetric(llm=self.eval_llm)
    
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
        execution_time_sec: float
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
        
        Returns:
            包含所有指標的字典
        """
        # 檢索指標
        hit_rate = self.hit_rate_metric.compute(retrieved_ids, gt_ids)
        mrr = self.mrr_metric.compute(retrieved_ids, gt_ids)
        recall, precision, f1_score = self.retrieval_f1_metric.compute(retrieved_ids, gt_ids)
        
        # 生成指標
        jieba_r, jieba_p, jieba_f1 = self.jieba_f1_metric.compute(gen_answer, gt_answer)
        rouge1, rouge2, rougeL, rougeLsum = self.rouge_metric.compute(gen_answer, gt_answer)
        bleu = self.bleu_metric.compute(gen_answer, gt_answer)
        meteor = self.meteor_metric.compute(gen_answer, gt_answer)
        bert_p, bert_r, bert_f1 = self.bertscore_metric.compute(gen_answer, gt_answer)
        tok_r, tok_p, tok_f1 = self.token_f1_metric.compute(gen_answer, gt_answer)
        
        # LLM-as-Judge 指標
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
        
        # 統整回傳
        return {
            "idx": idx,
            "dataset_source": source,
            "query": query,
            "ground_truth_answer": gt_answer,
            "generated_answer": gen_answer,
            "ground_truth_ids": ", ".join(gt_ids),
            "retrieved_ids": ", ".join(retrieved_ids),
            "execution_time_sec": execution_time_sec,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "retrieval_recall": recall,
            "retrieval_precision": precision,
            "retrieval_f1_score": f1_score,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "rougeLsum": rougeLsum,
            "bleu": bleu,
            "meteor": meteor,
            "bertscore_f1": bert_f1,
            "token_f1": tok_f1,
            "jieba_f1": jieba_f1,
            "correctness_score": correctness_score,
            "faithfulness_score": faithfulness_score,
        }
    
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
        
        results = []
        
        for idx, qa in tqdm(enumerate(qa_datasets), total=len(qa_datasets), desc=f"評估 {pipeline_name} 進度"):
            query = qa["query"]
            gt_answer = qa["ground_truth_answer"]
            gt_ids = qa["ground_truth_doc_ids"]
            
            # 處理 DI 資料集的特殊格式（gt_ids 是包含多行的單一字串）
            if len(gt_ids) > 0 and isinstance(gt_ids[0], str) and "\n" in gt_ids[0]:
                gt_ids = gt_ids[0].splitlines()
            
            try:
                # 執行檢索與生成
                result = await pipeline.aquery_and_log(query)
                
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
                    execution_time_sec=result["execution_time_sec"]
                )
                
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
    evaluator = RAGEvaluator(eval_llm=correctness_evaluator.llm)
    
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
