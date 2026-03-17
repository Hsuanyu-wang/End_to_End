"""
檢索評估指標

提供各種檢索品質評估指標，包括 Hit Rate, MRR, Precision, Recall, F1
"""

from typing import List, Tuple
from .base import BaseMetric


class HitRateMetric(BaseMetric):
    """
    Hit Rate (命中率) 指標
    
    定義：檢索結果中是否至少包含一個相關文件
    計算方式：如果檢索到的文件 ID 中有任何一個在 ground truth 中，則為 1，否則為 0
    
    範圍：[0, 1]
    """
    
    def __init__(self):
        super().__init__(
            name="hit_rate",
            description="檢索結果是否命中至少一個相關文件"
        )
    
    def compute(self, retrieved_ids: List[str], ground_truth_ids: List[str]) -> int:
        """
        計算 Hit Rate
        
        Args:
            retrieved_ids: 檢索到的文件 ID 列表
            ground_truth_ids: 標準答案的文件 ID 列表
        
        Returns:
            1 表示命中，0 表示未命中
        """
        if not ground_truth_ids:
            return 0
        
        return 1 if any(gt_id in retrieved_ids for gt_id in ground_truth_ids) else 0


class MRRMetric(BaseMetric):
    """
    Mean Reciprocal Rank (平均倒數排名) 指標
    
    定義：第一個相關文件在檢索結果中的排名倒數
    計算方式：1 / (第一個相關文件的排名位置)
    
    範圍：(0, 1]
    數值越高表示相關文件排名越前
    """
    
    def __init__(self):
        super().__init__(
            name="mrr",
            description="第一個相關文件的排名倒數"
        )
    
    def compute(self, retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """
        計算 MRR
        
        Args:
            retrieved_ids: 檢索到的文件 ID 列表（按排名順序）
            ground_truth_ids: 標準答案的文件 ID 列表
        
        Returns:
            MRR 分數
        """
        if not ground_truth_ids:
            return 0.0
        
        for i, r_id in enumerate(retrieved_ids):
            if r_id in ground_truth_ids:
                return 1.0 / (i + 1)
        
        return 0.0


class RetrievalF1Metric(BaseMetric):
    """
    檢索 F1 Score 指標
    
    定義：檢索結果的精準度與召回率的調和平均
    計算方式：
        - Recall = |檢索到的相關文件| / |所有相關文件|
        - Precision = |檢索到的相關文件| / |所有檢索文件|
        - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    範圍：[0, 1]
    """
    
    def __init__(self):
        super().__init__(
            name="retrieval_f1",
            description="檢索結果的精準度與召回率的調和平均"
        )
    
    def compute(self, retrieved_ids: List[str], ground_truth_ids: List[str]) -> Tuple[float, float, float]:
        """
        計算 Recall, Precision, F1 Score
        
        Args:
            retrieved_ids: 檢索到的文件 ID 列表
            ground_truth_ids: 標準答案的文件 ID 列表
        
        Returns:
            (recall, precision, f1_score) 三元組
        """
        if len(ground_truth_ids) == 0 or len(retrieved_ids) == 0:
            return 0.0, 0.0, 0.0
        
        # 計算交集
        relevant_retrieved = set(retrieved_ids) & set(ground_truth_ids)
        
        # Recall: 檢索到的相關文件數 / 所有相關文件數
        recall = float(len(relevant_retrieved) / len(ground_truth_ids))
        
        # Precision: 檢索到的相關文件數 / 所有檢索文件數
        precision = float(len(relevant_retrieved) / len(retrieved_ids))
        
        # F1 Score
        if (recall + precision) > 0:
            f1_score = 2 * recall * precision / (recall + precision)
        else:
            f1_score = 0.0
        
        return recall, precision, f1_score


def calculate_hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str]) -> int:
    """
    計算 Hit Rate（向後兼容函數）
    
    Args:
        retrieved_ids: 檢索到的文件 ID 列表
        ground_truth_ids: 標準答案的文件 ID 列表
    
    Returns:
        1 表示命中，0 表示未命中
    """
    metric = HitRateMetric()
    return metric.compute(retrieved_ids, ground_truth_ids)


def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """
    計算 MRR（向後兼容函數）
    
    Args:
        retrieved_ids: 檢索到的文件 ID 列表
        ground_truth_ids: 標準答案的文件 ID 列表
    
    Returns:
        MRR 分數
    """
    metric = MRRMetric()
    return metric.compute(retrieved_ids, ground_truth_ids)


def calculate_f1_score(retrieved_ids: List[str], ground_truth_ids: List[str]) -> Tuple[float, float, float]:
    """
    計算 Recall, Precision, F1 Score（向後兼容函數）
    
    Args:
        retrieved_ids: 檢索到的文件 ID 列表
        ground_truth_ids: 標準答案的文件 ID 列表
    
    Returns:
        (recall, precision, f1_score) 三元組
    """
    metric = RetrievalF1Metric()
    return metric.compute(retrieved_ids, ground_truth_ids)
