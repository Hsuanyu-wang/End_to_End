"""
評估指標套件

提供所有評估指標的統一介面
"""

from .base import BaseMetric, MetricRegistry
from .retrieval import (
    HitRateMetric,
    MRRMetric,
    RetrievalF1Metric,
    calculate_hit_rate,
    calculate_mrr,
    calculate_f1_score,
)
from .generation import (
    ROUGEMetric,
    BLEUMetric,
    METEORMetric,
    BERTScoreMetric,
    TokenF1Metric,
    JiebaF1Metric,
    calculate_rouge_score,
    calculate_bleu_score,
    calculate_meteor_score,
    calculate_bertscore_score,
    token_level_f1_score,
    jieba_f1_score,
)
from .llm_judge import (
    CorrectnessMetric,
    FaithfulnessMetric,
)
from .ragas_metrics import (
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    RAGASFaithfulnessMetric,
)

__all__ = [
    # 基底類別
    "BaseMetric",
    "MetricRegistry",
    # 檢索指標
    "HitRateMetric",
    "MRRMetric",
    "RetrievalF1Metric",
    "calculate_hit_rate",
    "calculate_mrr",
    "calculate_f1_score",
    # 生成指標
    "ROUGEMetric",
    "BLEUMetric",
    "METEORMetric",
    "BERTScoreMetric",
    "TokenF1Metric",
    "JiebaF1Metric",
    "calculate_rouge_score",
    "calculate_bleu_score",
    "calculate_meteor_score",
    "calculate_bertscore_score",
    "token_level_f1_score",
    "jieba_f1_score",
    # LLM Judge 指標
    "CorrectnessMetric",
    "FaithfulnessMetric",
    # RAGAS 指標
    "AnswerRelevancyMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "RAGASFaithfulnessMetric",
]
