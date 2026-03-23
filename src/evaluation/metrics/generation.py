"""
生成評估指標

提供各種文本生成品質評估指標，包括 ROUGE, BLEU, METEOR, BERTScore, Token F1, Jieba F1
所有 n-gram 類指標（ROUGE / BLEU / METEOR）皆使用 jieba 分詞，適用於中英混合文本。
"""

import re
import jieba
from typing import Tuple
from collections import Counter
import evaluate
from rouge_score import rouge_scorer
from .base import BaseMetric


def _jieba_tokenize(text: str) -> str:
    """以 jieba 分詞後用空格連接，供 BLEU / METEOR 等 whitespace-based 指標使用。"""
    if not text:
        return ""
    return " ".join(jieba.lcut(str(text)))


class _JiebaTokenizer:
    """供 rouge_score.RougeScorer 使用的自訂 tokenizer。"""
    def tokenize(self, text: str):
        return jieba.lcut(str(text))


class ROUGEMetric(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 指標

    使用 rouge_score 搭配 jieba tokenizer，正確處理中文 n-gram。

    變體：
        - ROUGE-1: unigram 重疊
        - ROUGE-2: bigram 重疊
        - ROUGE-L: 最長公共子序列
        - ROUGE-Lsum: 用於多句摘要

    範圍：[0, 1]
    """

    def __init__(self):
        super().__init__(
            name="rouge",
            description="評估 n-gram 重疊程度（jieba 分詞）"
        )
        self._tokenizer = _JiebaTokenizer()
        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            tokenizer=self._tokenizer,
        )

    def compute(self, generated_answer: str, ground_truth_answer: str) -> Tuple[float, float, float, float]:
        """
        計算 ROUGE 分數

        Args:
            generated_answer: 生成的答案
            ground_truth_answer: 標準答案

        Returns:
            (rouge1, rouge2, rougeL, rougeLsum) 四元組（取 fmeasure）
        """
        if not generated_answer or not ground_truth_answer:
            return 0.0, 0.0, 0.0, 0.0

        scores = self._scorer.score(
            target=str(ground_truth_answer),
            prediction=str(generated_answer),
        )

        return (
            scores["rouge1"].fmeasure,
            scores["rouge2"].fmeasure,
            scores["rougeL"].fmeasure,
            scores["rougeLsum"].fmeasure,
        )


class BLEUMetric(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) 指標

    使用 jieba 分詞後再送入 evaluate，適用於中英混合文本。

    範圍：[0, 1]
    """

    def __init__(self):
        super().__init__(
            name="bleu",
            description="評估精準匹配程度（jieba 分詞）"
        )
        self.metric = evaluate.load("bleu")

    def compute(self, generated_answer: str, ground_truth_answer: str) -> float:
        """
        計算 BLEU 分數

        Args:
            generated_answer: 生成的答案
            ground_truth_answer: 標準答案

        Returns:
            BLEU 分數
        """
        if not generated_answer or not ground_truth_answer:
            return 0.0

        pred_tok = _jieba_tokenize(generated_answer)
        ref_tok = _jieba_tokenize(ground_truth_answer)

        if not pred_tok or not ref_tok:
            return 0.0

        score = self.metric.compute(
            predictions=[pred_tok],
            references=[ref_tok]
        )

        return score.get("bleu", 0.0)


class METEORMetric(BaseMetric):
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) 指標

    使用 jieba 分詞後再送入 evaluate，適用於中英混合文本。

    範圍：[0, 1]
    """

    def __init__(self):
        super().__init__(
            name="meteor",
            description="結合精準度與召回率的評估（jieba 分詞）"
        )
        self.metric = evaluate.load("meteor")

    def compute(self, generated_answer: str, ground_truth_answer: str) -> float:
        """
        計算 METEOR 分數

        Args:
            generated_answer: 生成的答案
            ground_truth_answer: 標準答案

        Returns:
            METEOR 分數
        """
        if not generated_answer or not ground_truth_answer:
            return 0.0

        pred_tok = _jieba_tokenize(generated_answer)
        ref_tok = _jieba_tokenize(ground_truth_answer)

        if not pred_tok or not ref_tok:
            return 0.0

        score = self.metric.compute(
            predictions=[pred_tok],
            references=[ref_tok]
        )

        return score.get("meteor", 0.0)


class BERTScoreMetric(BaseMetric):
    """
    BERTScore 指標
    
    定義：使用 BERT embeddings 計算語義相似度
    特點：考慮語義而非僅字面匹配
    
    範圍：[0, 1]
    """
    
    def __init__(self, lang: str = "zh"):
        super().__init__(
            name="bertscore",
            description="基於 BERT embeddings 的語義相似度評估"
        )
        self.metric = evaluate.load("bertscore")
        self.lang = lang
    
    def compute(self, generated_answer: str, ground_truth_answer: str) -> Tuple[float, float, float]:
        """
        計算 BERTScore
        
        Args:
            generated_answer: 生成的答案
            ground_truth_answer: 標準答案
        
        Returns:
            (precision, recall, f1) 三元組
        """
        if not generated_answer or not ground_truth_answer:
            return 0.0, 0.0, 0.0
        
        score = self.metric.compute(
            predictions=[generated_answer],
            references=[ground_truth_answer],
            lang=self.lang
        )
        
        return (
            score["precision"][0],
            score["recall"][0],
            score["f1"][0]
        )


class TokenF1Metric(BaseMetric):
    """
    Token-level F1 Score 指標
    
    定義：基於字元集合的精準度與召回率
    特點：相容中英混合文本
    
    範圍：[0, 1]
    """
    
    def __init__(self):
        super().__init__(
            name="token_f1",
            description="字元層級的 F1 Score"
        )
    
    def compute(self, generated_answer: str, ground_truth_answer: str) -> Tuple[float, float, float]:
        """
        計算 Token-level F1 Score
        
        Args:
            generated_answer: 生成的答案
            ground_truth_answer: 標準答案
        
        Returns:
            (recall, precision, f1) 三元組
        """
        gen_tokens = set(generated_answer)
        gt_tokens = set(ground_truth_answer)
        
        if not gen_tokens or not gt_tokens:
            return 0.0, 0.0, 0.0
        
        common_tokens = gen_tokens.intersection(gt_tokens)
        
        recall = len(common_tokens) / len(gt_tokens)
        precision = len(common_tokens) / len(gen_tokens)
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return recall, precision, f1


class JiebaF1Metric(BaseMetric):
    """
    Jieba Word-level F1 Score 指標
    
    定義：使用 jieba 分詞後計算詞彙層級的 F1 Score
    特點：適用於中英混合文本，考慮詞彙而非字元
    
    範圍：[0, 1]
    """
    
    def __init__(self):
        super().__init__(
            name="jieba_f1",
            description="基於 jieba 分詞的 Word-level F1 Score"
        )
    
    def _normalize_and_tokenize(self, text: str) -> list:
        """
        使用 jieba 進行分詞與正規化
        
        Args:
            text: 輸入文本
        
        Returns:
            詞彙列表
        """
        if not text:
            return []
        
        text = str(text).lower()
        raw_tokens = jieba.lcut(text)
        
        # 過濾純標點符號
        valid_tokens = [
            token for token in raw_tokens 
            if re.search(r'[a-z0-9\u4e00-\u9fa5]', token)
        ]
        
        return valid_tokens
    
    def compute(self, generated_answer: str, ground_truth_answer: str) -> Tuple[float, float, float]:
        """
        計算 Jieba F1 Score
        
        Args:
            generated_answer: 生成的答案
            ground_truth_answer: 標準答案
        
        Returns:
            (recall, precision, f1) 三元組
        """
        gen_tokens = self._normalize_and_tokenize(generated_answer)
        gt_tokens = self._normalize_and_tokenize(ground_truth_answer)
        
        if not gen_tokens and not gt_tokens:
            return 1.0, 1.0, 1.0
        
        if not gen_tokens or not gt_tokens:
            return 0.0, 0.0, 0.0
        
        gen_counter = Counter(gen_tokens)
        gt_counter = Counter(gt_tokens)
        
        # Multiset Intersection
        common_tokens_count = sum((gen_counter & gt_counter).values())
        
        if common_tokens_count == 0:
            return 0.0, 0.0, 0.0
        
        recall = common_tokens_count / len(gt_tokens)
        precision = common_tokens_count / len(gen_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return recall, precision, f1


# 向後兼容函數
def calculate_rouge_score(generated_answer: str, ground_truth_answer: str) -> Tuple[float, float, float, float]:
    """計算 ROUGE 分數（向後兼容）"""
    metric = ROUGEMetric()
    return metric.compute(generated_answer, ground_truth_answer)


def calculate_bleu_score(generated_answer: str, ground_truth_answer: str) -> float:
    """計算 BLEU 分數（向後兼容）"""
    metric = BLEUMetric()
    return metric.compute(generated_answer, ground_truth_answer)


def calculate_meteor_score(generated_answer: str, ground_truth_answer: str) -> float:
    """計算 METEOR 分數（向後兼容）"""
    metric = METEORMetric()
    return metric.compute(generated_answer, ground_truth_answer)


def calculate_bertscore_score(generated_answer: str, ground_truth_answer: str, lang: str = "zh") -> Tuple[float, float, float]:
    """計算 BERTScore（向後兼容）"""
    metric = BERTScoreMetric(lang=lang)
    return metric.compute(generated_answer, ground_truth_answer)


def token_level_f1_score(generated_answer: str, ground_truth_answer: str) -> Tuple[float, float, float]:
    """計算 Token-level F1（向後兼容）"""
    metric = TokenF1Metric()
    return metric.compute(generated_answer, ground_truth_answer)


def jieba_f1_score(generated_answer: str, ground_truth_answer: str) -> Tuple[float, float, float]:
    """計算 Jieba F1 Score（向後兼容）"""
    metric = JiebaF1Metric()
    return metric.compute(generated_answer, ground_truth_answer)
