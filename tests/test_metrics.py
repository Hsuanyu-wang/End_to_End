"""
評估指標測試

測試各種評估指標的計算正確性
"""

import pytest
from src.evaluation.metrics import (
    HitRateMetric,
    MRRMetric,
    RetrievalF1Metric,
    TokenF1Metric,
    JiebaF1Metric,
)


class TestRetrievalMetrics:
    """檢索指標測試"""
    
    def test_hit_rate_hit(self):
        """測試命中情況"""
        metric = HitRateMetric()
        result = metric.compute(
            retrieved_ids=["doc1", "doc2", "doc3"],
            ground_truth_ids=["doc1", "doc4"]
        )
        assert result == 1
    
    def test_hit_rate_miss(self):
        """測試未命中情況"""
        metric = HitRateMetric()
        result = metric.compute(
            retrieved_ids=["doc2", "doc3"],
            ground_truth_ids=["doc1", "doc4"]
        )
        assert result == 0
    
    def test_hit_rate_empty_gt(self):
        """測試空的 ground truth"""
        metric = HitRateMetric()
        result = metric.compute(
            retrieved_ids=["doc1", "doc2"],
            ground_truth_ids=[]
        )
        assert result == 0
    
    def test_mrr_first_position(self):
        """測試第一位命中"""
        metric = MRRMetric()
        result = metric.compute(
            retrieved_ids=["doc1", "doc2", "doc3"],
            ground_truth_ids=["doc1"]
        )
        assert result == 1.0
    
    def test_mrr_second_position(self):
        """測試第二位命中"""
        metric = MRRMetric()
        result = metric.compute(
            retrieved_ids=["doc2", "doc1", "doc3"],
            ground_truth_ids=["doc1"]
        )
        assert result == 0.5
    
    def test_mrr_no_hit(self):
        """測試未命中"""
        metric = MRRMetric()
        result = metric.compute(
            retrieved_ids=["doc2", "doc3"],
            ground_truth_ids=["doc1"]
        )
        assert result == 0.0
    
    def test_retrieval_f1(self):
        """測試 F1 計算"""
        metric = RetrievalF1Metric()
        recall, precision, f1 = metric.compute(
            retrieved_ids=["doc1", "doc2", "doc3"],
            ground_truth_ids=["doc1", "doc4"]
        )
        
        # Recall = 1/2 = 0.5 (檢索到 doc1，漏了 doc4)
        # Precision = 1/3 = 0.333... (doc1 相關，doc2、doc3 不相關)
        # F1 = 2 * 0.5 * 0.333 / (0.5 + 0.333) = 0.4
        
        assert recall == pytest.approx(0.5, rel=1e-2)
        assert precision == pytest.approx(0.333, rel=1e-2)
        assert f1 == pytest.approx(0.4, rel=1e-2)
    
    def test_retrieval_f1_perfect(self):
        """測試完美檢索"""
        metric = RetrievalF1Metric()
        recall, precision, f1 = metric.compute(
            retrieved_ids=["doc1", "doc2"],
            ground_truth_ids=["doc1", "doc2"]
        )
        
        assert recall == 1.0
        assert precision == 1.0
        assert f1 == 1.0
    
    def test_retrieval_f1_empty(self):
        """測試空輸入"""
        metric = RetrievalF1Metric()
        recall, precision, f1 = metric.compute(
            retrieved_ids=[],
            ground_truth_ids=["doc1"]
        )
        
        assert recall == 0.0
        assert precision == 0.0
        assert f1 == 0.0


class TestGenerationMetrics:
    """生成指標測試"""
    
    def test_token_f1_identical(self):
        """測試完全相同的文本"""
        metric = TokenF1Metric()
        recall, precision, f1 = metric.compute(
            generated_answer="測試文本",
            ground_truth_answer="測試文本"
        )
        
        assert recall == 1.0
        assert precision == 1.0
        assert f1 == 1.0
    
    def test_token_f1_partial(self):
        """測試部分重疊"""
        metric = TokenF1Metric()
        recall, precision, f1 = metric.compute(
            generated_answer="測試",
            ground_truth_answer="測試文本"
        )
        
        # 共同字元: 測, 試 (2個)
        # Recall = 2/4 = 0.5
        # Precision = 2/2 = 1.0
        # F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 0.667
        
        assert recall == pytest.approx(0.5, rel=1e-2)
        assert precision == pytest.approx(1.0, rel=1e-2)
        assert f1 == pytest.approx(0.667, rel=1e-2)
    
    def test_token_f1_empty(self):
        """測試空文本"""
        metric = TokenF1Metric()
        recall, precision, f1 = metric.compute(
            generated_answer="",
            ground_truth_answer="測試"
        )
        
        assert recall == 0.0
        assert precision == 0.0
        assert f1 == 0.0
    
    def test_jieba_f1_identical(self):
        """測試 Jieba F1 - 完全相同"""
        metric = JiebaF1Metric()
        recall, precision, f1 = metric.compute(
            generated_answer="我喜歡吃蘋果",
            ground_truth_answer="我喜歡吃蘋果"
        )
        
        assert recall == 1.0
        assert precision == 1.0
        assert f1 == 1.0
    
    def test_jieba_f1_partial(self):
        """測試 Jieba F1 - 部分重疊"""
        metric = JiebaF1Metric()
        recall, precision, f1 = metric.compute(
            generated_answer="我喜歡吃蘋果",
            ground_truth_answer="我愛吃蘋果"
        )
        
        # 實際分詞結果:
        # generated: ["我", "喜歡", "吃", "蘋果"] (4個詞)
        # ground_truth: ["我愛吃", "蘋果"] (2個詞)
        # 共同詞: ["蘋果"] (1個)
        
        # Recall = 1/2 = 0.5
        # Precision = 1/4 = 0.25
        # F1 = 2 * 0.5 * 0.25 / (0.5 + 0.25) = 0.333
        
        assert recall == pytest.approx(0.5, rel=1e-1)
        assert precision == pytest.approx(0.25, rel=1e-1)
        assert f1 == pytest.approx(0.333, rel=1e-1)


class TestMetricEdgeCases:
    """邊界情況測試"""
    
    def test_all_empty(self):
        """測試所有輸入為空"""
        hit_rate = HitRateMetric()
        assert hit_rate.compute([], []) == 0
        
        mrr = MRRMetric()
        assert mrr.compute([], []) == 0.0
        
        f1 = RetrievalF1Metric()
        r, p, f = f1.compute([], [])
        assert r == 0.0 and p == 0.0 and f == 0.0
    
    def test_unicode_handling(self):
        """測試 Unicode 字元處理"""
        metric = TokenF1Metric()
        recall, precision, f1 = metric.compute(
            generated_answer="Hello 世界",
            ground_truth_answer="Hello 世界"
        )
        
        assert f1 == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
