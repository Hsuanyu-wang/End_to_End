#!/usr/bin/env python3
"""
快速驗證腳本

驗證重構後的專案結構是否正常運作
"""

import sys
import os

def test_imports():
    """測試所有主要模組是否可以正常 import"""
    print("測試模組 import...")
    
    try:
        # 配置
        from src.config import get_settings, ModelSettings
        print("✅ src.config")
        
        # 資料處理
        from src.data import DataProcessor, QADataLoader, data_processing
        print("✅ src.data")
        
        # 評估指標
        from src.evaluation.metrics import (
            HitRateMetric, MRRMetric, RetrievalF1Metric,
            ROUGEMetric, BLEUMetric, JiebaF1Metric,
            CorrectnessMetric, FaithfulnessMetric
        )
        print("✅ src.evaluation.metrics")
        
        # RAG Wrappers
        from src.rag.wrappers import (
            BaseRAGWrapper, VectorRAGWrapper,
            LightRAGWrapper, TemporalLightRAGWrapper
        )
        print("✅ src.rag.wrappers")
        
        # 評估引擎
        from src.evaluation import RAGEvaluator, EvaluationReporter
        print("✅ src.evaluation")
        
        print("\n所有模組 import 成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ Import 失敗: {e}")
        return False


def test_metric_computation():
    """測試指標計算"""
    print("\n測試指標計算...")
    
    try:
        from src.evaluation.metrics import HitRateMetric, JiebaF1Metric
        
        # Hit Rate
        hit_rate = HitRateMetric()
        result = hit_rate.compute(
            retrieved_ids=["doc1", "doc2"],
            ground_truth_ids=["doc1"]
        )
        assert result == 1, "Hit Rate 計算錯誤"
        print("✅ Hit Rate 計算正常")
        
        # Jieba F1
        jieba = JiebaF1Metric()
        r, p, f1 = jieba.compute(
            generated_answer="我喜歡吃蘋果",
            ground_truth_answer="我喜歡吃蘋果"
        )
        assert f1 == 1.0, "Jieba F1 計算錯誤"
        print("✅ Jieba F1 計算正常")
        
        print("\n指標計算測試通過！")
        return True
        
    except Exception as e:
        print(f"\n❌ 指標計算測試失敗: {e}")
        return False


def test_structure():
    """測試專案結構"""
    print("\n檢查專案結構...")
    
    required_dirs = [
        "src",
        "src/config",
        "src/data",
        "src/rag",
        "src/rag/wrappers",
        "src/rag/vector",
        "src/rag/graph",
        "src/evaluation",
        "src/evaluation/metrics",
        "scripts",
        "tests",
        "docs",
        "legacy",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = f"/home/End_to_End_RAG/{dir_path}"
        if os.path.exists(full_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} 不存在")
            all_exist = False
    
    if all_exist:
        print("\n專案結構完整！")
    else:
        print("\n專案結構不完整")
    
    return all_exist


def main():
    """主函數"""
    print("="*50)
    print("End-to-End RAG 重構驗證")
    print("="*50)
    
    # 測試結構
    structure_ok = test_structure()
    
    # 測試 import
    import_ok = test_imports()
    
    # 測試指標計算
    metrics_ok = test_metric_computation()
    
    # 總結
    print("\n" + "="*50)
    print("驗證結果總結")
    print("="*50)
    print(f"專案結構: {'✅ 通過' if structure_ok else '❌ 失敗'}")
    print(f"模組 Import: {'✅ 通過' if import_ok else '❌ 失敗'}")
    print(f"指標計算: {'✅ 通過' if metrics_ok else '❌ 失敗'}")
    
    if structure_ok and import_ok and metrics_ok:
        print("\n🎉 所有驗證通過！專案重構成功！")
        return 0
    else:
        print("\n⚠️ 部分驗證失敗，請檢查相關模組")
        return 1


if __name__ == "__main__":
    sys.exit(main())
