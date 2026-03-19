#!/usr/bin/env python3
"""
測試 RAGAS 指標整合

驗證新的 RAGAS 指標是否正確整合到評估系統中
"""

import sys
import asyncio
sys.path.insert(0, '/home/End_to_End_RAG')

from src.evaluation.metrics import (
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    RAGASFaithfulnessMetric,
)


def test_ragas_metrics_import():
    """測試 RAGAS 指標是否可以正常導入"""
    print("✅ 測試 1: RAGAS 指標導入")
    print(f"  - AnswerRelevancyMetric: {AnswerRelevancyMetric}")
    print(f"  - ContextPrecisionMetric: {ContextPrecisionMetric}")
    print(f"  - ContextRecallMetric: {ContextRecallMetric}")
    print(f"  - RAGASFaithfulnessMetric: {RAGASFaithfulnessMetric}")
    print()


def test_ragas_metrics_initialization():
    """測試 RAGAS 指標是否可以正常初始化"""
    print("✅ 測試 2: RAGAS 指標初始化")
    
    try:
        answer_relevancy = AnswerRelevancyMetric()
        print(f"  - AnswerRelevancyMetric 初始化: {'成功' if answer_relevancy else '失敗'}")
        print(f"    可用性: {answer_relevancy.available}")
        
        context_precision = ContextPrecisionMetric()
        print(f"  - ContextPrecisionMetric 初始化: {'成功' if context_precision else '失敗'}")
        print(f"    可用性: {context_precision.available}")
        
        context_recall = ContextRecallMetric()
        print(f"  - ContextRecallMetric 初始化: {'成功' if context_recall else '失敗'}")
        print(f"    可用性: {context_recall.available}")
        
        ragas_faithfulness = RAGASFaithfulnessMetric()
        print(f"  - RAGASFaithfulnessMetric 初始化: {'成功' if ragas_faithfulness else '失敗'}")
        print(f"    可用性: {ragas_faithfulness.available}")
        
    except Exception as e:
        print(f"  ❌ 初始化失敗: {e}")
    
    print()


async def test_ragas_metrics_computation():
    """測試 RAGAS 指標是否可以正常計算（需要 RAGAS 套件）"""
    print("✅ 測試 3: RAGAS 指標計算")
    
    # 測試資料
    query = "台灣的首都是哪裡？"
    response = "台灣的首都是台北市。"
    ground_truth = "台北市是台灣的首都。"
    contexts = ["台北市是中華民國的首都，也是台灣最大的城市。"]
    
    try:
        # 測試 AnswerRelevancy
        answer_relevancy = AnswerRelevancyMetric()
        if answer_relevancy.available:
            print("  - AnswerRelevancyMetric 計算中...")
            score = await answer_relevancy.compute_async(
                query=query,
                response=response,
                ground_truth=ground_truth
            )
            print(f"    結果: {score}")
        else:
            print("  - AnswerRelevancyMetric: RAGAS 套件未安裝，跳過")
        
        # 測試 ContextPrecision
        context_precision = ContextPrecisionMetric()
        if context_precision.available:
            print("  - ContextPrecisionMetric 計算中...")
            score = await context_precision.compute_async(
                query=query,
                contexts=contexts,
                ground_truth=ground_truth
            )
            print(f"    結果: {score}")
        else:
            print("  - ContextPrecisionMetric: RAGAS 套件未安裝，跳過")
        
        # 測試 ContextRecall
        context_recall = ContextRecallMetric()
        if context_recall.available:
            print("  - ContextRecallMetric 計算中...")
            score = await context_recall.compute_async(
                query=query,
                contexts=contexts,
                ground_truth=ground_truth
            )
            print(f"    結果: {score}")
        else:
            print("  - ContextRecallMetric: RAGAS 套件未安裝，跳過")
        
        # 測試 RAGASFaithfulness
        ragas_faithfulness = RAGASFaithfulnessMetric()
        if ragas_faithfulness.available:
            print("  - RAGASFaithfulnessMetric 計算中...")
            score = await ragas_faithfulness.compute_async(
                query=query,
                response=response,
                contexts=contexts
            )
            print(f"    結果: {score}")
        else:
            print("  - RAGASFaithfulnessMetric: RAGAS 套件未安裝，跳過")
        
    except Exception as e:
        print(f"  ❌ 計算失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_evaluator_integration():
    """測試 RAGEvaluator 是否正確整合 RAGAS 指標"""
    print("✅ 測試 4: RAGEvaluator 整合")
    
    try:
        from src.config.settings import get_settings
        from src.evaluation.evaluator import RAGEvaluator
        from llama_index.core import Settings
        
        # 載入配置以初始化 Settings.eval_llm
        try:
            get_settings()
            eval_llm = Settings.eval_llm
            print("  ✓ 成功載入配置與 eval_llm")
        except Exception as e:
            print(f"  ⚠️  無法載入配置: {e}")
            print("  使用預設 LLM 進行測試")
            eval_llm = Settings.llm
        
        # 建立評估器
        evaluator = RAGEvaluator(eval_llm=eval_llm)
        
        # 檢查 RAGAS 指標是否已初始化
        has_answer_relevancy = hasattr(evaluator, 'answer_relevancy_metric')
        has_context_precision = hasattr(evaluator, 'context_precision_metric')
        has_context_recall = hasattr(evaluator, 'context_recall_metric')
        has_ragas_faithfulness = hasattr(evaluator, 'ragas_faithfulness_metric')
        
        print(f"  - answer_relevancy_metric: {'✓' if has_answer_relevancy else '✗'}")
        print(f"  - context_precision_metric: {'✓' if has_context_precision else '✗'}")
        print(f"  - context_recall_metric: {'✓' if has_context_recall else '✗'}")
        print(f"  - ragas_faithfulness_metric: {'✓' if has_ragas_faithfulness else '✗'}")
        
        # 驗證指標屬性
        if has_answer_relevancy:
            print(f"    answer_relevancy_metric.name: {evaluator.answer_relevancy_metric.name}")
        if has_context_precision:
            print(f"    context_precision_metric.name: {evaluator.context_precision_metric.name}")
        if has_context_recall:
            print(f"    context_recall_metric.name: {evaluator.context_recall_metric.name}")
        if has_ragas_faithfulness:
            print(f"    ragas_faithfulness_metric.name: {evaluator.ragas_faithfulness_metric.name}")
        
        if all([has_answer_relevancy, has_context_precision, has_context_recall, has_ragas_faithfulness]):
            print("\n  ✅ 所有 RAGAS 指標已成功整合到 RAGEvaluator")
        else:
            print("\n  ❌ 部分 RAGAS 指標未正確整合")
        
    except Exception as e:
        print(f"  ❌ 整合測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print()


async def main():
    """執行所有測試"""
    print("=" * 70)
    print("RAGAS 指標整合測試")
    print("=" * 70)
    print()
    
    # 測試 1: 導入
    test_ragas_metrics_import()
    
    # 測試 2: 初始化
    test_ragas_metrics_initialization()
    
    # 測試 3: 計算（需要 RAGAS 套件與 API Key）
    await test_ragas_metrics_computation()
    
    # 測試 4: 整合
    test_evaluator_integration()
    
    print("=" * 70)
    print("測試完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
