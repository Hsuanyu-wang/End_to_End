#!/usr/bin/env python3
"""
Token Budget功能測試腳本

測試新實現的token計數和budget控制功能
"""

import sys
sys.path.insert(0, '/home/End_to_End_RAG')

from src.utils.token_counter import TokenCounter, count_tokens
from src.rag.token_budget_controller import TokenBudgetController


def test_token_counter():
    """測試TokenCounter"""
    print("="*70)
    print("測試1: TokenCounter基本功能")
    print("="*70)
    
    counter = TokenCounter()
    
    # 測試單一文本
    text1 = "這是一個測試文本，用於計算token數量。"
    tokens1 = counter.count_tokens(text1)
    print(f"文本: {text1}")
    print(f"Token數: {tokens1}\n")
    
    # 測試批次計算
    texts = [
        "第一個文檔內容",
        "第二個文檔內容，包含更多文字",
        "第三個文檔，測試批次計算功能"
    ]
    total_tokens = counter.count_tokens_batch(texts)
    print(f"批次計算 {len(texts)} 個文本:")
    print(f"總Token數: {total_tokens}\n")
    
    # 測試詳細資訊
    details = counter.count_tokens_with_details(texts)
    print("詳細統計:")
    print(f"  總tokens: {details['total_tokens']}")
    print(f"  平均每文本: {details['avg_tokens_per_text']:.2f}")
    print(f"  範圍: [{details['min_tokens']}, {details['max_tokens']}]")
    
    print("✅ TokenCounter測試通過\n")


def test_token_budget_controller():
    """測試TokenBudgetController"""
    print("="*70)
    print("測試2: TokenBudgetController")
    print("="*70)
    
    controller = TokenBudgetController()
    
    # 模擬Vector RAG的token使用
    vector_tokens = [1523, 1487, 1612, 1445, 1589, 1502, 1556, 1478]
    controller.set_baseline("Vector_Hybrid", vector_tokens)
    
    print(f"\nBaseline設定完成")
    print(f"目標Token Budget: {controller.get_target_tokens()}\n")
    
    # 測試LightRAG參數調整
    print("建議的LightRAG參數:")
    for mode in ["hybrid", "local", "global"]:
        params = controller.adjust_lightrag_params(mode=mode)
        print(f"\n{mode.upper()}模式:")
        print(f"  max_total_tokens: {params['max_total_tokens']}")
        print(f"  max_entity_tokens: {params['max_entity_tokens']}")
        print(f"  max_relation_tokens: {params['max_relation_tokens']}")
        print(f"  chunk_top_k: {params['chunk_top_k']}")
    
    # 添加其他方法的統計
    lightrag_tokens = [1512, 1489, 1587, 1476, 1523]
    controller.add_method_stats("LightRAG_Hybrid", lightrag_tokens)
    
    # 生成報告
    report = controller.generate_report()
    print("\n" + report)
    
    print("✅ TokenBudgetController測試通過\n")


def test_convenience_functions():
    """測試便捷函數"""
    print("="*70)
    print("測試3: 便捷函數")
    print("="*70)
    
    text = "測試便捷函數的文本內容"
    tokens = count_tokens(text)
    print(f"使用便捷函數 count_tokens():")
    print(f"文本: {text}")
    print(f"Token數: {tokens}")
    
    print("✅ 便捷函數測試通過\n")


def main():
    """執行所有測試"""
    print("\n" + "="*70)
    print("Token Budget功能測試套件")
    print("="*70 + "\n")
    
    try:
        test_token_counter()
        test_token_budget_controller()
        test_convenience_functions()
        
        print("="*70)
        print("✅ 所有測試通過！")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
