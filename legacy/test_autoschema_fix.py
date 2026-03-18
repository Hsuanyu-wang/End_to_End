#!/usr/bin/env python3
"""
測試 AutoSchemaKG 修正後的實作
"""
import sys
sys.path.insert(0, '/home/End_to_End_RAG')

from src.graph_builder.autoschema_builder import AutoSchemaKGBuilder
from src.data.processors import data_processing

def test_autoschema():
    print("=" * 80)
    print("測試 AutoSchemaKG 修正實作")
    print("=" * 80)
    
    # 1. 載入文檔 (使用少量文檔進行測試)
    print("\n📂 步驟 1: 載入測試文檔...")
    documents = data_processing(mode="natural_text", data_type="DI")
    test_docs = documents[:2]  # 只用 2 筆測試
    print(f"✅ 已載入 {len(test_docs)} 筆文檔進行測試")
    
    # 2. 初始化 Builder
    print("\n🔧 步驟 2: 初始化 AutoSchemaKG Builder...")
    builder = AutoSchemaKGBuilder(output_dir="/tmp/autoschema_test")
    builder.initialize({
        'model_name': 'llama3.3:latest',
        'batch_size_triple': 3,
        'batch_size_concept': 16,
        'max_workers': 3
    })
    
    # 3. 執行建圖
    print("\n🚀 步驟 3: 執行知識圖譜建立...")
    try:
        result = builder.build(test_docs)
        
        print("\n" + "=" * 80)
        print("✅ 測試成功！")
        print("=" * 80)
        print(f"\n結果摘要:")
        print(f"  - 節點數: {len(result['nodes'])}")
        print(f"  - 邊數: {len(result['edges'])}")
        print(f"  - Schema 資訊: {result['schema_info']}")
        print(f"  - 輸出目錄: {result['storage_path']}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 測試失敗！")
        print("=" * 80)
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_autoschema()
    sys.exit(0 if success else 1)
