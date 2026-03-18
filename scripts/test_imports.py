#!/usr/bin/env python3
"""
模組導入測試腳本

驗證所有新實作的模組是否可以正常導入
"""

import sys
import os

# 加入專案路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """測試所有新模組的導入"""
    print("🧪 開始測試模組導入...")
    print()
    
    tests = []
    
    # 1. Schema Cache
    try:
        from src.rag.schema.schema_cache import SchemaCacheManager, SchemaCacheKey
        print("✅ Schema Cache 模組導入成功")
        tests.append(("Schema Cache", True))
    except Exception as e:
        print(f"❌ Schema Cache 模組導入失敗: {e}")
        tests.append(("Schema Cache", False))
    
    # 2. Retrieval Pipeline
    try:
        from src.rag.retrieval.retrieval_pipeline import (
            RetrievalPipeline, 
            RetrievalContext,
            RetrievalComponent,
            BaseRetriever,
            EntityDisambiguationComponent,
            RerankerComponent,
            ToGIterativeRetriever
        )
        print("✅ Retrieval Pipeline 模組導入成功")
        tests.append(("Retrieval Pipeline", True))
    except Exception as e:
        print(f"❌ Retrieval Pipeline 模組導入失敗: {e}")
        tests.append(("Retrieval Pipeline", False))
    
    # 3. Reranker
    try:
        from src.evaluation.reranker import CrossEncoderReranker
        print("✅ Reranker 模組導入成功")
        tests.append(("Reranker", True))
    except Exception as e:
        print(f"❌ Reranker 模組導入失敗: {e}")
        tests.append(("Reranker", False))
    
    # 4. Entity Disambiguation
    try:
        from src.rag.schema.entity_disambiguation import EntityDisambiguator
        print("✅ Entity Disambiguation 模組導入成功")
        tests.append(("Entity Disambiguation", True))
    except Exception as e:
        print(f"❌ Entity Disambiguation 模組導入失敗: {e}")
        tests.append(("Entity Disambiguation", False))
    
    # 5. Schema Convergence
    try:
        from src.rag.schema.convergence import SchemaQualityMetrics, should_stop_schema_evolution
        print("✅ Schema Convergence 模組導入成功")
        tests.append(("Schema Convergence", True))
    except Exception as e:
        print(f"❌ Schema Convergence 模組導入失敗: {e}")
        tests.append(("Schema Convergence", False))
    
    # 6. ToG Retriever
    try:
        from src.graph_retriever.tog_retriever import ToGRetriever
        print("✅ ToG Retriever 模組導入成功")
        tests.append(("ToG Retriever", True))
    except Exception as e:
        print(f"❌ ToG Retriever 模組導入失敗: {e}")
        tests.append(("ToG Retriever", False))
    
    # 7. Adaptive Router
    try:
        from src.graph_retriever.adaptive_router import QueryRouter, QueryComplexity, RetrievalMode
        print("✅ Adaptive Router 模組導入成功")
        tests.append(("Adaptive Router", True))
    except Exception as e:
        print(f"❌ Adaptive Router 模組導入失敗: {e}")
        tests.append(("Adaptive Router", False))
    
    # 8. Updated Factory (with cache support)
    try:
        from src.rag.schema.factory import get_schema_by_method
        print("✅ Schema Factory (updated) 模組導入成功")
        tests.append(("Schema Factory", True))
    except Exception as e:
        print(f"❌ Schema Factory 模組導入失敗: {e}")
        tests.append(("Schema Factory", False))
    
    print()
    print("=" * 60)
    print("測試結果摘要:")
    print("=" * 60)
    
    success_count = sum(1 for _, success in tests if success)
    total_count = len(tests)
    
    for name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print()
    print(f"總計: {success_count}/{total_count} 通過")
    
    if success_count == total_count:
        print()
        print("🎉 所有模組導入測試通過！")
        return 0
    else:
        print()
        print("⚠️  部分模組導入失敗，請檢查錯誤訊息")
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())
