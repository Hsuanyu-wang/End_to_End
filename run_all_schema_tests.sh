#!/bin/bash

# 測試所有 Schema 方法的評估腳本

echo "======================================================"
echo "開始測試所有 Schema 方法"
echo "======================================================"

# 測試 1: lightrag_default
echo ""
echo "📋 測試 1/3: lightrag_default Schema"
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_schema_method lightrag_default \
    --data_type DI \
    --qa_dataset_fast_test

if [ $? -eq 0 ]; then
    echo "✅ lightrag_default 測試成功"
else
    echo "❌ lightrag_default 測試失敗"
    exit 1
fi

# 測試 2: iterative_evolution
echo ""
echo "📋 測試 2/3: iterative_evolution Schema"
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_schema_method iterative_evolution \
    --data_type DI \
    --qa_dataset_fast_test

if [ $? -eq 0 ]; then
    echo "✅ iterative_evolution 測試成功"
else
    echo "❌ iterative_evolution 測試失敗"
    exit 1
fi

# 測試 3: llm_dynamic
echo ""
echo "📋 測試 3/3: llm_dynamic Schema"
python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --lightrag_schema_method llm_dynamic \
    --data_type DI \
    --qa_dataset_fast_test

if [ $? -eq 0 ]; then
    echo "✅ llm_dynamic 測試成功"
else
    echo "❌ llm_dynamic 測試失敗"
    exit 1
fi

echo ""
echo "======================================================"
echo "✅ 所有 Schema 方法測試完成！"
echo "======================================================"
echo ""
echo "📊 結果檔案位於："
echo "  /home/End_to_End_RAG/results/"
echo ""
echo "每個結果資料夾中的 detailed_results.csv 第一行都包含："
echo "  - schema_method: Schema 生成方法"
echo "  - schema_entities: 實體類型列表（JSON）"
echo "  - schema_relations: 關係類型列表（JSON）"
echo "  - schema_validation: 驗證規則（JSON）"
