#!/bin/bash
# 使用 retrieval_max_tokens 進行公平比較的示例腳本

echo "=================================================="
echo "RAG 方法公平比較 - 限制檢索內容 Token 數"
echo "=================================================="
echo ""

# 設定共同參數
DATA_TYPE="DI"
TOP_K=10
RETRIEVAL_MAX_TOKENS=2048  # 限制所有方法檢索內容為 2048 tokens

echo "📊 測試配置:"
echo "  - Data Type: ${DATA_TYPE}"
echo "  - Top K: ${TOP_K}"
echo "  - Retrieval Max Tokens: ${RETRIEVAL_MAX_TOKENS}"
echo ""

# 1. 測試 Vector RAG 方法
echo "=================================================="
echo "1. 測試 Vector RAG 方法 (hybrid, vector, bm25)"
echo "=================================================="

python scripts/run_evaluation.py \
    --vector_method hybrid \
    --top_k ${TOP_K} \
    --retrieval_max_tokens ${RETRIEVAL_MAX_TOKENS} \
    --data_type ${DATA_TYPE} \
    --qa_dataset_fast_test

python scripts/run_evaluation.py \
    --vector_method vector \
    --top_k ${TOP_K} \
    --retrieval_max_tokens ${RETRIEVAL_MAX_TOKENS} \
    --data_type ${DATA_TYPE} \
    --qa_dataset_fast_test

python scripts/run_evaluation.py \
    --vector_method bm25 \
    --top_k ${TOP_K} \
    --retrieval_max_tokens ${RETRIEVAL_MAX_TOKENS} \
    --data_type ${DATA_TYPE} \
    --qa_dataset_fast_test

# 2. 測試 LightRAG 方法
echo ""
echo "=================================================="
echo "2. 測試 LightRAG 方法 (hybrid, local, global)"
echo "=================================================="

python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode hybrid \
    --top_k ${TOP_K} \
    --retrieval_max_tokens ${RETRIEVAL_MAX_TOKENS} \
    --data_type ${DATA_TYPE} \
    --qa_dataset_fast_test

python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode local \
    --top_k ${TOP_K} \
    --retrieval_max_tokens ${RETRIEVAL_MAX_TOKENS} \
    --data_type ${DATA_TYPE} \
    --qa_dataset_fast_test

python scripts/run_evaluation.py \
    --graph_rag_method lightrag \
    --lightrag_mode global \
    --top_k ${TOP_K} \
    --retrieval_max_tokens ${RETRIEVAL_MAX_TOKENS} \
    --data_type ${DATA_TYPE} \
    --qa_dataset_fast_test

echo ""
echo "=================================================="
echo "✅ 所有測試完成！"
echo "=================================================="
echo ""
echo "結果位於: results/test/ 目錄"
echo ""
echo "注意: 所有方法都在相同的 retrieval token budget (${RETRIEVAL_MAX_TOKENS}) 下執行"
echo "      這確保了公平比較各方法的檢索品質"
