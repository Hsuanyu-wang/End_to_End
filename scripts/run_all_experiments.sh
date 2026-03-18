#!/bin/bash

# 完整 RAG 實驗批次執行腳本
# 執行所有 RAG 方法的完整評估（不使用 fast_test）

set -e  # 遇到錯誤立即退出

SCRIPT_DIR="/home/End_to_End_RAG"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/experiment_logs"
DATA_TYPE="DI"  # 可改為 GEN

# 建立日誌目錄
mkdir -p "$LOG_DIR"

# 取得時間戳記
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/experiment_${TIMESTAMP}.log"

echo "======================================================"
echo "開始完整 RAG 實驗"
echo "======================================================"
echo "開始時間: $(date)"
echo "資料類型: $DATA_TYPE"
echo "日誌檔案: $LOG_FILE"
echo "======================================================"
echo

# 記錄函數
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 執行測試函數
run_test() {
    local test_name=$1
    local args=$2
    
    log "開始測試: $test_name"
    log "參數: $args"
    
    local start_time=$(date +%s)
    
    if python scripts/run_evaluation.py $args --data_type "$DATA_TYPE" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        log "✅ $test_name 完成 (耗時: ${elapsed}秒)"
        return 0
    else
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        log "❌ $test_name 失敗 (耗時: ${elapsed}秒)"
        return 1
    fi
}

# 記錄測試結果
declare -a RESULTS
SUCCESS_COUNT=0
FAILURE_COUNT=0

# ====== Vector RAG 測試 ======
log "====== Vector RAG 測試 ======"

if run_test "Vector Hybrid RAG" "--vector_method hybrid --top_k 2"; then
    RESULTS+=("✅ Vector Hybrid RAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ Vector Hybrid RAG")
    ((FAILURE_COUNT++))
fi

if run_test "Vector Only RAG" "--vector_method vector --top_k 2"; then
    RESULTS+=("✅ Vector Only RAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ Vector Only RAG")
    ((FAILURE_COUNT++))
fi

if run_test "BM25 RAG" "--vector_method bm25 --top_k 2"; then
    RESULTS+=("✅ BM25 RAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ BM25 RAG")
    ((FAILURE_COUNT++))
fi

# ====== Advanced Vector RAG 測試 ======
log "====== Advanced Vector RAG 測試 ======"

if run_test "Parent-Child RAG" "--adv_vector_method parent_child --top_k 2"; then
    RESULTS+=("✅ Parent-Child RAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ Parent-Child RAG")
    ((FAILURE_COUNT++))
fi

if run_test "Self-Query RAG" "--adv_vector_method self_query --top_k 2"; then
    RESULTS+=("✅ Self-Query RAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ Self-Query RAG")
    ((FAILURE_COUNT++))
fi

# ====== Graph RAG 測試 ======
log "====== Graph RAG 測試 ======"

if run_test "Property Graph RAG" "--graph_rag_method propertyindex --top_k 2"; then
    RESULTS+=("✅ Property Graph RAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ Property Graph RAG")
    ((FAILURE_COUNT++))
fi

if run_test "Dynamic Schema Graph RAG" "--graph_rag_method dynamic_schema --top_k 2"; then
    RESULTS+=("✅ Dynamic Schema Graph RAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ Dynamic Schema Graph RAG")
    ((FAILURE_COUNT++))
fi

# ====== LightRAG Schema 方法測試 ======
log "====== LightRAG Schema 方法測試 ======"

if run_test "LightRAG (Default Schema)" "--graph_rag_method lightrag --lightrag_mode hybrid --lightrag_schema_method lightrag_default"; then
    RESULTS+=("✅ LightRAG (Default Schema)")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG (Default Schema)")
    ((FAILURE_COUNT++))
fi

if run_test "LightRAG (Iterative Evolution)" "--graph_rag_method lightrag --lightrag_mode hybrid --lightrag_schema_method iterative_evolution"; then
    RESULTS+=("✅ LightRAG (Iterative Evolution)")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG (Iterative Evolution)")
    ((FAILURE_COUNT++))
fi

if run_test "LightRAG (LLM Dynamic)" "--graph_rag_method lightrag --lightrag_mode hybrid --lightrag_schema_method llm_dynamic"; then
    RESULTS+=("✅ LightRAG (LLM Dynamic)")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG (LLM Dynamic)")
    ((FAILURE_COUNT++))
fi

# ====== AutoSchemaKG 測試 ======
log "====== AutoSchemaKG 測試 ======"

if run_test "AutoSchemaKG" "--graph_rag_method autoschema --top_k 2"; then
    RESULTS+=("✅ AutoSchemaKG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ AutoSchemaKG")
    ((FAILURE_COUNT++))
fi

# ====== 模組化組合測試 ======
log "====== 模組化組合測試 ======"

if run_test "AutoSchemaKG + LightRAG" "--graph_preset autoschema_lightrag --top_k 2"; then
    RESULTS+=("✅ AutoSchemaKG + LightRAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ AutoSchemaKG + LightRAG")
    ((FAILURE_COUNT++))
fi

if run_test "LightRAG + CSR" "--graph_preset lightrag_csr --top_k 2"; then
    RESULTS+=("✅ LightRAG + CSR")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG + CSR")
    ((FAILURE_COUNT++))
fi

if run_test "DynamicSchema + CSR" "--graph_preset dynamic_csr --top_k 2"; then
    RESULTS+=("✅ DynamicSchema + CSR")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ DynamicSchema + CSR")
    ((FAILURE_COUNT++))
fi

if run_test "DynamicSchema + LightRAG" "--graph_preset dynamic_lightrag --top_k 2"; then
    RESULTS+=("✅ DynamicSchema + LightRAG")
    ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ DynamicSchema + LightRAG")
    ((FAILURE_COUNT++))
fi

# ====== LightRAG 檢索模式測試 (可選) ======
# 如果要測試所有檢索模式，取消以下註解

# log "====== LightRAG 檢索模式測試 ======"
# 
# for mode in local global mix naive bypass; do
#     if run_test "LightRAG ($mode mode)" "--graph_rag_method lightrag --lightrag_mode $mode --lightrag_schema_method lightrag_default"; then
#         RESULTS+=("✅ LightRAG ($mode mode)")
#         ((SUCCESS_COUNT++))
#     else
#         RESULTS+=("❌ LightRAG ($mode mode)")
#         ((FAILURE_COUNT++))
#     fi
# done

# ====== 總結 ======
echo
log "======================================================"
log "實驗總結"
log "======================================================"
log "結束時間: $(date)"
log ""

for result in "${RESULTS[@]}"; do
    log "$result"
done

log ""
log "成功: $SUCCESS_COUNT"
log "失敗: $FAILURE_COUNT"
log "總計: $((SUCCESS_COUNT + FAILURE_COUNT))"
log "======================================================"

echo
echo "完整日誌已儲存至: $LOG_FILE"
echo "結果檔案位於: $RESULTS_DIR"
echo

# 產生簡易 HTML 報告（可選）
HTML_REPORT="$LOG_DIR/experiment_${TIMESTAMP}_report.html"
cat > "$HTML_REPORT" << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAG 實驗報告 - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .success { color: green; }
        .failure { color: red; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>RAG 實驗報告</h1>
    <p>時間: $(date)</p>
    <p>資料類型: $DATA_TYPE</p>
    
    <h2>測試結果</h2>
    <table>
        <tr>
            <th>測試項目</th>
            <th>狀態</th>
        </tr>
EOF

for result in "${RESULTS[@]}"; do
    if [[ $result == ✅* ]]; then
        echo "        <tr><td>${result#✅ }</td><td class='success'>✅ 成功</td></tr>" >> "$HTML_REPORT"
    else
        echo "        <tr><td>${result#❌ }</td><td class='failure'>❌ 失敗</td></tr>" >> "$HTML_REPORT"
    fi
done

cat >> "$HTML_REPORT" << EOF
    </table>
    
    <h2>統計</h2>
    <p>成功: <span class="success">$SUCCESS_COUNT</span></p>
    <p>失敗: <span class="failure">$FAILURE_COUNT</span></p>
    <p>總計: $((SUCCESS_COUNT + FAILURE_COUNT))</p>
</body>
</html>
EOF

log "HTML 報告已產生: $HTML_REPORT"

exit 0
