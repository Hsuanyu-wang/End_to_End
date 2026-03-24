#!/bin/bash

# 完整 RAG 實驗批次執行腳本（統一框架）
# 使用新的 --graph_type / --graph_retrieval 統一 CLI 參數

set -e

SCRIPT_DIR="/home/End_to_End_RAG"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/experiment_logs"
DATA_TYPE="DI"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/experiment_${TIMESTAMP}.log"

echo "======================================================"
echo "開始完整 RAG 實驗（統一框架）"
echo "======================================================"
echo "開始時間: $(date)"
echo "資料類型: $DATA_TYPE"
echo "日誌檔案: $LOG_FILE"
echo "======================================================"
echo

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_test() {
    local test_name=$1
    local args=$2

    log "開始測試: $test_name"
    log "參數: $args"

    local start_time=$(date +%s)

    if python run_evaluation.py $args --data_type "$DATA_TYPE" 2>&1 | tee -a "$LOG_FILE"; then
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

declare -a RESULTS
SUCCESS_COUNT=0
FAILURE_COUNT=0

# # ====== Vector RAG 測試 ======
# log "====== Vector RAG 測試 ======"

# if run_test "Vector Hybrid RAG" "--vector_method hybrid --top_k 2"; then
#     RESULTS+=("✅ Vector Hybrid RAG"); ((SUCCESS_COUNT++))
# else
#     RESULTS+=("❌ Vector Hybrid RAG"); ((FAILURE_COUNT++))
# fi

# if run_test "Vector Only RAG" "--vector_method vector --top_k 2"; then
#     RESULTS+=("✅ Vector Only RAG"); ((SUCCESS_COUNT++))
# else
#     RESULTS+=("❌ Vector Only RAG"); ((FAILURE_COUNT++))
# fi

# if run_test "BM25 RAG" "--vector_method bm25 --top_k 2"; then
#     RESULTS+=("✅ BM25 RAG"); ((SUCCESS_COUNT++))
# else
#     RESULTS+=("❌ BM25 RAG"); ((FAILURE_COUNT++))
# fi

# # ====== Advanced Vector RAG 測試 ======
# log "====== Advanced Vector RAG 測試 ======"

# if run_test "Parent-Child RAG" "--adv_vector_method parent_child --top_k 2"; then
#     RESULTS+=("✅ Parent-Child RAG"); ((SUCCESS_COUNT++))
# else
#     RESULTS+=("❌ Parent-Child RAG"); ((FAILURE_COUNT++))
# fi

# if run_test "Self-Query RAG" "--adv_vector_method self_query --top_k 2"; then
#     RESULTS+=("✅ Self-Query RAG"); ((SUCCESS_COUNT++))
# else
#     RESULTS+=("❌ Self-Query RAG"); ((FAILURE_COUNT++))
# fi

# ====== LightRAG Schema 方法測試 ======
log "====== LightRAG Schema 方法測試 ======"

if run_test "LightRAG (Default Schema)" "--graph_type lightrag --lightrag_mode hybrid --schema_method lightrag_default"; then
    RESULTS+=("✅ LightRAG (Default Schema)"); ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG (Default Schema)"); ((FAILURE_COUNT++))
fi

if run_test "LightRAG (Iterative Evolution)" "--graph_type lightrag --lightrag_mode hybrid --schema_method iterative_evolution"; then
    RESULTS+=("✅ LightRAG (Iterative Evolution)"); ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG (Iterative Evolution)"); ((FAILURE_COUNT++))
fi

if run_test "LightRAG (LLM Dynamic)" "--graph_type lightrag --lightrag_mode hybrid --schema_method llm_dynamic"; then
    RESULTS+=("✅ LightRAG (LLM Dynamic)"); ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG (LLM Dynamic)"); ((FAILURE_COUNT++))
fi

# ====== LightRAG Traversal Strategy 測試 ======
log "====== LightRAG Traversal Strategy 測試 ======"

if run_test "LightRAG + PPR" "--graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.85 --ppr_weight_mode semantic"; then
    RESULTS+=("✅ LightRAG + PPR"); ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG + PPR"); ((FAILURE_COUNT++))
fi

if run_test "LightRAG + PCST" "--graph_type lightrag --lightrag_mode hybrid --graph_retrieval pcst --pcst_cost_mode inverse_weight"; then
    RESULTS+=("✅ LightRAG + PCST"); ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG + PCST"); ((FAILURE_COUNT++))
fi

if run_test "LightRAG + ToG" "--graph_type lightrag --lightrag_mode hybrid --graph_retrieval tog --tog_max_iterations 3 --tog_beam_width 5"; then
    RESULTS+=("✅ LightRAG + ToG"); ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ LightRAG + ToG"); ((FAILURE_COUNT++))
fi

# # ====== PropertyGraph 測試 ======
# log "====== PropertyGraph 測試 ======"

# if run_test "PropertyGraph (Ensemble)" "--graph_type property_graph --pg_extractors implicit,schema,simple --pg_retrievers vector,synonym --pg_combination_mode ensemble"; then
#     RESULTS+=("✅ PropertyGraph (Ensemble)"); ((SUCCESS_COUNT++))
# else
#     RESULTS+=("❌ PropertyGraph (Ensemble)"); ((FAILURE_COUNT++))
# fi

# ====== AutoSchema 測試 ======
log "====== AutoSchema 測試 ======"

if run_test "AutoSchema + LightRAG" "--graph_type autoschema --top_k 2"; then
    RESULTS+=("✅ AutoSchema + LightRAG"); ((SUCCESS_COUNT++))
else
    RESULTS+=("❌ AutoSchema + LightRAG"); ((FAILURE_COUNT++))
fi

# ====== SimMerge Plugin 網格搜尋 ======
log "====== SimMerge Plugin 網格搜尋 ======"

THRESHOLDS=(0.95 0.9 0.85 0.8)
MAX_THRESHOLDS=(none 0.95 0.9)
TEXT_MODES=("name" "name_desc")

for TH in "${THRESHOLDS[@]}"; do
    for TH_MAX in "${MAX_THRESHOLDS[@]}"; do
        for TM in "${TEXT_MODES[@]}"; do
            if [[ "$TH_MAX" == "none" ]]; then
                test_name="SimMerge th=${TH} th_max=none tm=${TM}"
                test_args="--graph_type lightrag --lightrag_mode hybrid --schema_method lightrag_default --plugin_simmerge --simmerge_threshold ${TH} --simmerge_text_mode ${TM}"
                if run_test "$test_name" "$test_args"; then
                    RESULTS+=("✅ ${test_name}"); ((SUCCESS_COUNT++))
                else
                    RESULTS+=("❌ ${test_name}"); ((FAILURE_COUNT++))
                fi
            else
                # 上界不含：需 threshold_max > threshold
                if ! awk "BEGIN{exit !(${TH_MAX} > ${TH})}"; then
                    continue
                fi
                test_name="SimMerge th=${TH} th_max=${TH_MAX} tm=${TM}"
                test_args="--graph_type lightrag --lightrag_mode hybrid --schema_method lightrag_default --plugin_simmerge --simmerge_threshold ${TH} --simmerge_threshold_max ${TH_MAX} --simmerge_text_mode ${TM}"
                if run_test "$test_name" "$test_args"; then
                    RESULTS+=("✅ ${test_name}"); ((SUCCESS_COUNT++))
                else
                    RESULTS+=("❌ ${test_name}"); ((FAILURE_COUNT++))
                fi
            fi
        done
    done
done

# ====== LightRAG 檢索模式測試 (可選) ======
# 取消以下註解以測試所有模式
# log "====== LightRAG 檢索模式測試 ======"
# for mode in local global mix naive bypass; do
#     if run_test "LightRAG ($mode mode)" "--graph_type lightrag --lightrag_mode $mode --schema_method lightrag_default"; then
#         RESULTS+=("✅ LightRAG ($mode mode)"); ((SUCCESS_COUNT++))
#     else
#         RESULTS+=("❌ LightRAG ($mode mode)"); ((FAILURE_COUNT++))
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
        <tr><th>測試項目</th><th>狀態</th></tr>
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
