# DATA_TYPES=("BIO" "RAGAS_CSR_FULL" "RAGAS_CSR")
# LIGHTRAG_MODES=("mix")
# THRESHOLDS=(0.95 0.9 0.85 0.8)
# # TEXT_MODES = ("name_desc", "name")

# for THRESHOLD in "${THRESHOLDS[@]}"; do
#     for DATA_TYPE in "${DATA_TYPES[@]}"; do
#         for LIGHTRAG_MODE in "${LIGHTRAG_MODES[@]}"; do
            # echo "=================================================="
            # echo "開始執行: data_type = $DATA_TYPE, lightrag_mode = $LIGHTRAG_MODE, simmerge_threshold = $THRESHOLD"
            # echo "=================================================="

            # python run_evaluation.py \
            #     --graph_type lightrag \
            #     --lightrag_mode "$LIGHTRAG_MODES" \
            #     --data_type "$DATA_TYPE" \
            #     --ollama_url http://192.168.63.174:11434
            #     --postfix "gemma4"

            # python run_evaluation.py \
            #     --graph_type lightrag \
            #     --lightrag_mode "$LIGHTRAG_MODES" \
            #     --data_type "$DATA_TYPE" \
            #     --plugin_simmerge \
            #     --simmerge_threshold "$THRESHOLD" \
            #     --simmerge_text_mode "$TEXT_MODES" \
            #     --simmerge_threshold_max 0.95 \
            #     --ollama_url http://192.168.63.174:11434

#             # python run_evaluation.py \
#             #     --graph_type lightrag \
#             #     --schema_method iterative_evolution \
#             #     --lightrag_mode "$LIGHTRAG_MODES" \
#             #     --evolution_ratio "$(python3 -c "print(50/100)")" \
#             #     --postfix "r50" \
#             #     --data_type "$DATA_TYPE" \
#             #     --ollama_url http://192.168.63.174:11434

#             echo "執行完成。"
#             echo ""
#         done
#     done
# done

# 跑 ITER
DATA_TYPES=("BIO") # "BIO" "RAGAS_CSR_FULL" "RAGAS_CSR"
LIGHTRAG_MODES=("mix")
THRESHOLD=0.9 # 0.95, 0.85, 0.8
RATIO=50
EVOLUTION_RATIO=0.5
TEXT_MODE=("name_desc") # , "name"
OLLAMA_URL="http://192.168.63.174:11434"

### 跑 MODE ###
# echo "=================================================="
# echo "開始執行: data_type = $DATA_TYPES, lightrag_mode = $LIGHTRAG_MODES, ratio = $RATIO"
# echo "=================================================="

# python run_evaluation.py \
#         --graph_type lightrag \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --data_type "$DATA_TYPES" \
#         --ollama_url http://192.168.63.174:11434

## 跑 ITERATIVE_EVOLUTION ###
# echo "=================================================="
# echo "開始執行: data_type = $DATA_TYPES, lightrag_mode = $LIGHTRAG_MODES, ratio = $RATIO, simmerge_threshold = $THRESHOLD"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method iterative_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --postfix "r$RATIO" \
#         --data_type "$DATA_TYPES" \
#         --ollama_url http://192.168.63.174:11434

### 跑 SIMMERGE ###
# echo "=================================================="
# echo "開始執行: data_type = $DATA_TYPES, lightrag_mode = $LIGHTRAG_MODES, ratio = $RATIO, simmerge_threshold = $THRESHOLD"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method iterative_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --postfix "r$RATIO" \
#         --data_type "$DATA_TYPES" \
#         --plugin_simmerge \
#         --simmerge_threshold "$THRESHOLD" \
#         --simmerge_text_mode "$TEXT_MODE" \
#         --ollama_url http://192.168.63.174:11434

# ## 跑 ITERATIVE_EVOLUTION + SIMMERGE ###
# echo "=================================================="
# echo "開始執行: data_type = $DATA_TYPES, lightrag_mode = $LIGHTRAG_MODES, ratio = $RATIO, simmerge_threshold = $THRESHOLD"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method iterative_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --postfix "r$RATIO" \
#         --data_type "$DATA_TYPES" \
#         --plugin_simmerge \
#         --simmerge_threshold "$THRESHOLD" \
#         --simmerge_text_mode "$TEXT_MODE" \
#         --ollama_url http://192.168.63.174:11434


# 跑 DI 真實資料

# DATA_TYPES=("DI") # "BIO" "RAGAS_CSR_FULL" "RAGAS_CSR"
# LIGHTRAG_MODES=("mix")
# THRESHOLDS=(0.9) # 0.95, 0.85, 0.8
# RATIOS=(50)
# TEXT_MODES=("name_desc") # , "name"

# ### 跑 MODE ###
# echo "=================================================="
# echo "開始執行: data_type = DI, lightrag_mode = $LIGHTRAG_MODES, ratio = $RATIOS"
# echo "=================================================="

# python run_evaluation.py \
#         --graph_type lightrag \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --data_type "$DATA_TYPES" \
#         --ollama_url http://192.168.63.174:11434

# ### 跑 ITERATIVE_EVOLUTION ###
# echo "=================================================="
# echo "開始執行: data_type = DI, lightrag_mode = $LIGHTRAG_MODES, ratio = $RATIOS, simmerge_threshold = $THRESHOLDS"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method iterative_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --postfix "r$RATIO" \
#         --data_type "$DATA_TYPES" \
#         --ollama_url http://192.168.63.174:11434

# ### 跑 SIMMERGE ###
# echo "=================================================="
# echo "開始執行: data_type = DI, lightrag_mode = $LIGHTRAG_MODES, ratio = $RATIOS, simmerge_threshold = $THRESHOLDS"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method iterative_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --postfix "r$RATIO" \
#         --data_type "$DATA_TYPES" \
#         --plugin_simmerge \
#         --simmerge_threshold "$THRESHOLD" \
#         --simmerge_text_mode "$TEXT_MODE" \
#         --ollama_url http://192.168.63.174:11434

# ## 跑 ITERATIVE_EVOLUTION + SIMMERGE ###
# echo "=================================================="
# echo "開始執行: data_type = DI, lightrag_mode = $LIGHTRAG_MODES, ratio = $RATIOS, simmerge_threshold = $THRESHOLDS"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method iterative_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --postfix "r$RATIO" \
#         --data_type "$DATA_TYPES" \
#         --plugin_simmerge \
#         --simmerge_threshold "$THRESHOLD" \
#         --simmerge_text_mode "$TEXT_MODE" \
#         --ollama_url http://192.168.63.174:11434

# ============================================================
# AAE (Anchored Additive Evolution) 實驗組
# 對照組：lightrag_default / iterative_evolution_r50（已在上方）
# ============================================================

# MIN_FREQUENCY=2   # 新 type 需在幾個 batch 被建議才納入 schema

# ## AAE f1（min_frequency=1，接近純 additive）###
# echo "=================================================="
# echo "[AAE] data_type=$DATA_TYPES, mode=$LIGHTRAG_MODES, ratio=$RATIO, f=1"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method anchored_additive_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --evolution_min_frequency 1 \
#         --data_type "$DATA_TYPES" \
#         --ollama_url http://192.168.63.174:11434

# ## AAE f2（標準設定）###
# echo "=================================================="
# echo "[AAE] data_type=$DATA_TYPES, mode=$LIGHTRAG_MODES, ratio=$RATIO, f=2"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method anchored_additive_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --evolution_min_frequency 2 \
#         --data_type "$DATA_TYPES" \
#         --ollama_url http://192.168.63.174:11434

# # ## AAE f2 + embedding 分群採樣 ###
# echo "=================================================="
# echo "[AAE+cluster] data_type=$DATA_TYPES, mode=$LIGHTRAG_MODES, ratio=$RATIO, f=2"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method anchored_additive_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --evolution_min_frequency 2 \
#         --evolution_cluster \
#         --data_type "$DATA_TYPES" \
#         --ollama_url http://192.168.63.174:11434

# ## AAE f2 + cluster + SER ###
# echo "=================================================="
# echo "[AAE+cluster+SER] data_type=$DATA_TYPES, mode=$LIGHTRAG_MODES, ratio=$RATIO, f=2"
# echo "=================================================="
# python run_evaluation.py \
#         --graph_type lightrag \
#         --schema_method anchored_additive_evolution \
#         --lightrag_mode "$LIGHTRAG_MODES" \
#         --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#         --evolution_min_frequency 2 \
#         --evolution_cluster \
#         --data_type "$DATA_TYPES" \
#         --plugin_simmerge \
#         --simmerge_threshold "$THRESHOLD" \
#         --simmerge_text_mode "$TEXT_MODES" \
#         --ollama_url http://192.168.63.174:11434


# ### + PETN + SER ###
# python run_evaluation.py \
#     --graph_type lightrag --lightrag_mode mix \
#     --data_type RAGAS_CSR --ollama_url http://192.168.63.174:11434 \
#     --plugin_petn \
#     --plugin_simmerge --simmerge_threshold 0.9 --simmerge_text_mode name_desc

# ### + EDC + SER ###
# python run_evaluation.py \
#     --graph_type lightrag --lightrag_mode mix \
#     --data_type RAGAS_CSR --ollama_url http://192.168.63.174:11434 \
#     --plugin_edc \
#     --plugin_simmerge --simmerge_threshold 0.9 --simmerge_text_mode name_desc

# ### + AAE + SER ###
# python run_evaluation.py \
#     --graph_type lightrag --lightrag_mode mix \
#     --data_type RAGAS_CSR --ollama_url http://192.168.63.174:11434 \
#     --schema_method anchored_additive_evolution \
#     --evolution_ratio 0.5 \
#     --evolution_cluster \
#     --plugin_simmerge --simmerge_threshold 0.9 --simmerge_text_mode name_desc

# ### + AAE + SER W/O BOOTSTRAP###
# python run_evaluation.py \
#     --graph_type lightrag --lightrag_mode mix \
#     --data_type RAGAS_CSR --ollama_url http://192.168.63.174:11434 \
#     --schema_method anchored_additive_evolution --evolution_cluster --no_evolution_bootstrap \
#     --evolution_ratio 0.5 \
#     --evolution_cluster \
#     --plugin_simmerge --simmerge_threshold 0.9 --simmerge_text_mode name_desc

# ### + AAE + SER W/O CONSOLIDATE ###
# python run_evaluation.py \
#     --graph_type lightrag --lightrag_mode mix \
#     --data_type RAGAS_CSR --ollama_url http://192.168.63.174:11434 \
#     --schema_method anchored_additive_evolution --evolution_cluster --no_evolution_consolidate \
#     --evolution_ratio 0.5 \
#     --evolution_cluster \
#     --plugin_simmerge --simmerge_threshold 0.9 --simmerge_text_mode name_desc

# ### + AAE + SER W/O ADAPTIVE FREQUENCY ###
# python run_evaluation.py \
#     --graph_type lightrag --lightrag_mode mix \
#     --data_type RAGAS_CSR --ollama_url http://192.168.63.174:11434 \
#     --schema_method anchored_additive_evolution --evolution_cluster --no_evolution_adaptive_freq --evolution_min_frequency 1 \
#     --evolution_ratio 0.5 \
#     --evolution_cluster \
#     --plugin_simmerge --simmerge_threshold 0.9 --simmerge_text_mode name_desc

##################################### 測 retrieval 方法 #####################################
# # 2) k_hop
# python run_evaluation.py \
#   --graph_type lightrag \
#   --graph_retrieval k_hop \
#   --graph_hop_k 2 \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --postfix "SER_r${RATIO}_khop_k2" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --ollama_url "$OLLAMA_URL"

# # 3) ppr
# python run_evaluation.py \
#   --graph_type lightrag \
#   --graph_retrieval ppr \
#   --ppr_alpha 0.85 \
#   --ppr_weight_mode semantic \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --postfix "SER_r${RATIO}_ppr_a085_semantic" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --ollama_url "$OLLAMA_URL" \
#   --eval_metrics_mode kfc_only
# # 4) pcst
# python run_evaluation.py \
#   --graph_type lightrag \
#   --graph_retrieval pcst \
#   --pcst_cost_mode inverse_weight \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --postfix "SER_r${RATIO}_pcst_invw" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --ollama_url "$OLLAMA_URL" \
#   --eval_metrics_mode kfc_only
# # 5) tog
# python run_evaluation.py \
#   --graph_type lightrag \
#   --graph_retrieval tog \
#   --tog_max_iterations 3 \
#   --tog_beam_width 5 \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --postfix "SER_r${RATIO}_tog_i3_b5" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --ollama_url "$OLLAMA_URL" \
#   --eval_metrics_mode kfc_only
# # 6) tog_refine
# python run_evaluation.py \
#   --graph_type lightrag \
#   --graph_retrieval tog_refine \
#   --tog_max_iterations 3 \
#   --tog_beam_width 5 \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --postfix "SER_r${RATIO}_togref_i3_b5" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --ollama_url "$OLLAMA_URL" \
#   --eval_metrics_mode kfc_only
# # 7) anchor_hybrid_khop
# python run_evaluation.py \
#   --graph_type lightrag \
#   --graph_retrieval anchor_hybrid_khop \
#   --graph_hop_k 2 \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --postfix "SER_r${RATIO}_anchorhybrid_k2" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --ollama_url "$OLLAMA_URL" \
#   --eval_metrics_mode kfc_only

  ##################################### 測數量影響 #####################################
RAGAS_CSR_QA_50_0_50="/home/End_to_End_RAG/Data/Ragas_GEN/CSR/Timeline_QA_20260418_090428_50_0_50.csv"
BIO_QA_50_0_50="/home/End_to_End_RAG/Data/Ragas_GEN/Bio/Timeline_QA_20260417_124713_50_0_50.csv"
RAGAS_CSR_FULL_QA_50_0_50="/home/End_to_End_RAG/Data/Ragas_GEN/CSR_FULL/Timeline_QA_20260418_063238_50_0_50.csv"

# STSM（Schema-Typed SimMerge）
# 流程：lightrag_default 建圖（完整 extraction）→ SimMerge → Schema-Typed Merge
# schema 只用於 merge guidance，不約束 extraction，避免 coverage 下降
# echo "=================================================="
# echo "開始執行: SER+STSM（SimMerge + Schema-Typed Merge）"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --schema_method lightrag_default \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --plugin_schema_typed_simmerge \
#   --schema_typed_merge_method anchored_additive_evolution \
#   --schema_typed_merge_sim_threshold 0.82 \
#   --schema_typed_merge_type_threshold 0.60 \
#   --evolution_ratio "$EVOLUTION_RATIO" \
#   --ollama_url http://192.168.63.174:11434

# # MODE naive
# echo "=================================================="
# echo "開始執行: NAIVE 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode naive \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$BIO_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # STSM NameDesc Embedding
# echo "=================================================="
# echo "開始執行: STSM NameDesc Embedding 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --qa_csv "$BIO_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only \
#   --plugin_schema_typed_simmerge \
#   --schema_typed_merge_method lightrag_default \
#   --schema_typed_merge_dry_run \
#   --schema_typed_merge_force_recopy \
#   --schema_typed_merge_entity_text_mode name \
#   --schema_typed_merge_type_text_mode name

#   # ITER + STSM NameDesc Embedding
# echo "=================================================="
# echo "開始執行: ITER + STSM NameDesc Embedding 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --schema_method iterative_evolution \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --qa_csv "$BIO_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only \
#   --plugin_schema_typed_simmerge \
#   --schema_typed_merge_method lightrag_default \
#   --schema_typed_merge_dry_run \
#   --schema_typed_merge_force_recopy \
#   --schema_typed_merge_entity_text_mode name \
#   --schema_typed_merge_type_text_mode name

# # MODE mix
# echo "=================================================="
# echo "開始執行: MIX 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$BIO_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # ITER+SER
# echo "=================================================="
# echo "開始執行: ITER+SER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --schema_method iterative_evolution \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --qa_csv "$BIO_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # ITER
# echo "=================================================="
# echo "開始執行: ITER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --schema_method iterative_evolution \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$BIO_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# ITER
echo "=================================================="
echo "開始執行: ITER 50_0_50"
echo "=================================================="
python run_evaluation.py \
  --graph_type lightrag \
  --schema_method iterative_evolution \
  --lightrag_mode "$LIGHTRAG_MODES" \
  --evolution_ratio "$(python3 -c "print(30/100)")" \
  --data_type "$DATA_TYPES" \
  --qa_csv "$BIO_QA_50_0_50" \
  --postfix "r30_50_0_50" \
  --ollama_url http://192.168.63.174:11434 \
  --eval_metrics_mode kfc_only

# # SER
# echo "=================================================="
# echo "開始執行: SER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --qa_csv "$BIO_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# DATA_TYPES=("RAGAS_CSR") # "BIO" "RAGAS_CSR_FULL" "RAGAS_CSR"

# # MODE
# echo "=================================================="
# echo "開始執行: NAIVE 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode naive \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$RAGAS_CSR_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# echo "=================================================="
# echo "開始執行: MIX 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$RAGAS_CSR_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # ITER+SER
# echo "=================================================="
# echo "開始執行: ITER+SER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --schema_method iterative_evolution \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --qa_csv "$RAGAS_CSR_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # ITER
# echo "=================================================="
# echo "開始執行: ITER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --schema_method iterative_evolution \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$RAGAS_CSR_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # SER
# echo "=================================================="
# echo "開始執行: SER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --qa_csv "$RAGAS_CSR_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # MODE
# echo "=================================================="
# echo "開始執行: NAIVE 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode naive \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$RAGAS_CSR_FULL_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# echo "=================================================="
# echo "開始執行: MIX 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$RAGAS_CSR_FULL_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

## ITER+SER
# echo "=================================================="
# echo "開始執行: ITER+SER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --schema_method iterative_evolution \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --qa_csv "$RAGAS_CSR_FULL_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # ITER
# echo "=================================================="
# echo "開始執行: ITER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --schema_method iterative_evolution \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --evolution_ratio "$(python3 -c "print($RATIO/100)")" \
#   --data_type "$DATA_TYPES" \
#   --qa_csv "$RAGAS_CSR_FULL_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only

# # SER
# echo "=================================================="
# echo "開始執行: SER 50_0_50"
# echo "=================================================="
# python run_evaluation.py \
#   --graph_type lightrag \
#   --lightrag_mode "$LIGHTRAG_MODES" \
#   --data_type "$DATA_TYPES" \
#   --plugin_simmerge \
#   --simmerge_threshold "$THRESHOLD" \
#   --simmerge_text_mode "$TEXT_MODE" \
#   --qa_csv "$RAGAS_CSR_FULL_QA_50_0_50" \
#   --postfix "50_0_50" \
#   --ollama_url http://192.168.63.174:11434 \
#   --eval_metrics_mode kfc_only