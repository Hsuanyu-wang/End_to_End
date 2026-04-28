#########################################################################################################
# PropertyGraph methods
#########################################################################################################
# ext_list=("implicit" "schema" "simple" "dynamic")
# ret_list=("vector" "synonym" "text2cypher")
# modes=("cascade" "single") # "ensemble"
# DIerate_combinations() {
#     local items=("$@")
#     local brace_expr="{"
#     for item in "${items[@]}"; do
#         brace_expr+="$item,"
#     done
#     brace_expr="${brace_expr%,}}"
#     eval echo "$brace_expr" | tr ' ' '\n' | sed -e 's/,,/,/g' -e 's/^,//' -e 's/,$//' | grep -v '^$'
# }
# all_exts=$(DIerate_combinations "${ext_list[@]}")
# all_rets=$(DIerate_combinations "${ret_list[@]}")
# for ext in $all_exts; do
#     for ret in $all_rets; do
#         for mode in "${modes[@]}"; do
#             echo "----------------------------------------------------"
#             echo "Running: Ext=$ext, Ret=$ret, Mode=$mode"
#             echo "----------------------------------------------------"
            
#             python run_evaluation.py \
#                 --graph_type property_graph \
#                 --pg_extractors "$ext" \
#                 --pg_retrievers "$ret" \
#                 --pg_combination_mode "$mode"
#         done
#     done
# done

#########################################################################################################
# autoschema plugin
#########################################################################################################
# python run_evaluation.py --graph_type autoschema --data_type DI

#########################################################################################################
# dynamic builder plugin
#########################################################################################################
# python run_evaluation.py --graph_type dynamic --data_type DI

#########################################################################################################
# LightRAG methods
#########################################################################################################
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --graph_retrieval native --lightrag_mode hybrid --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --graph_retrieval native --lightrag_mode original --lightrag_native_mode hybrid --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --graph_retrieval native --lightrag_mode naive --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --graph_retrieval native --lightrag_mode original --lightrag_native_mode naive --ollama_url http://192.168.63.174:11434

#########################################################################################################
# simmerge plugin
#########################################################################################################
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.95 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.9 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.8 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_threshold_max 0.95 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_threshold_max 0.95 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434

#########################################################################################################
# schema plugin
#########################################################################################################
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method lightrag_default --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method adaptive_consolidation --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method iterative_evolution --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method llm_dynamic --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method llamaindex_dynamic --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method no_schema --ollama_url http://192.168.63.174:11434

#########################################################################################################
# temporal plugin
#########################################################################################################
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --plugin_temporal

#########################################################################################################
# Retrieval methods
#########################################################################################################
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval native

# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.9 --ppr_weight_mode semantic
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.8 --ppr_weight_mode semantic
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.85 --ppr_weight_mode degree
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.9 --ppr_weight_mode degree
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.8 --ppr_weight_mode degree
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.85 --ppr_weight_mode combined
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.9 --ppr_weight_mode combined
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval ppr --ppr_alpha 0.8 --ppr_weight_mode combined

# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval pcst --pcst_cost_mode inverse_log_weight
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval pcst --pcst_cost_mode uniform

# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval tog --tog_max_iterations 3 --tog_beam_width 5

# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval k_hop --graph_hop_k 2
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval anchor_hybrid_khop --graph_hop_k 2 --top_k 20
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval tog_refine --tog_max_iterations 3 --tog_beam_width 5

# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval pg_ensemble
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval pg_cascade
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval pg_single


#########################################################################################################
# UltraDomain methods
#########################################################################################################
# python run_evaluation.py --data_type LIGHTRAG_CS --graph_type lightrag --lightrag_mode hybrid
# python run_evaluation.py --data_type DI --graph_type lightrag --lightrag_mode hybrid --ollama_url http://192.168.63.174:11434
# python run_evaluation.py --data_type LIGHTRAG_LEGAL --graph_type lightrag --lightrag_mode hybrid
# python run_evaluation.py --data_type LIGHTRAG_AGRICULTURE --graph_type lightrag --lightrag_mode hybrid
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.95 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.95 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.9 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.9 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.9 --simmerge_threshold_max 0.95 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.9 --simmerge_threshold_max 0.95 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.8 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.8 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434

python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_threshold_max 0.95 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_threshold_max 0.95 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.80 --simmerge_threshold_max 0.95 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.80 --simmerge_threshold_max 0.95 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_threshold_max 0.90 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_threshold_max 0.90 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.80 --simmerge_threshold_max 0.90 --ollama_url http://192.168.63.174:11434
python run_evaluation.py --data_mode markdown --data_type DI --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.80 --simmerge_threshold_max 0.90 --simmerge_text_mode name_desc --ollama_url http://192.168.63.174:11434

# python run_evaluation.py --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method lightrag_default  --ollama_url http://192.168.63.174:11434
# python run_evaluation.py --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method no_schema  --ollama_url http://192.168.63.174:11434
# python run_evaluation.py --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method iterative_evolution  --ollama_url http://192.168.63.174:11434
# python run_evaluation.py --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method llm_dynamic  --ollama_url http://192.168.63.174:11434
# python run_evaluation.py --data_type DI --graph_type lightrag --lightrag_mode hybrid --schema_method llamaindex_dynamic  --ollama_url http://192.168.63.174:11434