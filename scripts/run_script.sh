### PropertyGraph methods ###
# ext_list=("implicit" "schema" "simple" "dynamic")
# ret_list=("vector" "synonym" "text2cypher")
# modes=("cascade" "single") # "ensemble"
# generate_combinations() {
#     local items=("$@")
#     local brace_expr="{"
#     for item in "${items[@]}"; do
#         brace_expr+="$item,"
#     done
#     brace_expr="${brace_expr%,}}"
#     eval echo "$brace_expr" | tr ' ' '\n' | sed -e 's/,,/,/g' -e 's/^,//' -e 's/,$//' | grep -v '^$'
# }
# all_exts=$(generate_combinations "${ext_list[@]}")
# all_rets=$(generate_combinations "${ret_list[@]}")
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

### autoschema plugin ###
# python run_evaluation.py --graph_type autoschema --data_type DI

### dynamic builder plugin ###
# python run_evaluation.py --graph_type dynamic --data_type DI

### LightRAG methods ###
python run_evaluation.py --graph_type lightrag --graph_retrieval native --lightrag_mode hybrid
python run_evaluation.py --graph_type lightrag --graph_retrieval native --lightrag_mode original --lightrag_native_mode hybrid
python run_evaluation.py --graph_type lightrag --graph_retrieval native --lightrag_mode naive
python run_evaluation.py --graph_type lightrag --graph_retrieval native --lightrag_mode original --lightrag_native_mode naive

### simmerge plugin ###
python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.90
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_text_mode name_desc
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --plugin_simmerge --simmerge_threshold 0.85 --simmerge_threshold_max 0.95 --simmerge_text_mode name_desc

### schema plugin ###
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --schema_method lightrag_default
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --schema_method iterative_evolution
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --schema_method llm_dynamic
# python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --schema_method llamaindex_dynamic
python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --schema_method no_schema

### temporal plugin ###
python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --plugin_temporal

### Retrieval methods ###
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

python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval one_hop
python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval pg_ensemble
python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval pg_cascade
python run_evaluation.py --graph_type lightrag --lightrag_mode hybrid --graph_retrieval pg_single
