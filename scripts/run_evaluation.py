#!/usr/bin/env python3
"""
RAG 評估主執行腳本（統一框架）

此腳本提供完整的 RAG Pipeline 評估功能，支援：
- Vector RAG (hybrid, vector, bm25)
- Advanced Vector RAG (self_query, parent_child)
- Graph RAG：統一入口 --graph_type + --graph_retrieval
  - LightRAG (native / strategy-based: ppr, pcst, tog, k_hop, anchor_hybrid_khop)
  - PropertyGraph (pg_ensemble / pg_cascade / pg_single)
  - AutoSchema / Dynamic Builder
- Plugin: --plugin_simmerge, --plugin_temporal
- 多種評估指標（檢索、生成、LLM-as-Judge）
"""

import os
import re
import asyncio
import argparse
import warnings
import json
import nest_asyncio

import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import src.plugins.similar_entity_merge_plugin  # noqa: F401 — 註冊 similar_entity_merge

from src.config.settings import get_settings
from src.data.loaders import load_and_normalize_qa_by_type
from src.data.processors import data_processing
from src.rag.vector import get_vector_query_engine, get_self_query_engine, get_parent_child_query_engine
from src.rag.graph import (
    get_lightrag_engine,
    build_lightrag_index,
    TemporalLightRAGPackage,
)
from src.rag.schema import get_schema_by_method
from src.rag.wrappers import (
    VectorRAGWrapper,
    LightRAGWrapper,
    LightRAGWrapper_Original,
    TemporalLightRAGWrapper,
)
from src.rag.wrappers.lightrag_wrapper import LightRAGStrategyWrapper
from src.evaluation import run_evaluation
from src.evaluation.reporters import run_evaluation_with_token_budget

nest_asyncio.apply()


# ---------------------------------------------------------------------------
# CLI 參數解析
# ---------------------------------------------------------------------------

def _data_type_arg(value: str) -> str:
    """驗證資料類別字串；結果會寫入 results/{exp|test}/<此名稱>/。"""
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise argparse.ArgumentTypeError(
            "資料類別僅能使用英數與底線（例如 DI、GEN、GEN2）"
        )
    return value


def parse_arguments():
    """解析命令列參數（統一框架）"""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline 評估系統（統一框架）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Vector RAG ──
    g_vec = parser.add_argument_group("Vector RAG")
    g_vec.add_argument(
        "--vector_method", type=str, default="none",
        choices=["none", "hybrid", "vector", "bm25", "all"],
        help="Vector RAG 方法",
    )
    g_vec.add_argument(
        "--adv_vector_method", type=str, default="none",
        choices=["none", "parent_child", "self_query", "all"],
        help="進階 Vector RAG 方法",
    )

    # ── Graph RAG（統一入口）──
    g_graph = parser.add_argument_group("Graph RAG（統一入口）")
    g_graph.add_argument(
        "--graph_type", type=str, default="none",
        choices=["none", "lightrag", "property_graph", "autoschema", "dynamic"],
        help="Graph Builder 類型",
    )
    g_graph.add_argument(
        "--graph_retrieval", type=str, default="native",
        choices=["native", "k_hop", "ppr", "pcst", "tog", "tog_refine", "anchor_hybrid_khop",
                 "pg_ensemble", "pg_cascade", "pg_single"],
        help="Graph 檢索策略 (native=Builder 預設; k_hop/ppr/pcst/tog/tog_refine/anchor_hybrid_khop=traversal strategy; pg_*=PropertyGraph)",
    )

    # ── LightRAG 模式 ──
    g_lr = parser.add_argument_group("LightRAG 模式")
    g_lr.add_argument(
        "--lightrag_mode", type=str, default="hybrid",
        choices=["hybrid", "local", "global", "mix", "naive", "bypass", "original", "all"],
        help="LightRAG 檢索模式 (original=官方端到端; all=展開六種 context 模式)",
    )
    g_lr.add_argument(
        "--lightrag_native_mode", type=str, default="hybrid",
        choices=["local", "global", "hybrid", "mix", "naive", "bypass"],
        help="僅在 --lightrag_mode original 時生效：官方 aquery 使用的 mode",
    )

    # ── Schema / Extractor ──
    g_schema = parser.add_argument_group("Schema / Extractor")
    g_schema.add_argument(
        "--schema_method", type=str, default="lightrag_default",
        choices=["lightrag_default", "adaptive_consolidation", "iterative_evolution",
                 "anchored_additive_evolution", "llm_dynamic", "llamaindex_dynamic", "no_schema"],
        help="Schema 生成方法",
    )
    g_schema.add_argument(
        "--evolution_ratio", type=float, default=0.5,
        help="iterative_evolution / anchored_additive_evolution 使用語料的最大比例（0.0~1.0）",
    )
    g_schema.add_argument(
        "--evolution_min_frequency", type=int, default=2,
        help="[AAE] 新增 entity type 需在幾個批次中被 LLM 建議才納入最終 schema（預設 2）",
    )
    g_schema.add_argument(
        "--evolution_cluster", action="store_true",
        help="[AAE] 使用 embedding K-means 分群取代順序切片，確保批次多樣性",
    )
    g_schema.add_argument(
        "--evolution_bootstrap", action="store_true", default=True,
        help="[AAE] 使用 embedding cluster 取代表文件產生領域初始 schema（預設啟用）",
    )
    g_schema.add_argument(
        "--no_evolution_bootstrap", dest="evolution_bootstrap", action="store_false",
        help="[AAE] 停用 domain bootstrap，改用通用 _BASE_SCHEMA 作為初始錨點",
    )
    g_schema.add_argument(
        "--evolution_consolidate", action="store_true", default=True,
        help="[AAE] 迭代完成後執行 schema consolidation（清理值型態、統一英文）（預設啟用）",
    )
    g_schema.add_argument(
        "--no_evolution_consolidate", dest="evolution_consolidate", action="store_false",
        help="[AAE] 停用 schema consolidation",
    )
    g_schema.add_argument(
        "--evolution_adaptive_freq", action="store_true", default=True,
        help="[AAE] 依批次數動態調整 min_frequency（預設啟用，忽略 --evolution_min_frequency）",
    )
    g_schema.add_argument(
        "--no_evolution_adaptive_freq", dest="evolution_adaptive_freq", action="store_false",
        help="[AAE] 停用 adaptive min_frequency，改用 --evolution_min_frequency 固定值",
    )
    g_schema.add_argument(
        "--pg_extractors", type=str, default="implicit,schema,simple",
        help="PropertyGraph extractors (逗號分隔): implicit,schema,simple,dynamic",
    )
    g_schema.add_argument(
        "--pg_retrievers", type=str, default="vector,synonym",
        help="PropertyGraph retrievers (逗號分隔): vector,synonym,text2cypher",
    )
    g_schema.add_argument(
        "--pg_combination_mode", type=str, default="ensemble",
        choices=["ensemble", "cascade", "single"],
        help="PropertyGraph 多 retriever 組合模式",
    )

    # ── Plugins ──
    g_plug = parser.add_argument_group("Plugins")
    g_plug.add_argument(
        "--plugin_simmerge", action="store_true",
        help="啟用相似實體合併 (Similar Entity Merge)",
    )
    g_plug.add_argument("--simmerge_threshold", type=float, default=None, help="SimMerge 餘弦相似度下界")
    g_plug.add_argument("--simmerge_threshold_max", type=float, default=None, help="SimMerge 餘弦相似度上界（不含）")
    g_plug.add_argument(
        "--simmerge_text_mode", type=str, default="",
        choices=["", "name", "name_desc"],
        help="SimMerge embedding 用文本模式",
    )
    g_plug.add_argument("--simmerge_force_recopy", action="store_true", help="SimMerge 強制重新複製 storage")
    g_plug.add_argument("--simmerge_dry_run", action="store_true", help="SimMerge 僅產生 log 不實際合併")
    g_plug.add_argument("--plugin_temporal", action="store_true", help="啟用時序圖譜 (Temporal Graph)")
    g_plug.add_argument(
        "--plugin_petn", action="store_true",
        help="啟用 Post-hoc Entity Type Normalization（建圖後用 LLM 重新標注 entity_type）",
    )
    g_plug.add_argument(
        "--petn_schema_method", type=str, default="anchored_additive_evolution",
        choices=["lightrag_default", "anchored_additive_evolution", "iterative_evolution"],
        help="PETN 使用的 schema 來源方法（預設 anchored_additive_evolution）",
    )
    g_plug.add_argument("--petn_force_recopy", action="store_true", help="PETN 強制重新複製 storage")
    g_plug.add_argument(
        "--plugin_edc", action="store_true",
        help="啟用 Post-hoc Entity Description Consolidation（建圖後用 LLM 整合實體描述）",
    )
    g_plug.add_argument(
        "--edc_min_desc_len", type=int, default=200,
        help="EDC 只處理描述長度超過此值的 entity（預設 200）",
    )
    g_plug.add_argument("--edc_force_recopy", action="store_true", help="EDC 強制重新複製 storage")
    g_plug.add_argument(
        "--plugin_schema_typed_simmerge", action="store_true",
        help="啟用 Schema-Typed SimMerge（STSM）：以 schema evolution 的 entity types 作為 type-aware 合併錨點，"
             "串接於 SimMerge 之後，補捉同語義類別但名稱表達不同的 entity 重複",
    )
    g_plug.add_argument(
        "--schema_typed_merge_method", type=str, default="anchored_additive_evolution",
        choices=["lightrag_default", "anchored_additive_evolution", "iterative_evolution",
                 "llm_dynamic", "llamaindex_dynamic"],
        help="STSM 使用的 schema 方法（只用於 merge guidance，不影響 extraction；預設 anchored_additive_evolution）",
    )
    g_plug.add_argument(
        "--schema_typed_merge_sim_threshold", type=float, default=0.82,
        help="STSM 同 type 群內 entity 合併的餘弦相似度下界（預設 0.82；低於 SimMerge 因有 type 約束）",
    )
    g_plug.add_argument(
        "--schema_typed_merge_type_threshold", type=float, default=0.60,
        help="STSM entity 歸類到 schema type 的最低餘弦相似度（預設 0.60）",
    )
    g_plug.add_argument(
        "--schema_typed_merge_entity_text_mode", type=str, default="name",
        choices=["name", "name_desc"],
        help="STSM entity embedding 文本模式（name 或 name_desc；預設 name）",
    )
    g_plug.add_argument(
        "--schema_typed_merge_type_text_mode", type=str, default="name",
        choices=["name", "name_desc"],
        help="STSM schema type embedding 文本模式（name 或 name_desc；預設 name）",
    )
    g_plug.add_argument(
        "--schema_typed_merge_force_recopy", action="store_true",
        help="STSM 強制重新複製 storage（清除快取並重新執行 type-aware merge）",
    )
    g_plug.add_argument(
        "--schema_typed_merge_dry_run", action="store_true",
        help="STSM 僅產生 log，不實際執行合併（供調試用）",
    )

    # ── Traversal Strategy 參數 ──
    g_strat = parser.add_argument_group("Traversal Strategy 參數")
    g_strat.add_argument("--ppr_alpha", type=float, default=0.85, help="PPR damping factor")
    g_strat.add_argument(
        "--ppr_weight_mode", type=str, default="semantic",
        choices=["semantic", "degree", "combined"],
        help="PPR edge weight 計算方式",
    )
    g_strat.add_argument(
        "--pcst_cost_mode", type=str, default="inverse_weight",
        choices=["inverse_weight", "inverse_log_weight", "uniform"],
        help="PCST edge cost 計算方式",
    )
    g_strat.add_argument("--tog_max_iterations", type=int, default=3, help="ToG 最大迭代次數")
    g_strat.add_argument("--tog_beam_width", type=int, default=5, help="ToG beam search 寬度")
    # g_strat.add_argument("--tog_prune_top_n", type=int, default=3, help="ToG LLM 修剪 top-N 鄰居")
    g_strat.add_argument("--graph_hop_k", type=int, default=1, help="K-hop 策略 hop 深度")

    # ── 方法參數 ──
    g_method = parser.add_argument_group("方法參數")
    g_method.add_argument("--top_k", type=int, default=20, help="檢索數量")
    g_method.add_argument("--retrieval_max_tokens", type=int, default=0, help="檢索內容最大 token 數")
    g_method.add_argument(
        "--eval_metrics_mode",
        type=str,
        default="kfc_only",
        choices=["kfc_only", "full"],
        help="評估指標模式（kfc_only=僅計算 KFC；full=計算完整指標）",
    )

    # ── Token Budget ──
    g_tb = parser.add_argument_group("Token Budget")
    g_tb.add_argument("--enable_token_budget", action="store_true", help="啟用 token budget 控制")
    g_tb.add_argument("--token_budget_baseline", type=str, default="vector_hybrid", help="baseline 方法名稱")

    # ── 資料 / 模型 ──
    g_data = parser.add_argument_group("資料 / 模型")
    g_data.add_argument("--data_type", type=_data_type_arg, default="DI", help="資料類型 (DI, GEN, ...)")
    g_data.add_argument(
        "--data_mode", type=str, default="natural_text",
        choices=["natural_text", "markdown", "key_value_text", "unstructured_text"],
        help="資料格式",
    )
    g_data.add_argument("--model_type", type=str, default="small", choices=["small", "big"], help="模型大小")

    # ── Ollama / LLM override（覆蓋 config.yml 預設）──
    g_data.add_argument(
        "--ollama_url",
        type=str,
        default=None,
        help="覆蓋 config.yml 的 model/eval_model/builder_model.ollama_url（Base URL）:http://192.168.63.184:11434。",
    )
    g_data.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="覆蓋 config.yml 的 model.llm_model（主 LLM）。",
    )
    g_data.add_argument(
        "--eval_llm_model",
        type=str,
        default=None,
        help="覆蓋 config.yml 的 eval_model.llm_model（LLM-as-Judge）。",
    )
    g_data.add_argument(
        "--builder_llm_model",
        type=str,
        default=None,
        help="覆蓋 config.yml 的 builder_model.llm_model（圖譜/Schema 用）。",
    )

    # ── 測試 / 輸出 ──
    g_test = parser.add_argument_group("測試 / 輸出")
    g_test.add_argument("--qa_dataset_fast_test", action="store_true", help="快速測試模式（僅前 2 題）")
    g_test.add_argument("--vector_build_fast_test", action="store_true", help="Vector 索引快速建立")
    g_test.add_argument("--graph_build_fast_test", action="store_true", help="Graph 索引快速建立")
    g_test.add_argument("--qa_csv", type=str, default="", help="覆蓋 data_type 對應的 QA 檔案路徑（CSV/JSONL）")
    g_test.add_argument("--postfix", type=str, default="", help="結果資料夾名稱後綴")
    g_test.add_argument("--sup", type=str, default="", help="快取方法標識")

    # ── Deprecated 參數（向後相容）──
    g_dep = parser.add_argument_group("Deprecated（向後相容，請改用新參數）")
    g_dep.add_argument("--graph_rag_method", type=str, default="none", help="[DEPRECATED] 改用 --graph_type")
    g_dep.add_argument("--unified_graph_type", type=str, default="none", help="[DEPRECATED] 改用 --graph_type")
    g_dep.add_argument("--graph_preset", type=str, default="", help="[DEPRECATED] 改用 --graph_type + --graph_retrieval")
    g_dep.add_argument("--graph_builder", type=str, default="", help="[DEPRECATED] 改用 --graph_type")
    g_dep.add_argument("--graph_retriever", type=str, default="", help="[DEPRECATED] 改用 --graph_retrieval")
    g_dep.add_argument("--graph_traversal_strategy", type=str, default="default", help="[DEPRECATED] 改用 --graph_retrieval")
    g_dep.add_argument("--lightrag_plugins", type=str, default="", help="[DEPRECATED] 改用 --plugin_simmerge")
    g_dep.add_argument("--lightrag_sim_merge_threshold", type=float, default=None, help="[DEPRECATED] 改用 --simmerge_threshold")
    g_dep.add_argument("--lightrag_sim_merge_threshold_max", type=float, default=None, help="[DEPRECATED] 改用 --simmerge_threshold_max")
    g_dep.add_argument("--lightrag_sim_merge_text_mode", type=str, default="", help="[DEPRECATED] 改用 --simmerge_text_mode")
    g_dep.add_argument("--lightrag_sim_merge_force_recopy", action="store_true", help="[DEPRECATED] 改用 --simmerge_force_recopy")
    g_dep.add_argument("--lightrag_sim_merge_dry_run", action="store_true", help="[DEPRECATED] 改用 --simmerge_dry_run")
    g_dep.add_argument("--lightrag_schema_method", type=str, default="", help="[DEPRECATED] 改用 --schema_method")
    g_dep.add_argument("--lightrag_temporal_graph", action="store_true", help="[DEPRECATED] 改用 --plugin_temporal")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Deprecated 參數映射
# ---------------------------------------------------------------------------

_PRESET_MAP = {
    "autoschema_lightrag": ("autoschema", "native"),
    "lightrag_csr":        ("lightrag", "native"),
    "dynamic_csr":         ("dynamic", "native"),
    "dynamic_lightrag":    ("dynamic", "native"),
}

_DEPRECATED_GRAPH_RAG_MAP = {
    "lightrag":       "lightrag",
    "propertyindex":  "property_graph",
    "autoschema":     "autoschema",
    "dynamic_schema": "dynamic",
}


def _resolve_deprecated_args(args):
    """將舊 CLI 參數映射到新參數，並印出遷移提示。"""
    warned = False

    def _warn(old, new):
        nonlocal warned
        warnings.warn(f"{old} 已棄用，請改用 {new}", DeprecationWarning, stacklevel=3)
        print(f"⚠️  {old} 已棄用，請改用 {new}")
        warned = True

    # --graph_rag_method → --graph_type
    if args.graph_rag_method not in ("none", "") and args.graph_type == "none":
        mapped = _DEPRECATED_GRAPH_RAG_MAP.get(args.graph_rag_method)
        if mapped:
            args.graph_type = mapped
            _warn(f"--graph_rag_method {args.graph_rag_method}", f"--graph_type {mapped}")
        elif args.graph_rag_method == "all":
            args.graph_type = "lightrag"
            _warn("--graph_rag_method all", "--graph_type lightrag")
        else:
            print(f"⚠️  --graph_rag_method {args.graph_rag_method} 已棄用且無對應新參數，略過")

    # --unified_graph_type → --graph_type
    if args.unified_graph_type not in ("none", "") and args.graph_type == "none":
        args.graph_type = args.unified_graph_type
        _warn(f"--unified_graph_type {args.unified_graph_type}", f"--graph_type {args.graph_type}")

    # --graph_preset → --graph_type + --graph_retrieval
    if args.graph_preset and args.graph_type == "none":
        gt, gr = _PRESET_MAP.get(args.graph_preset, (None, None))
        if gt:
            args.graph_type = gt
            args.graph_retrieval = gr
            _warn(f"--graph_preset {args.graph_preset}", f"--graph_type {gt} --graph_retrieval {gr}")
        else:
            print(f"⚠️  --graph_preset {args.graph_preset} 無對應新參數，略過")

    # --graph_builder → --graph_type
    if args.graph_builder and args.graph_type == "none":
        args.graph_type = args.graph_builder
        _warn(f"--graph_builder {args.graph_builder}", f"--graph_type {args.graph_type}")

    # --graph_traversal_strategy → --graph_retrieval
    if args.graph_traversal_strategy != "default" and args.graph_retrieval == "native":
        args.graph_retrieval = args.graph_traversal_strategy
        _warn(f"--graph_traversal_strategy {args.graph_traversal_strategy}",
              f"--graph_retrieval {args.graph_retrieval}")

    # --lightrag_plugins similar_entity_merge → --plugin_simmerge
    if args.lightrag_plugins and not args.plugin_simmerge:
        plugins = [p.strip() for p in args.lightrag_plugins.split(",") if p.strip()]
        if "similar_entity_merge" in plugins:
            args.plugin_simmerge = True
            _warn("--lightrag_plugins similar_entity_merge", "--plugin_simmerge")

    # --lightrag_sim_merge_* → --simmerge_*
    if args.lightrag_sim_merge_threshold is not None and args.simmerge_threshold is None:
        args.simmerge_threshold = args.lightrag_sim_merge_threshold
    if args.lightrag_sim_merge_threshold_max is not None and args.simmerge_threshold_max is None:
        args.simmerge_threshold_max = args.lightrag_sim_merge_threshold_max
    if args.lightrag_sim_merge_text_mode and not args.simmerge_text_mode:
        args.simmerge_text_mode = args.lightrag_sim_merge_text_mode
    if args.lightrag_sim_merge_force_recopy:
        args.simmerge_force_recopy = True
    if args.lightrag_sim_merge_dry_run:
        args.simmerge_dry_run = True

    # --lightrag_schema_method → --schema_method
    if args.lightrag_schema_method and args.lightrag_schema_method != args.schema_method:
        if args.schema_method == "lightrag_default" and args.lightrag_schema_method != "lightrag_default":
            args.schema_method = args.lightrag_schema_method
            _warn(f"--lightrag_schema_method {args.lightrag_schema_method}",
                  f"--schema_method {args.schema_method}")

    # --lightrag_temporal_graph → --plugin_temporal
    if args.lightrag_temporal_graph and not args.plugin_temporal:
        args.plugin_temporal = True
        _warn("--lightrag_temporal_graph", "--plugin_temporal")

    # PropertyGraph 預設 graph_retrieval 映射
    if args.graph_type == "property_graph" and args.graph_retrieval == "native":
        args.graph_retrieval = f"pg_{args.pg_combination_mode}"

    return warned


# ---------------------------------------------------------------------------
# Vector Pipeline Setup（不變）
# ---------------------------------------------------------------------------

def setup_vector_pipelines(args, Settings, pipelines_to_test):
    """設置 Vector RAG Pipelines"""
    if args.vector_method == "none":
        print("Skip Vector RAG")
        return

    method_mapping = {
        "hybrid": "Vector_hybrid_RAG",
        "vector": "Vector_vector_RAG",
        "bm25": "Vector_bm25_RAG",
    }

    methods_to_test = ["hybrid", "vector", "bm25"] if args.vector_method == "all" else [args.vector_method]

    for method in methods_to_test:
        engine = get_vector_query_engine(
            Settings,
            vector_method=method,
            top_k=args.top_k,
            data_mode=args.data_mode,
            data_type=args.data_type,
            fast_build=args.vector_build_fast_test,
            retrieval_max_tokens=args.retrieval_max_tokens,
        )
        if engine is not None:
            wrapper = VectorRAGWrapper(name=method_mapping[method], query_engine=engine)
            wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
            pipelines_to_test.append(wrapper)


def setup_advanced_vector_pipelines(args, Settings, pipelines_to_test):
    """設置進階 Vector RAG Pipelines"""
    if args.adv_vector_method == "none":
        print("Skip Advanced Vector RAG")
        return

    if args.adv_vector_method in ["self_query", "all"]:
        engine = get_self_query_engine(
            Settings,
            data_mode=args.data_mode,
            data_type=args.data_type,
            top_k=args.top_k,
            fast_build=args.vector_build_fast_test,
            retrieval_max_tokens=args.retrieval_max_tokens,
        )
        wrapper = VectorRAGWrapper(name="Self_Query_RAG", query_engine=engine)
        wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
        pipelines_to_test.append(wrapper)

    if args.adv_vector_method in ["parent_child", "all"]:
        engine = get_parent_child_query_engine(
            Settings,
            data_mode=args.data_mode,
            data_type=args.data_type,
            top_k=args.top_k,
            fast_build=args.vector_build_fast_test,
            retrieval_max_tokens=args.retrieval_max_tokens,
        )
        wrapper = VectorRAGWrapper(name="Parent_Child_RAG", query_engine=engine)
        wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
        pipelines_to_test.append(wrapper)


# ---------------------------------------------------------------------------
# Graph Pipeline 工具函數
# ---------------------------------------------------------------------------

def parse_extractor_config(extractors_str: str) -> dict:
    """解析 PropertyGraph extractor 配置字串"""
    config = {}
    extractors = [e.strip() for e in extractors_str.split(",") if e.strip()]
    for extractor in extractors:
        if extractor == "implicit":
            config["implicit"] = {"enabled": True}
        elif extractor == "schema":
            config["schema"] = {
                "enabled": True,
                "entities": ["Record", "Engineer", "Customer", "System", "Issue"],
                "relations": ["HANDLED", "BELONGS_TO", "AFFECTS", "RESOLVES"],
                "strict": False,
            }
        elif extractor == "simple":
            config["simple"] = {"enabled": True, "max_paths_per_chunk": 10, "num_workers": 4}
        elif extractor == "dynamic":
            config["dynamic"] = {"enabled": True, "max_triplets_per_chunk": 20, "num_workers": 4}
    return config


def parse_retriever_config(retrievers_str: str) -> dict:
    """解析 PropertyGraph retriever 配置字串"""
    config = {}
    retrievers = [r.strip() for r in retrievers_str.split(",") if r.strip()]
    for retriever in retrievers:
        if retriever == "vector":
            config["vector"] = {"enabled": True, "similarity_top_k": 5}
        elif retriever == "synonym":
            config["synonym"] = {"enabled": True, "include_text": True}
        elif retriever == "text2cypher":
            config["text2cypher"] = {"enabled": True, "cypher_schema": None}
    return config


def _run_graph_quality_check(graph_result: dict, source_format: str) -> dict:
    """build() 完成後計算圖譜品質指標"""
    try:
        from src.evaluation.metrics.graph_quality import GraphQualityMetrics
        metrics = GraphQualityMetrics.compute_from_build(graph_result, source_format)
        if metrics.get("error"):
            print(f"⚠️  圖譜品質評估失敗: {metrics['error']}")
            return {}
        print(f"📊 圖譜品質: nodes={metrics['node_count']}, edges={metrics['edge_count']}, "
              f"density={metrics['density']}, components={metrics['num_connected_components']}, "
              f"orphans={metrics['orphan_node_count']}")
        storage_path = graph_result.get("storage_path", "")
        if storage_path:
            report_path = os.path.join(storage_path, "graph_quality_report.json")
            GraphQualityMetrics.save_report(metrics, report_path)
        return metrics
    except Exception as e:
        print(f"⚠️  圖譜品質評估異常: {e}")
        return {}


def _build_strategy_kwargs(args, Settings=None) -> dict:
    """從 CLI args 建立 traversal strategy 的初始化參數。"""
    strategy = args.graph_retrieval
    if strategy == "ppr":
        return {"alpha": args.ppr_alpha, "weight_mode": args.ppr_weight_mode}
    if strategy == "pcst":
        return {"cost_mode": args.pcst_cost_mode}
    if strategy in ("tog", "tog_refine"):
        return {"max_iterations": args.tog_max_iterations, "beam_width": args.tog_beam_width}
    if strategy == "k_hop":
        return {"hop_k": args.graph_hop_k}
    if strategy == "anchor_hybrid_khop":
        return {
            "hop_k": args.graph_hop_k,
            "embed_model": getattr(Settings, "embed_model", None) if Settings is not None else None,
        }
    return {}


# ---------------------------------------------------------------------------
# 統一 Graph Pipeline Setup（單一入口）
# ---------------------------------------------------------------------------

def setup_graph_pipeline(args, Settings, pipelines_to_test):
    """
    統一 Graph Pipeline 設置入口。

    根據 --graph_type 分派至對應的 Builder，
    根據 --graph_retrieval 分派至對應的 Retriever / Strategy。
    """
    if args.graph_type == "none":
        return

    print(f"\n{'='*60}")
    print(f"  Graph Pipeline: type={args.graph_type}, retrieval={args.graph_retrieval}")
    print(f"{'='*60}")

    if args.graph_type == "lightrag":
        _setup_lightrag(args, Settings, pipelines_to_test)
    elif args.graph_type == "property_graph":
        _setup_property_graph(args, Settings, pipelines_to_test)
    elif args.graph_type in ("autoschema", "dynamic"):
        _setup_builder(args, Settings, pipelines_to_test)
    else:
        print(f"⚠️  不支援的 graph_type: {args.graph_type}")


# ---------------------------------------------------------------------------
# LightRAG Pipeline
# ---------------------------------------------------------------------------

def _create_strategy_wrappers(args, Settings, lightrag_instance, schema_info, graph_quality, pipelines_to_test):
    """建立 LightRAGGraphRetriever + LightRAGStrategyWrapper 並加入 pipelines。"""
    from src.graph_retriever.lightrag_graph_retriever import LightRAGGraphRetriever

    strategy = args.graph_retrieval
    strategy_kwargs = _build_strategy_kwargs(args, Settings)
    if strategy in ("tog", "tog_refine"):
        # ToG 需要 llm 才能啟用 LLM-guided scoring / sufficiency 判斷
        strategy_kwargs.setdefault("llm", getattr(Settings, "llm", None))

    modes = (
        ["local", "global", "mix", "bypass", "hybrid", "naive"]
        if args.lightrag_mode == "all"
        else [args.lightrag_mode]
    )

    for mode in modes:
        retriever = LightRAGGraphRetriever(
            rag_instance=lightrag_instance,
            strategy=strategy,
            mode=mode,
            **strategy_kwargs,
        )
        retriever.initialize()

        wrapper = LightRAGStrategyWrapper(
            name=f"LightRAG_{strategy}_{mode}",
            rag_instance=lightrag_instance,
            retriever=retriever,
            mode=mode,
            schema_info=schema_info,
        )
        wrapper._graph_quality = graph_quality
        wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
        pipelines_to_test.append(wrapper)
        print(f"  Strategy wrapper: {wrapper.name}")


def _setup_lightrag(args, Settings, pipelines_to_test):
    """設置 LightRAG Pipeline（Build + Plugin + Retrieve）"""
    from src.storage import get_storage_path

    # --- 1. 解析 sup + schema ---
    # 將影響 graph 結果的所有參數編入 tag，讓不同設定各自有獨立 graph storage
    _schema_tag = args.schema_method
    if args.schema_method == "iterative_evolution":
        _ratio_pct = int(round(args.evolution_ratio * 100))
        _schema_tag = f"iterative_evolution_r{_ratio_pct}"
    elif args.schema_method == "anchored_additive_evolution":
        _ratio_pct = int(round(args.evolution_ratio * 100))
        _min_freq = getattr(args, "evolution_min_frequency", 2)
        _use_clus = getattr(args, "evolution_cluster", False)
        _use_boot = getattr(args, "evolution_bootstrap", True)
        _use_cons = getattr(args, "evolution_consolidate", True)
        _use_afq = getattr(args, "evolution_adaptive_freq", True)
        _schema_tag = f"aae_r{_ratio_pct}_f{_min_freq}"
        if _use_clus:
            _schema_tag += "_clus"
        if _use_boot:
            _schema_tag += "_boot"
        if _use_cons:
            _schema_tag += "_cons"
        if _use_afq:
            _schema_tag += "_afq"
    full_sup = args.sup + f"_{_schema_tag}" if args.sup else _schema_tag

    storage_mode = "" if args.lightrag_mode == "all" else args.lightrag_mode
    if storage_mode in ("none", "original"):
        storage_mode = ""

    fast_test_fb = args.graph_build_fast_test or args.qa_dataset_fast_test

    storage_path = get_storage_path(
        storage_type="lightrag",
        data_type=args.data_type,
        method=full_sup,
        mode=storage_mode,
        fast_test=fast_test_fb,
        custom_tag="",
    )

    lc = Settings.lightrag_config

    # --- 2. 建立 / 載入索引 ---
    schema_info = None
    is_no_schema = args.schema_method == "no_schema"
    if is_no_schema:
        schema_info = {
            "method": "no_schema",
            "entities": [],
            "relations": [],
            "validation_schema": {},
        }

    if not os.path.exists(storage_path) or not os.listdir(storage_path):
        print(f"📂 建立 LightRAG 索引: {storage_path}")

        if not is_no_schema:
            schema_info = get_schema_by_method(
                method=args.schema_method,
                text_corpus=data_processing(mode=args.data_mode, data_type=args.data_type),
                settings=Settings,
                data_type=args.data_type,
                data_mode=args.data_mode,
                return_full_schema=True,
                evolution_ratio=args.evolution_ratio,
                evolution_min_frequency=getattr(args, "evolution_min_frequency", 2),
                evolution_use_cluster=getattr(args, "evolution_cluster", False),
                evolution_use_bootstrap=getattr(args, "evolution_bootstrap", True),
                evolution_use_consolidate=getattr(args, "evolution_consolidate", True),
                evolution_adaptive_min_frequency=getattr(args, "evolution_adaptive_freq", True),
            )
            print(f"🌟 LightRAG 將使用以下實體類別建圖: {schema_info['entities']}")

            from llama_index.core import Settings as LlamaSettings
            LlamaSettings.lightrag_entity_types = schema_info["entities"]
            Settings.lightrag_config.entity_types = schema_info["entities"]
        else:
            print("🌟 使用 no_schema：不提供初始 schema，且不進行迭代演化")

        build_lightrag_index(
            Settings,
            mode=args.data_mode,
            data_type=args.data_type,
            sup=full_sup,
            fast_build=fast_test_fb,
            lightrag_mode=storage_mode,
            custom_tag="",
        )
    else:
        print(f"✅ LightRAG 索引已存在: {storage_path}")
        if not is_no_schema:
            schema_info = get_schema_by_method(
                method=args.schema_method,
                text_corpus=data_processing(mode=args.data_mode, data_type=args.data_type),
                settings=Settings,
                data_type=args.data_type,
                data_mode=args.data_mode,
                return_full_schema=True,
                evolution_ratio=args.evolution_ratio,
                evolution_min_frequency=getattr(args, "evolution_min_frequency", 2),
                evolution_use_cluster=getattr(args, "evolution_cluster", False),
                evolution_use_bootstrap=getattr(args, "evolution_bootstrap", True),
                evolution_use_consolidate=getattr(args, "evolution_consolidate", True),
                evolution_adaptive_min_frequency=getattr(args, "evolution_adaptive_freq", True),
            )
            print(f"🌟 使用現有 Schema: {schema_info['entities']}")
            from llama_index.core import Settings as LlamaSettings
            LlamaSettings.lightrag_entity_types = schema_info["entities"]
            Settings.lightrag_config.entity_types = schema_info["entities"]
        else:
            print("🌟 使用 no_schema：不載入 schema（沿用既有索引）")

    # --- 3. Plugins（PETN → EDC → SER，可任意組合串接）---
    # current_custom_tag 追蹤目前最終的 plugin storage 位置
    current_custom_tag = ""

    # Plugin: PETN（Post-hoc Entity Type Normalization）
    if getattr(args, "plugin_petn", False):
        from src.rag.plugins.lightrag_entity_type_normalize import (
            build_petn_custom_tag,
            ensure_petn_lightrag_index,
        )
        from src.data.processors import data_processing as _dp

        _petn_schema_method = getattr(args, "petn_schema_method", "anchored_additive_evolution")
        _petn_schema_info = get_schema_by_method(
            method=_petn_schema_method,
            text_corpus=_dp(mode=args.data_mode, data_type=args.data_type),
            settings=Settings,
            data_type=args.data_type,
            data_mode=args.data_mode,
            return_full_schema=True,
            evolution_ratio=getattr(args, "evolution_ratio", 0.5),
            evolution_min_frequency=getattr(args, "evolution_min_frequency", 2),
            evolution_use_cluster=getattr(args, "evolution_cluster", False),
            evolution_use_bootstrap=getattr(args, "evolution_bootstrap", True),
            evolution_use_consolidate=getattr(args, "evolution_consolidate", True),
            evolution_adaptive_min_frequency=getattr(args, "evolution_adaptive_freq", True),
        )
        _petn_entities = _petn_schema_info.get("entities", [])
        print(f"🏷️ [PETN] 使用 schema（{_petn_schema_method}）: {_petn_entities}")

        ensure_petn_lightrag_index(
            Settings,
            data_type=args.data_type,
            method=full_sup,
            mode=storage_mode,
            fast_test=fast_test_fb,
            schema_entities=_petn_entities,
            llm=Settings.builder_llm,
            schema_method=_petn_schema_method,
            source_custom_tag=current_custom_tag,
            force_recopy=getattr(args, "petn_force_recopy", False),
        )
        _petn_tag = build_petn_custom_tag()
        current_custom_tag = f"{current_custom_tag}_{_petn_tag}" if current_custom_tag else _petn_tag

    # Plugin: EDC（Post-hoc Entity Description Consolidation）
    if getattr(args, "plugin_edc", False):
        from src.rag.plugins.lightrag_entity_desc_consolidate import (
            build_edc_custom_tag,
            ensure_edc_lightrag_index,
        )

        ensure_edc_lightrag_index(
            Settings,
            data_type=args.data_type,
            method=full_sup,
            mode=storage_mode,
            fast_test=fast_test_fb,
            llm=Settings.builder_llm,
            source_custom_tag=current_custom_tag,
            min_desc_len=getattr(args, "edc_min_desc_len", 200),
            force_recopy=getattr(args, "edc_force_recopy", False),
        )
        _edc_tag = build_edc_custom_tag()
        current_custom_tag = f"{current_custom_tag}_{_edc_tag}" if current_custom_tag else _edc_tag

    # Plugin: SER（Similar Entity Merge）
    if args.plugin_simmerge:
        from src.rag.plugins.lightrag_similar_entity_merge import (
            build_simmerge_custom_tag,
            ensure_similar_merged_lightrag_index,
        )
        from src.rag.plugins.lightrag_simmerge_chained import (
            ensure_simmerge_from_source_lightrag_index,
        )

        th = args.simmerge_threshold if args.simmerge_threshold is not None else lc.similar_entity_merge_threshold
        tm = args.simmerge_text_mode if args.simmerge_text_mode else lc.similar_entity_merge_text_mode
        force = args.simmerge_force_recopy or lc.similar_entity_merge_force_recopy
        dry = args.simmerge_dry_run or lc.similar_entity_merge_dry_run
        th_max = args.simmerge_threshold_max if args.simmerge_threshold_max is not None else lc.similar_entity_merge_threshold_max

        if th_max is not None and th_max <= th:
            raise ValueError("similar_entity_merge：threshold_max 必須大於 threshold（上界不含、下界含）")

        if current_custom_tag:
            # 有前置 plugin（PETN 或 EDC），使用串接版 SER
            ensure_simmerge_from_source_lightrag_index(
                Settings,
                data_type=args.data_type,
                method=full_sup,
                mode=storage_mode,
                fast_test=fast_test_fb,
                threshold=th,
                threshold_max=th_max,
                text_mode=tm,
                force_recopy=force,
                dry_run=dry,
                source_custom_tag=current_custom_tag,
            )
        else:
            # 純 SER，沿用原始函式（不修改原邏輯）
            ensure_similar_merged_lightrag_index(
                Settings,
                data_type=args.data_type,
                method=full_sup,
                mode=storage_mode,
                fast_test=fast_test_fb,
                threshold=th,
                threshold_max=th_max,
                text_mode=tm,
                force_recopy=force,
                dry_run=dry,
            )

        _sim_tag = build_simmerge_custom_tag(th, tm, th_max)
        current_custom_tag = f"{current_custom_tag}_{_sim_tag}" if current_custom_tag else _sim_tag

    # Plugin: STSM（Schema-Typed SimMerge）
    if getattr(args, "plugin_schema_typed_simmerge", False):
        from src.rag.plugins.lightrag_schema_typed_merge import (
            build_schema_typed_merge_tag,
            ensure_schema_typed_merge_lightrag_index,
        )
        from src.data.processors import data_processing as _dp_stsm

        _stm_method = getattr(args, "schema_typed_merge_method", "anchored_additive_evolution")
        _stm_sim_th = getattr(args, "schema_typed_merge_sim_threshold", 0.82)
        _stm_type_th = getattr(args, "schema_typed_merge_type_threshold", 0.60)
        _stm_entity_tm = getattr(args, "schema_typed_merge_entity_text_mode", "name")
        _stm_type_tm = getattr(args, "schema_typed_merge_type_text_mode", "name")
        _stm_force = getattr(args, "schema_typed_merge_force_recopy", False)
        _stm_dry = getattr(args, "schema_typed_merge_dry_run", False)

        # 取得 schema types（只用於 merge guidance，不影響 extraction）
        _stm_schema_info = get_schema_by_method(
            method=_stm_method,
            text_corpus=_dp_stsm(mode=args.data_mode, data_type=args.data_type),
            settings=Settings,
            data_type=args.data_type,
            data_mode=args.data_mode,
            return_full_schema=True,
            evolution_ratio=args.evolution_ratio,
            evolution_min_frequency=getattr(args, "evolution_min_frequency", 2),
            evolution_use_cluster=getattr(args, "evolution_cluster", False),
            evolution_use_bootstrap=getattr(args, "evolution_bootstrap", True),
            evolution_use_consolidate=getattr(args, "evolution_consolidate", True),
            evolution_adaptive_min_frequency=getattr(args, "evolution_adaptive_freq", True),
        )
        _stm_types = (
            _stm_schema_info.get("entities", [])
            if isinstance(_stm_schema_info, dict)
            else list(_stm_schema_info)
        )
        print(f"🔷 [STSM] schema types ({len(_stm_types)}): {_stm_types}")

        ensure_schema_typed_merge_lightrag_index(
            Settings,
            data_type=args.data_type,
            method=full_sup,
            mode=storage_mode,
            fast_test=fast_test_fb,
            schema_types=_stm_types,
            schema_method_name=_stm_method,
            sim_threshold=_stm_sim_th,
            type_threshold=_stm_type_th,
            entity_text_mode=_stm_entity_tm,
            type_text_mode=_stm_type_tm,
            source_custom_tag=current_custom_tag,
            force_recopy=_stm_force,
            dry_run=_stm_dry,
        )

        _stsm_tag = build_schema_typed_merge_tag(
            _stm_method,
            _stm_sim_th,
            _stm_type_th,
            _stm_entity_tm,
            _stm_type_tm,
        )
        current_custom_tag = (
            f"{current_custom_tag}_{_stsm_tag}" if current_custom_tag else _stsm_tag
        )

    # --- 4. 取得 LightRAG 實例 ---
    lightrag_instance = get_lightrag_engine(
        Settings,
        data_type=args.data_type,
        sup=full_sup,
        mode=storage_mode,
        fast_test=fast_test_fb,
        custom_tag=current_custom_tag,
    )

    # --- 5. 圖譜品質 ---
    _lightrag_gq = {}
    _actual_path = (
        current_custom_tag
        and get_storage_path(
            storage_type="lightrag",
            data_type=args.data_type,
            method=full_sup,
            mode=storage_mode,
            fast_test=fast_test_fb,
            custom_tag=current_custom_tag,
        )
        or storage_path
    )
    _graphml = os.path.join(_actual_path, "graph_chunk_entity_relation.graphml")
    if os.path.exists(_graphml):
        try:
            from src.evaluation.metrics.graph_quality import GraphQualityMetrics
            _lightrag_gq = GraphQualityMetrics.compute_from_graphml(_graphml)
            print(
                f"📊 LightRAG 圖譜品質: nodes={_lightrag_gq['node_count']}, "
                f"edges={_lightrag_gq['edge_count']}, density={_lightrag_gq['density']}"
            )
        except Exception as e:
            print(f"⚠️  LightRAG 圖譜品質評估失敗: {e}")

    # --- 6. 建立 Wrappers ---
    if args.lightrag_mode != "none":
        if args.graph_retrieval not in ("native", "default"):
            _create_strategy_wrappers(
                args, Settings, lightrag_instance, schema_info, _lightrag_gq, pipelines_to_test
            )
        elif args.lightrag_mode == "all":
            for mode in ["local", "global", "mix", "bypass", "hybrid", "naive"]:
                wrapper = LightRAGWrapper(
                    name=f"LightRAG_{mode.capitalize()}",
                    rag_instance=lightrag_instance,
                    mode=mode,
                    schema_info=schema_info,
                )
                wrapper._graph_quality = _lightrag_gq
                wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
                pipelines_to_test.append(wrapper)
        elif args.lightrag_mode == "original":
            wrapper = LightRAGWrapper_Original(
                name=f"LightRAG_original_{args.lightrag_native_mode}",
                rag_instance=lightrag_instance,
                mode=args.lightrag_native_mode,
                schema_info=schema_info,
            )
            wrapper._graph_quality = _lightrag_gq
            wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
            pipelines_to_test.append(wrapper)
        else:
            wrapper = LightRAGWrapper(
                name=f"LightRAG_{args.lightrag_mode.capitalize()}",
                rag_instance=lightrag_instance,
                mode=args.lightrag_mode,
                schema_info=schema_info,
            )
            wrapper._graph_quality = _lightrag_gq
            wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
            pipelines_to_test.append(wrapper)

    # --- 7. Plugin: Temporal ---
    if args.plugin_temporal:
        _temporal_lr_mode = (
            args.lightrag_native_mode
            if args.lightrag_mode == "original"
            else (
                args.lightrag_mode
                if args.lightrag_mode not in ("none", "all", "original")
                else "hybrid"
            )
        )
        temporal_rag = TemporalLightRAGPackage(
            working_dir=os.path.join(
                Settings.lightrag_config.storage_path_DIR,
                args.data_type + "_temporal",
            )
        )
        wrapper = TemporalLightRAGWrapper(
            name="Temporal_LightRAG",
            rag_instance=temporal_rag,
            mode=_temporal_lr_mode,
            schema_info=schema_info,
        )
        pipelines_to_test.append(wrapper)


# ---------------------------------------------------------------------------
# PropertyGraph Pipeline
# ---------------------------------------------------------------------------

def _setup_property_graph(args, Settings, pipelines_to_test):
    """設置 PropertyGraph Pipeline（UnifiedGraphBuilder + UnifiedGraphRetriever）"""
    from src.graph_builder.unified import UnifiedGraphBuilder
    from src.graph_retriever.unified import UnifiedGraphRetriever
    from src.rag.wrappers import ModularGraphWrapper
    from src.graph_retriever.base_retriever import GraphData

    documents = data_processing(mode=args.data_mode, data_type=args.data_type)
    if args.graph_build_fast_test:
        documents = documents[:2]

    extractor_config = parse_extractor_config(args.pg_extractors)
    retriever_config = parse_retriever_config(args.pg_retrievers)

    combination_mode = args.graph_retrieval.replace("pg_", "") if args.graph_retrieval.startswith("pg_") else args.pg_combination_mode

    print(f"📦 PropertyGraph Extractors: {list(extractor_config.keys())}")
    print(f"🔍 PropertyGraph Retrievers: {list(retriever_config.keys())}")
    print(f"🔗 組合模式: {combination_mode}")

    try:
        builder = UnifiedGraphBuilder(
            settings=Settings,
            builder_type="property_graph",
            builder_config={"extractors": extractor_config},
        )

        graph_result = builder.build(documents)
        gq_metrics = _run_graph_quality_check(graph_result, "property_graph")

        retriever = UnifiedGraphRetriever(
            graph_source=graph_result,
            settings=Settings,
            retriever_type="property_graph",
            retriever_config=retriever_config,
            combination_mode=combination_mode,
        )

        wrapper_name = f"PG[{args.pg_extractors}]+[{args.pg_retrievers}]"
        wrapper = ModularGraphWrapper(
            name=wrapper_name,
            builder=builder,
            retriever=retriever,
            documents=None,
            enable_format_conversion=True,
        )
        wrapper._graph_quality = gq_metrics

        pg_metadata = graph_result.get("metadata", {})
        pg_metadata["graph_index"] = graph_result.get("graph_index")
        wrapper.graph_data = GraphData(
            nodes=graph_result.get("nodes", []),
            edges=graph_result.get("edges", []),
            metadata=pg_metadata,
            schema_info=graph_result.get("schema_info", {}),
            storage_path=graph_result.get("storage_path"),
            graph_format=graph_result.get("graph_format", "property_graph"),
        )

        wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
        pipelines_to_test.append(wrapper)
        print(f"✅ PropertyGraph Pipeline 設置完成: {wrapper_name}")

    except Exception as e:
        print(f"❌ PropertyGraph Pipeline 設置失敗: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# AutoSchema / Dynamic Builder Pipeline
# ---------------------------------------------------------------------------

def _setup_builder(args, Settings, pipelines_to_test):
    """設置 AutoSchema 或 Dynamic Builder Pipeline"""
    from src.rag.pipeline_factory import PipelineFactory

    documents = data_processing(mode=args.data_mode, data_type=args.data_type)
    if args.graph_build_fast_test:
        documents = documents[:2]

    builder_config = {
        "data_type": args.data_type,
        "data_mode": args.data_mode,
        "fast_test": args.graph_build_fast_test,
        "schema_method": args.schema_method,
        "sup": args.sup,
    }

    retriever_name = "lightrag"
    retriever_config = {
        "data_type": args.data_type,
        "fast_test": args.graph_build_fast_test,
        "mode": args.lightrag_mode if args.lightrag_mode not in ("none", "all") else "hybrid",
    }

    print(f"🔧 設置 {args.graph_type} Builder Pipeline...")

    try:
        pipeline = PipelineFactory.create_pipeline(
            builder_name=args.graph_type,
            retriever_name=retriever_name,
            settings=Settings,
            documents=documents,
            builder_config=builder_config,
            retriever_config=retriever_config,
            top_k=args.top_k,
            model_type=args.model_type,
        )

        if hasattr(pipeline, "set_retrieval_max_tokens"):
            pipeline.set_retrieval_max_tokens(args.retrieval_max_tokens)
        pipelines_to_test.append(pipeline)
        print(f"✅ {args.graph_type} Pipeline 設置完成: {pipeline.name}")

    except Exception as e:
        print(f"⚠️  {args.graph_type} Pipeline 設置失敗: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# build_postfix
# ---------------------------------------------------------------------------

def build_postfix(args):
    """建立結果資料夾名稱後綴"""
    parts = []

    def _sanitize(s: str) -> str:
        # 讓模型/URL 可安全用於資料夾名稱；同時避免過長的字元集造成問題
        s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s)
        return s.strip("-")

    if args.data_type:
        parts.append(args.data_type)

    # Vector method
    if args.vector_method != "none":
        parts.append(args.vector_method)
    if args.adv_vector_method != "none":
        parts.append(args.adv_vector_method)

    # Graph type
    if args.graph_type != "none":
        parts.append(args.graph_type)
        if args.graph_type == "property_graph":
            parts.append(f"ext_{args.pg_extractors.replace(',', '-')}")
            parts.append(f"ret_{args.pg_retrievers.replace(',', '-')}")
            parts.append(args.pg_combination_mode)

    # LightRAG mode
    if args.lightrag_mode not in ("none", "all"):
        parts.append(args.lightrag_mode)

    # Schema method
    if args.graph_type in ("lightrag",) and args.schema_method:
        parts.append(args.schema_method)

    # Graph retrieval strategy
    if args.graph_retrieval not in ("native", "default"):
        parts.append(args.graph_retrieval)
        if args.graph_retrieval == "ppr":
            parts.append(f"a{args.ppr_alpha}_{args.ppr_weight_mode}")
        elif args.graph_retrieval == "pcst":
            parts.append(args.pcst_cost_mode)
        elif args.graph_retrieval in ("tog", "tog_refine"):
            parts.append(f"i{args.tog_max_iterations}_b{args.tog_beam_width}")
        elif args.graph_retrieval in ("k_hop", "anchor_hybrid_khop"):
            parts.append(f"k{args.graph_hop_k}")

    # SimMerge tag
    if getattr(args, "simmerge_result_tag", None):
        parts.append(args.simmerge_result_tag)

    if args.postfix:
        parts.append(args.postfix)
    if args.sup:
        parts.append(args.sup)

    # CLI 覆蓋摘要（避免覆蓋到同名結果資料夾）
    # if getattr(args, "ollama_url", None):
    #     parts.append(f"ollama_{_sanitize(args.ollama_url)}")
    if getattr(args, "llm_model", None):
        parts.append(f"llm_{_sanitize(args.llm_model)}")
    if getattr(args, "eval_llm_model", None):
        parts.append(f"eval_llm_{_sanitize(args.eval_llm_model)}")
    if getattr(args, "builder_llm_model", None):
        parts.append(f"builder_llm_{_sanitize(args.builder_llm_model)}")

    if args.qa_dataset_fast_test:
        parts.append("fast_test")

    return "_".join(parts) if parts else ""


def _build_run_level_metadata(args, postfix: str) -> dict:
    """建立 run 層級 metadata（完整 CLI 與選用策略摘要）。"""
    cli_args = vars(args).copy()
    return {
        "postfix": postfix,
        "eval_metrics_mode": args.eval_metrics_mode,
        "full_cli_args_json": json.dumps(cli_args, ensure_ascii=False, sort_keys=True),
        "selected_strategies_json": json.dumps(
            {
                "vector_method": args.vector_method,
                "adv_vector_method": args.adv_vector_method,
                "graph_type": args.graph_type,
                "graph_retrieval": args.graph_retrieval,
                "lightrag_mode": args.lightrag_mode,
                "lightrag_native_mode": args.lightrag_native_mode,
                "schema_method": args.schema_method,
                "eval_metrics_mode": args.eval_metrics_mode,
                "pg_extractors": args.pg_extractors,
                "pg_retrievers": args.pg_retrievers,
                "pg_combination_mode": args.pg_combination_mode,
                "plugin_simmerge": args.plugin_simmerge,
                "plugin_temporal": args.plugin_temporal,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _infer_method_family(pipeline_name: str) -> str:
    """依 pipeline 名稱推斷方法家族，便於 summary 區隔。"""
    name = pipeline_name.lower()
    if name.startswith("vector_"):
        return "vector_rag"
    if "self_query" in name:
        return "advanced_vector_rag"
    if "parent_child" in name:
        return "advanced_vector_rag"
    if name.startswith("lightrag_") or "temporal_lightrag" in name:
        return "lightrag"
    if name.startswith("pg[") or "propertygraph" in name:
        return "property_graph"
    if "autoschema" in name:
        return "autoschema"
    if "dynamic" in name:
        return "dynamic"
    return "unknown"


def _build_method_signature(args, pipeline_name: str) -> str:
    """建立可讀且可重現的方法簽名字串。"""
    core = [
        f"pipeline={pipeline_name}",
        f"graph={args.graph_type}:{args.graph_retrieval}",
        f"vector={args.vector_method}",
        f"adv_vector={args.adv_vector_method}",
        f"lr_mode={args.lightrag_mode}",
        f"lr_native={args.lightrag_native_mode}",
        f"schema={args.schema_method}",
        f"pg_ext={args.pg_extractors}",
        f"pg_ret={args.pg_retrievers}",
        f"pg_mode={args.pg_combination_mode}",
        f"top_k={args.top_k}",
        f"retrieval_max_tokens={args.retrieval_max_tokens}",
        f"simmerge={int(args.plugin_simmerge)}",
        f"temporal={int(args.plugin_temporal)}",
    ]
    return "|".join(core)


def _build_pipeline_metadata_map(args, pipelines_to_test) -> dict:
    """建立各 pipeline 的 metadata（用於 global_summary 區隔）。"""
    metadata_map = {}
    for pipeline in pipelines_to_test:
        method_family = _infer_method_family(pipeline.name)
        metadata_map[pipeline.name] = {
            "method_family": method_family,
            "method_signature": _build_method_signature(args, pipeline.name),
        }
    return metadata_map


def _sync_ragas_eval_env(args, settings_obj) -> None:
    """
    將本次 CLI 的模型端點同步到 RAGAS 使用的 EVAL_* 環境變數。

    目標是讓主流程與 RAGAS 指標使用同一組後端，避免 endpoint 漂移。
    """
    cli_host = (args.ollama_url or "").strip()
    cli_eval_model = (args.eval_llm_model or "").strip()
    cli_embed_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip()

    if not (cli_host or cli_eval_model):
        return

    # 優先使用 CLI 覆蓋，否則回落到已初始化的 Settings 物件內容。
    eval_model = cli_eval_model or getattr(settings_obj.eval_llm, "model", "") or ""
    embed_model = (
        cli_embed_model
        or getattr(settings_obj.embed_model, "model_name", "")
        or getattr(settings_obj.embed_model, "model", "")
        or ""
    )

    if cli_host:
        os.environ["EVAL_LLM_BINDING_HOST"] = cli_host
        os.environ["EVAL_EMBEDDING_BINDING_HOST"] = cli_host
    if eval_model:
        os.environ["EVAL_LLM_MODEL"] = str(eval_model).strip()
    if embed_model:
        os.environ["EVAL_EMBEDDING_MODEL"] = str(embed_model).strip()


# ---------------------------------------------------------------------------
# Help / Main
# ---------------------------------------------------------------------------

def print_no_pipeline_help(args):
    """未選取任何 pipeline 時的啟用方式提示"""
    print("   可啟用方式擇一或併用：")
    print("   • Vector：--vector_method hybrid|vector|bm25|all")
    print("   • 進階 Vector：--adv_vector_method self_query|parent_child|all")
    print("   • LightRAG：--graph_type lightrag [--graph_retrieval native|k_hop|ppr|pcst|tog|tog_refine|anchor_hybrid_khop]")
    print("   • PropertyGraph：--graph_type property_graph [--pg_extractors ...] [--pg_retrievers ...]")
    print("   • AutoSchema：--graph_type autoschema")
    print("   • Dynamic：--graph_type dynamic")
    print("   • Plugin：--plugin_simmerge / --plugin_temporal")


def main():
    """主執行函數"""
    args = parse_arguments()

    # 舊參數向新參數映射
    _resolve_deprecated_args(args)

    Settings = get_settings(
        model_type=args.model_type,
        ollama_url=args.ollama_url,
        llm_model=args.llm_model,
        eval_llm_model=args.eval_llm_model,
        builder_llm_model=args.builder_llm_model,
    )
    _sync_ragas_eval_env(args, Settings)

    # Plugin result tag（用於結果目錄 postfix，依 PETN → EDC → SER 串接順序組合）
    args.simmerge_result_tag = None
    _plugin_tag_parts = []

    if getattr(args, "plugin_petn", False):
        _plugin_tag_parts.append("petn")

    if getattr(args, "plugin_edc", False):
        _plugin_tag_parts.append("edc")

    if args.plugin_simmerge:
        from src.rag.plugins.lightrag_similar_entity_merge import build_simmerge_custom_tag

        _lc = Settings.lightrag_config
        _th = args.simmerge_threshold if args.simmerge_threshold is not None else _lc.similar_entity_merge_threshold
        _tm = args.simmerge_text_mode if args.simmerge_text_mode else _lc.similar_entity_merge_text_mode
        _th_max = args.simmerge_threshold_max if args.simmerge_threshold_max is not None else _lc.similar_entity_merge_threshold_max
        if _th_max is not None and _th_max <= _th:
            raise ValueError("similar_entity_merge：threshold_max 必須大於 threshold（上界不含、下界含）")
        _plugin_tag_parts.append(build_simmerge_custom_tag(_th, _tm, _th_max))

    if getattr(args, "plugin_schema_typed_simmerge", False):
        from src.rag.plugins.lightrag_schema_typed_merge import build_schema_typed_merge_tag as _build_stsm_tag

        _stm_method = getattr(args, "schema_typed_merge_method", "anchored_additive_evolution")
        _stm_sim_th = getattr(args, "schema_typed_merge_sim_threshold", 0.82)
        _stm_type_th = getattr(args, "schema_typed_merge_type_threshold", 0.60)
        _stm_entity_tm = getattr(args, "schema_typed_merge_entity_text_mode", "name")
        _stm_type_tm = getattr(args, "schema_typed_merge_type_text_mode", "name")
        _plugin_tag_parts.append(
            _build_stsm_tag(
                _stm_method,
                _stm_sim_th,
                _stm_type_th,
                _stm_entity_tm,
                _stm_type_tm,
            )
        )

    if _plugin_tag_parts:
        args.simmerge_result_tag = "_".join(_plugin_tag_parts)

    # 快速測試模式連動
    if args.qa_dataset_fast_test:
        args.vector_build_fast_test = True
        args.graph_build_fast_test = True
        print("⚡ 已啟用 --qa_dataset_fast_test，自動連動開啟 vector_build_fast_test 與 graph_build_fast_test！")

    # 建立 Pipeline 列表
    pipelines_to_test = []
    setup_vector_pipelines(args, Settings, pipelines_to_test)
    setup_advanced_vector_pipelines(args, Settings, pipelines_to_test)
    setup_graph_pipeline(args, Settings, pipelines_to_test)

    # 統一套用 retrieval token 上限
    for pipeline in pipelines_to_test:
        if hasattr(pipeline, "set_retrieval_max_tokens"):
            pipeline.set_retrieval_max_tokens(args.retrieval_max_tokens)

    if not pipelines_to_test:
        print("⚠️ 未選擇任何 RAG pipeline，請檢查啟動參數。")
        print_no_pipeline_help(args)
        return

    # 載入 QA 資料集
    try:
        datasets = load_and_normalize_qa_by_type(args.data_type, args.qa_csv)
    except Exception as e:
        print(
            f"⚠️ 無法載入 --data_type={args.data_type!r} 的 QA 資料：{e}\n"
            "請檢查 config.yml 的 data.datasets 映射與 qa_loader 設定。"
        )
        return

    if args.qa_dataset_fast_test:
        print("⚡ 啟用快速測試模式，僅抽取前 2 題進行評估...")
        datasets = datasets[:2]
    else:
        print("啟用完整測試模式，進行完整 QA 評估...")

    postfix = build_postfix(args)
    run_metadata = _build_run_level_metadata(args, postfix)
    print(f"🧪 評估指標模式: {args.eval_metrics_mode}")
    pipeline_metadata_map = _build_pipeline_metadata_map(args, pipelines_to_test)
    result_bucket = "test" if args.qa_dataset_fast_test else "exp"
    results_root_dir = os.path.join(_PROJECT_ROOT, "results", result_bucket, args.data_type)
    print(f"📂 結果分流目錄: {results_root_dir}")

    if args.enable_token_budget:
        print("\n🎯 啟用Token Budget控制模式")
        asyncio.run(
            run_evaluation_with_token_budget(
                datasets,
                pipelines_to_test,
                postfix=postfix,
                baseline_method=args.token_budget_baseline,
                results_root_dir=results_root_dir,
                is_fast_test=args.qa_dataset_fast_test,
                run_metadata=run_metadata,
                pipeline_metadata_map=pipeline_metadata_map,
            )
        )
    else:
        asyncio.run(
            run_evaluation(
                datasets,
                pipelines_to_test,
                postfix=postfix,
                results_root_dir=results_root_dir,
                is_fast_test=args.qa_dataset_fast_test,
                run_metadata=run_metadata,
                pipeline_metadata_map=pipeline_metadata_map,
            )
        )


if __name__ == "__main__":
    main()
