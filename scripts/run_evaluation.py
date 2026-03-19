#!/usr/bin/env python3
"""
RAG 評估主執行腳本

此腳本提供完整的 RAG Pipeline 評估功能，支援：
- Vector RAG (hybrid, vector, bm25)
- Advanced Vector RAG (self_query, parent_child)
- Graph RAG (property_graph, dynamic_schema, LightRAG)
- 多種評估指標（檢索、生成、LLM-as-Judge）
"""

import os
import asyncio
import argparse
import nest_asyncio

# 新的 import 路徑
import sys
sys.path.insert(0, '/home/End_to_End_RAG')

from src.config.settings import get_settings
from src.data.loaders import load_and_normalize_qa_CSR_DI, load_and_normalize_qa_CSR_full
from src.data.processors import data_processing
from src.rag.vector import get_vector_query_engine, get_self_query_engine, get_parent_child_query_engine
from src.rag.graph import (
    get_graph_query_engine,
    get_dynamic_schema_graph_query_engine,
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
from src.evaluation import run_evaluation
from src.evaluation.reporters import run_evaluation_with_token_budget

nest_asyncio.apply()

# 註解：generation_max_tokens 功能已隱藏，改用 retrieval_max_tokens
# def apply_generation_cap_to_settings(Settings, max_tokens: int):
#     """嘗試將生成 token 上限套用到全域 Settings.llm。"""
#     if not max_tokens or max_tokens <= 0:
#         return
#     llm = getattr(Settings, "llm", None)
#     if llm is None:
#         return
#     try:
#         if hasattr(llm, "max_tokens"):
#             llm.max_tokens = max_tokens
#         if hasattr(llm, "num_predict"):
#             llm.num_predict = max_tokens
#         if hasattr(llm, "additional_kwargs") and isinstance(llm.additional_kwargs, dict):
#             llm.additional_kwargs["max_tokens"] = max_tokens
#             llm.additional_kwargs["num_predict"] = max_tokens
#         print(f"🎯 已套用全域 generation_max_tokens={max_tokens}")
#     except Exception as e:
#         print(f"⚠️ 套用 generation token 上限失敗，改用各 wrapper fallback。錯誤: {e}")


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="RAG Pipeline 評估系統")
    
    # RAG 方法選擇
    parser.add_argument(
        "--vector_method",
        type=str,
        default="none",
        choices=["none", "hybrid", "vector", "bm25", "all"],
        help="Vector RAG 方法"
    )
    parser.add_argument(
        "--adv_vector_method",
        type=str,
        default="none",
        choices=["none", "parent_child", "self_query", "all"],
        help="進階 Vector RAG 方法"
    )
    parser.add_argument(
        "--graph_rag_method",
        type=str,
        default="none",
        choices=["none", "propertyindex", "lightrag", "dynamic_schema", 
                 "autoschema", "graphiti", "neo4j", "cq_driven", "all"],
        help="Graph RAG 方法"
    )
    parser.add_argument(
        "--lightrag_mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "local", "global", "hybrid", "mix", "naive", "bypass", "original", "all"],
        help="""LightRAG 檢索模式:
        - local: 關注特定實體與細節
        - global: 關注整體趨勢與總結
        - hybrid: 混合 local 與 global
        - mix: 知識圖譜 + 向量檢索
        - naive: 僅向量檢索
        - bypass: 直接查詢 LLM
        - original: 使用 LightRAG 原生模式
        """
    )
    parser.add_argument(
        "--lightrag_plugins",
        type=str,
        default="",
        help="LightRAG 插件列表，用逗號分隔 (例如: autoschema,dynamic_path)"
    )
    
    # 方法參數
    parser.add_argument("--top_k", type=int, default=20, help="檢索數量")
    parser.add_argument("--retrieval_max_tokens", type=int, default=16384, help="檢索內容最大 token 數（限制傳給 LLM 的 context 長度）")
    # parser.add_argument("--generation_max_tokens", type=int, default=512, help="生成回答最大 token 數（已隱藏）")
    
    # Token Budget相關
    parser.add_argument(
        "--enable_token_budget",
        action="store_true",
        help="啟用token budget控制（動態調整LightRAG參數以匹配Vector RAG的token使用量）"
    )
    parser.add_argument(
        "--token_budget_baseline",
        type=str,
        default="vector_hybrid",
        help="用作baseline的方法名稱（預設為vector_hybrid）"
    )
    
    # Schema 相關
    parser.add_argument(
        "--lightrag_schema_method",
        type=str,
        default="lightrag_default",
        choices=["lightrag_default", "iterative_evolution", "llm_dynamic", "llamaindex_dynamic"],
        help="LightRAG Schema 生成方法"
    )
    parser.add_argument(
        "--lightrag_temporal_graph",
        action="store_true",
        help="啟用時序 LightRAG"
    )
    
    # 模組化 Pipeline 參數(新增)
    parser.add_argument(
        "--graph_builder",
        type=str,
        default="",
        choices=["", "autoschema", "lightrag", "property", "dynamic"],
        help="Graph Builder 選擇(用於模組化組合)"
    )
    parser.add_argument(
        "--graph_retriever",
        type=str,
        default="",
        choices=["", "lightrag", "csr", "neo4j"],
        help="Graph Retriever 選擇(用於模組化組合)"
    )
    parser.add_argument(
        "--graph_preset",
        type=str,
        default="",
        choices=["", "autoschema_lightrag", "lightrag_csr", "dynamic_csr", "dynamic_lightrag"],
        help="預設模組化組合"
    )
    
    # 資料參數
    parser.add_argument(
        "--data_type",
        type=str,
        default="DI",
        choices=["DI", "GEN"],
        help="資料類型"
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default="natural_text",
        choices=["natural_text", "markdown", "key_value_text", "unstructured_text"],
        help="資料格式"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="small",
        choices=["small", "big"],
        help="模型大小"
    )
    
    # 測試參數
    parser.add_argument(
        "--qa_dataset_fast_test",
        action="store_true",
        help="快速測試模式（僅評估前 2 題）"
    )
    parser.add_argument(
        "--vector_build_fast_test",
        action="store_true",
        help="Vector 索引快速建立"
    )
    parser.add_argument(
        "--graph_build_fast_test",
        action="store_true",
        help="Graph 索引快速建立"
    )
    
    # 其他參數
    parser.add_argument("--postfix", type=str, default="", help="結果資料夾名稱後綴")
    parser.add_argument("--sup", type=str, default="", help="快取方法標識")
    
    return parser.parse_args()


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
            retrieval_max_tokens=args.retrieval_max_tokens
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
            retrieval_max_tokens=args.retrieval_max_tokens
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
            retrieval_max_tokens=args.retrieval_max_tokens
        )
        wrapper = VectorRAGWrapper(name="Parent_Child_RAG", query_engine=engine)
        wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
        pipelines_to_test.append(wrapper)


def setup_graph_pipelines(args, Settings, pipelines_to_test):
    """設置 Graph RAG Pipelines"""
    if args.graph_rag_method == "none":
        print("Skip Graph RAG")
        return
    
    # Property Graph
    if args.graph_rag_method in ["propertyindex", "all"]:
        engine = get_graph_query_engine(
            Settings,
            data_mode=args.data_mode,
            data_type=args.data_type,
            fast_build=args.graph_build_fast_test,
            graph_method="propertyindex",
            top_k=args.top_k
        )
        if engine is not None:
            pipelines_to_test.append(VectorRAGWrapper(name="Graph_PropertyIndex_RAG", query_engine=engine))
    
    # Dynamic Schema (使用新的 Wrapper)
    if args.graph_rag_method in ["dynamic_schema", "all"]:
        setup_dynamic_schema_pipeline(args, Settings, pipelines_to_test)
    
    # AutoSchemaKG (新增)
    if args.graph_rag_method in ["autoschema", "all"]:
        setup_autoschema_pipeline(args, Settings, pipelines_to_test)
    
    # LightRAG
    if args.graph_rag_method in ["lightrag", "all"]:
        setup_lightrag_pipeline(args, Settings, pipelines_to_test)
    
    # 架構預留方法
    if args.graph_rag_method in ["graphiti", "neo4j", "cq_driven"]:
        print(f"⚠️  {args.graph_rag_method} 端到端方法架構已預留,核心邏輯待實作")


def setup_lightrag_pipeline(args, Settings, pipelines_to_test):
    """設置 LightRAG Pipeline"""
    # 處理插件
    plugins = []
    if args.lightrag_plugins:
        plugins = [p.strip() for p in args.lightrag_plugins.split(",") if p.strip()]
        if plugins:
            print(f"🔌 啟用 LightRAG 插件: {', '.join(plugins)}")
            # 載入插件
            from src.plugins import get_plugin
            for plugin_name in plugins:
                plugin = get_plugin(plugin_name)
                if plugin:
                    print(f"  ✅ 已載入插件: {plugin.get_name()}")
                else:
                    print(f"  ⚠️  插件不存在: {plugin_name}")
    
    # 處理 sup 與 schema 方法
    full_sup = args.sup + f"_{args.lightrag_schema_method}" if args.sup else args.lightrag_schema_method
    
    # 使用 StorageManager（已在 lightrag.py 中整合，此處只需檢查是否需要建圖）
    from src.storage import get_storage_path
    
    # 決定使用的 mode（用於 storage path）
    # 如果指定了單一 mode，使用該 mode；如果是 "all"，則不加 mode 後綴
    storage_mode = "" if args.lightrag_mode == "all" else args.lightrag_mode
    if storage_mode == "none":
        storage_mode = ""
    
    storage_path = get_storage_path(
        storage_type="lightrag",
        data_type=args.data_type,
        method=full_sup,
        mode=storage_mode,
        fast_test=args.graph_build_fast_test or args.qa_dataset_fast_test
    )
    
    # 建立 LightRAG 索引（如果不存在）
    schema_info = None
    if not os.path.exists(storage_path) or not os.listdir(storage_path):
        print(f"📂 建立 LightRAG 索引: {storage_path}")
        
        # 取得完整 Schema 資訊
        schema_info = get_schema_by_method(
            method=args.lightrag_schema_method,
            text_corpus=data_processing(mode=args.data_mode, data_type=args.data_type),
            settings=Settings,
            return_full_schema=True
        )
        print(f"🌟 LightRAG 將使用以下實體類別建圖: {schema_info['entities']}")
        
        # 更新 LightRAG 配置（只傳遞 entity types）
        from llama_index.core import Settings as LlamaSettings
        LlamaSettings.lightrag_entity_types = schema_info['entities']
        
        # 建立索引
        build_lightrag_index(
            Settings,
            mode=args.data_mode,
            data_type=args.data_type,
            sup=full_sup,
            fast_build=args.graph_build_fast_test or args.qa_dataset_fast_test,
            lightrag_mode=storage_mode
        )
    else:
        # 如果索引已存在，從已建立的索引讀取 schema 資訊
        print(f"✅ LightRAG 索引已存在: {storage_path}")
        schema_info = get_schema_by_method(
            method=args.lightrag_schema_method,
            text_corpus=data_processing(mode=args.data_mode, data_type=args.data_type),
            settings=Settings,
            return_full_schema=True
        )
        print(f"🌟 使用現有 Schema: {schema_info['entities']}")
    
    # 取得 LightRAG 實例
    lightrag_instance = get_lightrag_engine(
        Settings, 
        data_type=args.data_type, 
        sup=full_sup,
        mode=storage_mode,
        fast_test=args.graph_build_fast_test or args.qa_dataset_fast_test
    )
    
    # 根據模式建立 Wrapper
    if args.lightrag_mode != "none":
        if args.lightrag_mode == "all":
            modes_to_test = ["local", "global", "mix", "bypass", "hybrid", "naive"]
            for mode in modes_to_test:
                wrapper = LightRAGWrapper(
                    name=f"LightRAG_{mode.capitalize()}",
                    rag_instance=lightrag_instance,
                    mode=mode,
                    schema_info=schema_info
                )
                wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
                pipelines_to_test.append(wrapper)
        elif args.lightrag_mode == "original":
            wrapper = LightRAGWrapper_Original(
                name="LightRAG_Original",
                rag_instance=lightrag_instance,
                mode="hybrid",
                schema_info=schema_info
            )
            wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
            pipelines_to_test.append(wrapper)
        else:
            wrapper = LightRAGWrapper(
                name=f"LightRAG_{args.lightrag_mode.capitalize()}",
                rag_instance=lightrag_instance,
                mode=args.lightrag_mode,
                schema_info=schema_info
            )
            wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
            pipelines_to_test.append(wrapper)
    
    # Temporal LightRAG
    if args.lightrag_temporal_graph:
        temporal_rag = TemporalLightRAGPackage(
            working_dir=os.path.join(Settings.lightrag_config.storage_path_DIR, args.data_type + "_temporal")
        )
        wrapper = TemporalLightRAGWrapper(
            name="Temporal_LightRAG",
            rag_instance=temporal_rag,
            mode=args.lightrag_mode if args.lightrag_mode != "none" else "hybrid",
            schema_info=schema_info
        )
        pipelines_to_test.append(wrapper)


def setup_autoschema_pipeline(args, Settings, pipelines_to_test):
    """設置 AutoSchemaKG 端到端 Pipeline"""
    from src.rag.wrappers import AutoSchemaWrapper
    from src.data.processors import data_processing
    from src.storage import get_storage_path
    
    print("🔬 設置 AutoSchemaKG Pipeline...")
    
    # 載入文檔
    documents = data_processing(mode=args.data_mode, data_type=args.data_type)
    if args.graph_build_fast_test:
        documents = documents[:2]
    
    autoschema_output_dir = get_storage_path(
        storage_type="graph_index",
        data_type=args.data_type,
        method=f"autoschema_{args.model_type}",
        fast_test=args.graph_build_fast_test or args.qa_dataset_fast_test,
        top_k=args.top_k
    )
    print(f"📂 AutoSchemaKG 儲存路徑: {autoschema_output_dir}")

    # 建立 Wrapper
    wrapper = AutoSchemaWrapper(
        name="AutoSchemaKG",
        output_dir=autoschema_output_dir,
        documents=documents,
        model_type=args.model_type,
        top_k=args.top_k,
        settings=Settings
    )
    wrapper.set_retrieval_max_tokens(args.retrieval_max_tokens)
    
    pipelines_to_test.append(wrapper)
    print("✅ AutoSchemaKG Pipeline 設置完成")


def setup_dynamic_schema_pipeline(args, Settings, pipelines_to_test):
    """設置 DynamicSchema 端到端 Pipeline"""
    from src.rag.wrappers import DynamicSchemaWrapper
    from src.data.processors import data_processing
    
    print("🔍 設置 DynamicSchema Pipeline...")
    
    # 載入文檔
    documents = data_processing(mode=args.data_mode, data_type=args.data_type)
    if args.graph_build_fast_test:
        documents = documents[:2]
    
    # 建立 Wrapper
    wrapper = DynamicSchemaWrapper(
        name="DynamicSchema",
        documents=documents,
        model_type=args.model_type,
        data_type=args.data_type,
        fast_test=args.graph_build_fast_test,
        top_k=args.top_k,
        settings=Settings
    )
    
    pipelines_to_test.append(wrapper)
    print("✅ DynamicSchema Pipeline 設置完成")


def setup_modular_graph_pipeline(args, Settings, pipelines_to_test):
    """設置模組化 Graph Pipeline"""
    from src.rag.pipeline_factory import PipelineFactory
    from src.data.processors import data_processing
    
    # 檢查是否指定了模組化參數
    if not args.graph_preset and not (args.graph_builder and args.graph_retriever):
        return
    
    print("🔧 設置模組化 Graph Pipeline...")
    
    # 載入文檔
    documents = data_processing(mode=args.data_mode, data_type=args.data_type)
    if args.graph_build_fast_test:
        documents = documents[:2]
    
    # Builder 配置
    builder_config = {
        'data_type': args.data_type,
        'fast_test': args.graph_build_fast_test,
        'schema_method': args.lightrag_schema_method,
        'sup': args.sup
    }
    
    # Retriever 配置
    retriever_config = {
        'data_type': args.data_type,
        'fast_test': args.graph_build_fast_test,
        'mode': args.lightrag_mode if args.lightrag_mode != "none" else "hybrid"
    }
    
    # 建立 Pipeline
    try:
        pipeline = PipelineFactory.create_pipeline(
            preset_name=args.graph_preset if args.graph_preset else None,
            builder_name=args.graph_builder if not args.graph_preset else None,
            retriever_name=args.graph_retriever if not args.graph_preset else None,
            settings=Settings,
            documents=documents,
            builder_config=builder_config,
            retriever_config=retriever_config,
            top_k=args.top_k,
            model_type=args.model_type
        )
        
        pipelines_to_test.append(pipeline)
        if hasattr(pipeline, "set_retrieval_max_tokens"):
            pipeline.set_retrieval_max_tokens(args.retrieval_max_tokens)
        print(f"✅ 模組化 Pipeline 設置完成: {pipeline.name}")
        
    except Exception as e:
        print(f"⚠️  模組化 Pipeline 設置失敗: {e}")


def build_postfix(args):
    """
    建立結果資料夾名稱後綴
    
    命名規則與 StorageManager 保持一致：
    {data_type}_{method}_{mode}_{top_k}_{custom_tag}_{sup}_{fast_test}
    """
    parts = []
    
    # 1. Data type
    if args.data_type:
        parts.append(args.data_type)
    
    # 2. Method (vector_method 或 graph_rag_method)
    if args.vector_method != "none":
        parts.append(args.vector_method)
    
    if args.adv_vector_method != "none":
        parts.append(args.adv_vector_method)
    
    if args.graph_rag_method != "none":
        parts.append(args.graph_rag_method)
    
    # 3. Mode (LightRAG mode)
    if args.lightrag_mode != "none" and args.lightrag_mode != "all":
        parts.append(args.lightrag_mode)
    
    # 4. Schema method (如果是 LightRAG)
    if args.graph_rag_method == "lightrag" and args.lightrag_schema_method:
        parts.append(args.lightrag_schema_method)
    
    # 5. Custom postfix
    if args.postfix:
        parts.append(args.postfix)
    
    # 6. Sup
    if args.sup:
        parts.append(args.sup)
    
    # 7. Fast test
    if args.qa_dataset_fast_test:
        parts.append("fast_test")
    
    return "_".join(parts) if parts else ""


def main():
    """主執行函數"""
    # 解析參數
    args = parse_arguments()
    
    # 取得設定
    Settings = get_settings(model_type=args.model_type)
    # apply_generation_cap_to_settings(Settings, args.generation_max_tokens)  # 已隱藏 generation token 限制
    
    # 快速測試模式連動
    if args.qa_dataset_fast_test:
        args.vector_build_fast_test = True
        args.graph_build_fast_test = True
        print("⚡ 已啟用 --qa_dataset_fast_test，自動連動開啟 vector_build_fast_test 與 graph_build_fast_test！")
    
    # 建立 Pipeline 列表
    pipelines_to_test = []
    
    # 設置各種 Pipeline
    setup_vector_pipelines(args, Settings, pipelines_to_test)
    setup_advanced_vector_pipelines(args, Settings, pipelines_to_test)
    setup_graph_pipelines(args, Settings, pipelines_to_test)
    setup_modular_graph_pipeline(args, Settings, pipelines_to_test)

    # 統一套用 retrieval token 上限到所有 wrapper
    for pipeline in pipelines_to_test:
        if hasattr(pipeline, "set_retrieval_max_tokens"):
            pipeline.set_retrieval_max_tokens(args.retrieval_max_tokens)
    
    # 檢查是否有 Pipeline
    if not pipelines_to_test:
        print("⚠️ 未選擇任何 RAG pipeline，請檢查啟動參數。")
        return
    
    # 載入資料集
    if args.data_type == "DI":
        datasets = load_and_normalize_qa_CSR_DI(csv_path=Settings.data_config.qa_file_path_DI)
    elif args.data_type == "GEN":
        datasets = load_and_normalize_qa_CSR_full(jsonl_path=Settings.data_config.qa_file_path_GEN)
    
    # 快速測試模式
    if args.qa_dataset_fast_test:
        print("⚡ 啟用快速測試模式，僅抽取前 2 題進行評估...")
        datasets = datasets[:2]
    else:
        print("啟用完整測試模式，進行完整 QA 評估...")
    
    # 建立 postfix
    postfix = build_postfix(args)
    result_bucket = "test" if args.qa_dataset_fast_test else "exp"
    results_root_dir = f"/home/End_to_End_RAG/results/{result_bucket}"
    print(f"📂 結果分流目錄: {results_root_dir}")
    
    # 如果啟用token budget，使用兩階段評估
    if args.enable_token_budget:
        print("\n🎯 啟用Token Budget控制模式")
        asyncio.run(run_evaluation_with_token_budget(
            datasets, 
            pipelines_to_test, 
            postfix=postfix,
            baseline_method=args.token_budget_baseline,
            results_root_dir=results_root_dir,
            is_fast_test=args.qa_dataset_fast_test,
        ))
    else:
        # 執行標準評估
        asyncio.run(
            run_evaluation(
                datasets,
                pipelines_to_test,
                postfix=postfix,
                results_root_dir=results_root_dir,
                is_fast_test=args.qa_dataset_fast_test,
            )
        )


if __name__ == "__main__":
    main()
