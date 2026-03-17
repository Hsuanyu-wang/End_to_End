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

nest_asyncio.apply()


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
        choices=["none", "propertyindex", "lightrag", "dynamic_schema", "all"],
        help="Graph RAG 方法"
    )
    parser.add_argument(
        "--lightrag_mode",
        type=str,
        default="none",
        choices=["none", "local", "global", "hybrid", "mix", "naive", "bypass", "original", "all"],
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
    
    # 方法參數
    parser.add_argument("--top_k", type=int, default=2, help="檢索數量")
    
    # Schema 相關
    parser.add_argument(
        "--lightrag_schema_method",
        type=str,
        default="lightrag_default",
        choices=["lightrag_default", "iterative_evolution", "llm_dynamic"],
        help="LightRAG Schema 生成方法"
    )
    parser.add_argument(
        "--lightrag_temporal_graph",
        action="store_true",
        help="啟用時序 LightRAG"
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
            fast_build=args.vector_build_fast_test
        )
        if engine is not None:
            pipelines_to_test.append(
                VectorRAGWrapper(name=method_mapping[method], query_engine=engine)
            )


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
            fast_build=args.vector_build_fast_test
        )
        pipelines_to_test.append(VectorRAGWrapper(name="Self_Query_RAG", query_engine=engine))
    
    if args.adv_vector_method in ["parent_child", "all"]:
        engine = get_parent_child_query_engine(
            Settings,
            data_mode=args.data_mode,
            data_type=args.data_type,
            top_k=args.top_k,
            fast_build=args.vector_build_fast_test
        )
        pipelines_to_test.append(VectorRAGWrapper(name="Parent_Child_RAG", query_engine=engine))


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
    
    # Dynamic Schema
    if args.graph_rag_method in ["dynamic_schema", "all"]:
        engine = get_dynamic_schema_graph_query_engine(
            Settings,
            data_mode=args.data_mode,
            data_type=args.data_type,
            fast_build=args.graph_build_fast_test,
            graph_method="dynamic_schema",
            top_k=args.top_k
        )
        if engine is not None:
            pipelines_to_test.append(VectorRAGWrapper(name="Graph_DynamicSchema_RAG", query_engine=engine))
    
    # LightRAG
    if args.graph_rag_method in ["lightrag", "all"]:
        setup_lightrag_pipeline(args, Settings, pipelines_to_test)


def setup_lightrag_pipeline(args, Settings, pipelines_to_test):
    """設置 LightRAG Pipeline"""
    # 處理 sup 與 schema 方法
    full_sup = args.sup + f"_{args.lightrag_schema_method}" if args.sup else args.lightrag_schema_method
    storage_path = os.path.join(Settings.lightrag_storage_path_DIR, args.data_type) + "_" + full_sup
    
    # 建立 LightRAG 索引（如果不存在）
    if not os.path.exists(storage_path):
        if not os.path.exists(Settings.lightrag_storage_path_DIR):
            os.mkdir(Settings.lightrag_storage_path_DIR)
        os.mkdir(storage_path)
        
        # 取得動態 Schema
        custom_entity_types = get_schema_by_method(
            method=args.lightrag_schema_method,
            text_corpus=data_processing(mode=args.data_mode, data_type=args.data_type),
            settings=Settings
        )
        print(f"🌟 LightRAG 將使用以下實體類別建圖: {custom_entity_types}")
        
        # 更新 Settings
        Settings.lightrag_entity_types = custom_entity_types
        
        # 建立索引
        build_lightrag_index(
            Settings,
            mode=args.data_mode,
            data_type=args.data_type,
            sup=full_sup,
            fast_build=args.graph_build_fast_test
        )
    
    # 取得 LightRAG 實例
    lightrag_instance = get_lightrag_engine(Settings, data_type=args.data_type, sup=full_sup)
    
    # 根據模式建立 Wrapper
    if args.lightrag_mode != "none":
        if args.lightrag_mode == "all":
            modes_to_test = ["local", "global", "mix", "bypass", "hybrid", "naive"]
            for mode in modes_to_test:
                wrapper = LightRAGWrapper(
                    name=f"LightRAG_{mode.capitalize()}",
                    rag_instance=lightrag_instance,
                    mode=mode
                )
                pipelines_to_test.append(wrapper)
        elif args.lightrag_mode == "original":
            wrapper = LightRAGWrapper_Original(
                name="LightRAG_Original",
                rag_instance=lightrag_instance,
                mode="hybrid"
            )
            pipelines_to_test.append(wrapper)
        else:
            wrapper = LightRAGWrapper(
                name=f"LightRAG_{args.lightrag_mode.capitalize()}",
                rag_instance=lightrag_instance,
                mode=args.lightrag_mode
            )
            pipelines_to_test.append(wrapper)
    
    # Temporal LightRAG
    if args.lightrag_temporal_graph:
        temporal_rag = TemporalLightRAGPackage(
            working_dir=os.path.join(Settings.lightrag_storage_path_DIR, args.data_type + "_temporal")
        )
        wrapper = TemporalLightRAGWrapper(
            name="Temporal_LightRAG",
            rag_instance=temporal_rag,
            mode=args.lightrag_mode if args.lightrag_mode != "none" else "hybrid"
        )
        pipelines_to_test.append(wrapper)


def build_postfix(args):
    """建立結果資料夾名稱後綴"""
    postfix = ""
    
    if args.data_type == "DI":
        postfix += "_DI"
    elif args.data_type == "GEN":
        postfix += "_GEN"
    
    if args.vector_method != "none":
        postfix += f"_{args.vector_method}"
    
    if args.graph_rag_method != "none":
        postfix += f"_{args.graph_rag_method}"
    
    if args.lightrag_mode != "none":
        postfix += f"_{args.lightrag_mode}"
    
    if args.postfix:
        postfix += f"_{args.postfix}"
    
    if args.sup:
        postfix += f"_{args.sup}"
    
    if args.qa_dataset_fast_test:
        postfix += "_fast_test"
    
    return postfix


def main():
    """主執行函數"""
    # 解析參數
    args = parse_arguments()
    
    # 取得設定
    Settings = get_settings(model_type=args.model_type)
    
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
    
    # 檢查是否有 Pipeline
    if not pipelines_to_test:
        print("⚠️ 未選擇任何 RAG pipeline，請檢查啟動參數。")
        return
    
    # 載入資料集
    if args.data_type == "DI":
        datasets = load_and_normalize_qa_CSR_DI(csv_path=Settings.qa_file_path_DI)
    elif args.data_type == "GEN":
        datasets = load_and_normalize_qa_CSR_full(jsonl_path=Settings.qa_file_path_GEN)
    
    # 快速測試模式
    if args.qa_dataset_fast_test:
        print("⚡ 啟用快速測試模式，僅抽取前 2 題進行評估...")
        datasets = datasets[:2]
    else:
        print("啟用完整測試模式，進行完整 QA 評估...")
    
    # 建立 postfix
    postfix = build_postfix(args)
    
    # 執行評估
    asyncio.run(run_evaluation(datasets, pipelines_to_test, postfix=postfix))


if __name__ == "__main__":
    main()
