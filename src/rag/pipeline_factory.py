"""
Pipeline Factory

提供模組化 Graph Pipeline 的組合與建立功能。
透過 GraphBuilderRegistry / GraphRetrieverRegistry 動態解析 builder/retriever 名稱。
"""

from typing import Dict, Any, Optional
from src.rag.wrappers.modular_graph_wrapper import ModularGraphWrapper
from src.graph_builder.base_builder import BaseGraphBuilder
from src.graph_retriever.base_retriever import BaseGraphRetriever


class PipelineFactory:
    """
    Pipeline Factory

    透過 builder_name / retriever_name 建立模組化 Pipeline。
    名稱由 GraphBuilderRegistry 和 GraphRetrieverRegistry（或直接 import）解析。
    """

    @staticmethod
    def create_builder(
        builder_name: str,
        settings: Any = None,
        builder_config: Optional[Dict[str, Any]] = None,
    ) -> BaseGraphBuilder:
        """
        建立 Graph Builder

        Args:
            builder_name: Builder 名稱（autoschema / lightrag / property / dynamic）
            settings: LlamaIndex Settings
            builder_config: Builder 配置參數

        Returns:
            Builder 實例
        """
        from src.graph_builder import (
            AutoSchemaKGBuilder,
            LightRAGBuilder,
            DynamicSchemaBuilder,
            BaselineGraphBuilder,
        )

        builder_config = builder_config or {}

        if builder_name == "autoschema":
            from src.storage import get_autoschema_output_dir

            output_dir = get_autoschema_output_dir(
                data_type=builder_config.get("data_type", "DI"),
                data_mode=builder_config.get("data_mode") or "",
                sup=builder_config.get("sup", "") or "",
                fast_test=bool(builder_config.get("fast_test", False)),
                root_dir=builder_config.get("storage_root"),
            )
            return AutoSchemaKGBuilder(
                graph_store=None,
                settings=settings,
                output_dir=output_dir,
            )
        elif builder_name == "lightrag":
            return LightRAGBuilder(
                graph_store=None,
                settings=settings,
                data_type=builder_config.get("data_type", "DI"),
                schema_method=builder_config.get("schema_method", "lightrag_default"),
                sup=builder_config.get("sup", ""),
                fast_test=builder_config.get("fast_test", False),
            )
        elif builder_name == "dynamic":
            return DynamicSchemaBuilder(
                graph_store=None,
                settings=settings,
                data_type=builder_config.get("data_type", "DI"),
                fast_test=builder_config.get("fast_test", False),
                max_triplets_per_chunk=builder_config.get("max_triplets_per_chunk", 20),
                num_workers=builder_config.get("num_workers", 4),
            )
        elif builder_name == "property":
            return BaselineGraphBuilder(
                graph_store=None,
                settings=settings,
                data_type=builder_config.get("data_type", "DI"),
                fast_test=builder_config.get("fast_test", False),
            )
        else:
            raise ValueError(f"未知的 Builder: {builder_name}")

    @staticmethod
    def create_retriever(
        retriever_name: str,
        settings: Any = None,
        retriever_config: Optional[Dict[str, Any]] = None,
    ) -> BaseGraphRetriever:
        """
        建立 Graph Retriever

        Args:
            retriever_name: Retriever 名稱（lightrag / csr / neo4j）
            settings: LlamaIndex Settings
            retriever_config: Retriever 配置參數

        Returns:
            Retriever 實例
        """
        from src.graph_retriever import LightRAGRetriever

        retriever_config = retriever_config or {}

        if retriever_name == "lightrag":
            return LightRAGRetriever(
                mode=retriever_config.get("mode", "hybrid"),
                rag_instance=retriever_config.get("rag_instance"),
                settings=settings,
                data_type=retriever_config.get("data_type", "DI"),
                sup=retriever_config.get("sup", ""),
                fast_test=retriever_config.get("fast_test", False),
            )
        elif retriever_name == "neo4j":
            raise NotImplementedError("Neo4j Retriever 尚未實作")
        else:
            raise ValueError(f"未知的 Retriever: {retriever_name}")

    @staticmethod
    def create_pipeline(
        builder_name: str = None,
        retriever_name: str = None,
        settings: Any = None,
        documents: list = None,
        builder_config: Optional[Dict[str, Any]] = None,
        retriever_config: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        model_type: str = "small",
        preset_name: str = None,
    ) -> ModularGraphWrapper:
        """
        建立模組化 Pipeline

        Args:
            builder_name: Builder 名稱
            retriever_name: Retriever 名稱
            settings: LlamaIndex Settings
            documents: 文檔列表
            builder_config: Builder 配置
            retriever_config: Retriever 配置
            top_k: 檢索數量
            model_type: 模型類型
            preset_name: [DEPRECATED] 已移除 preset 機制，請直接指定 builder_name + retriever_name

        Returns:
            ModularGraphWrapper 實例
        """
        if preset_name:
            import warnings
            warnings.warn(
                "preset_name 已棄用，請直接指定 builder_name 和 retriever_name",
                DeprecationWarning,
                stacklevel=2,
            )

        if not builder_name or not retriever_name:
            raise ValueError("必須同時指定 builder_name 和 retriever_name")

        builder = PipelineFactory.create_builder(
            builder_name=builder_name,
            settings=settings,
            builder_config=builder_config,
        )

        retriever = PipelineFactory.create_retriever(
            retriever_name=retriever_name,
            settings=settings,
            retriever_config=retriever_config,
        )

        pipeline_name = f"{builder.get_name()}+{retriever.get_name()}"

        pipeline = ModularGraphWrapper(
            name=pipeline_name,
            builder=builder,
            retriever=retriever,
            documents=documents,
            model_type=model_type,
            top_k=top_k,
        )

        return pipeline
