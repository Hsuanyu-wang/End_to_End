"""
Pipeline Factory

提供模組化 Graph Pipeline 的組合與建立功能
"""

from typing import Dict, Any, Optional
from src.rag.wrappers.modular_graph_wrapper import ModularGraphWrapper
from src.graph_builder.base_builder import BaseGraphBuilder
from src.graph_retriever.base_retriever import BaseGraphRetriever


# 預設組合配置
PRESET_PIPELINES = {
    "autoschema_lightrag": {
        "builder": "autoschema",
        "retriever": "lightrag",
        "description": "AutoSchemaKG Builder + LightRAG Retriever"
    },
    "lightrag_csr": {
        "builder": "lightrag",
        "retriever": "csr",
        "description": "LightRAG Builder + CSR Graph Retriever"
    },
    "dynamic_csr": {
        "builder": "dynamic",
        "retriever": "csr",
        "description": "DynamicSchema Builder + CSR Graph Retriever"
    },
    "dynamic_lightrag": {
        "builder": "dynamic",
        "retriever": "lightrag",
        "description": "DynamicSchema Builder + LightRAG Retriever"
    },
}


class PipelineFactory:
    """
    Pipeline Factory
    
    提供統一的 Pipeline 建立介面,支援預設組合與自訂組合
    """
    
    @staticmethod
    def create_builder(
        builder_name: str,
        settings: Any = None,
        builder_config: Optional[Dict[str, Any]] = None
    ) -> BaseGraphBuilder:
        """
        建立 Graph Builder
        
        Args:
            builder_name: Builder 名稱(autoschema/lightrag/property/dynamic)
            settings: LlamaIndex Settings
            builder_config: Builder 配置參數
        
        Returns:
            Builder 實例
        """
        from src.graph_builder import (
            AutoSchemaKGBuilder,
            LightRAGBuilder,
            DynamicSchemaBuilder,
            BaselineGraphBuilder
        )
        
        builder_config = builder_config or {}
        
        if builder_name == "autoschema":
            return AutoSchemaKGBuilder(
                graph_store=None,
                settings=settings,
                output_dir=builder_config.get('output_dir')
            )
        elif builder_name == "lightrag":
            return LightRAGBuilder(
                graph_store=None,
                settings=settings,
                data_type=builder_config.get('data_type', 'DI'),
                schema_method=builder_config.get('schema_method', 'lightrag_default'),
                sup=builder_config.get('sup', ''),
                fast_test=builder_config.get('fast_test', False)
            )
        elif builder_name == "dynamic":
            return DynamicSchemaBuilder(
                graph_store=None,
                settings=settings,
                data_type=builder_config.get('data_type', 'DI'),
                fast_test=builder_config.get('fast_test', False),
                max_triplets_per_chunk=builder_config.get('max_triplets_per_chunk', 20),
                num_workers=builder_config.get('num_workers', 4)
            )
        elif builder_name == "property":
            # PropertyGraph Builder (基於 BaselineGraphBuilder)
            return BaselineGraphBuilder(
                graph_store=None,
                settings=settings
            )
        else:
            raise ValueError(f"未知的 Builder: {builder_name}")
    
    @staticmethod
    def create_retriever(
        retriever_name: str,
        settings: Any = None,
        retriever_config: Optional[Dict[str, Any]] = None
    ) -> BaseGraphRetriever:
        """
        建立 Graph Retriever
        
        Args:
            retriever_name: Retriever 名稱(lightrag/csr/neo4j)
            settings: LlamaIndex Settings
            retriever_config: Retriever 配置參數
        
        Returns:
            Retriever 實例
        """
        from src.graph_retriever import LightRAGRetriever, CSRGraphQueryEngine
        
        retriever_config = retriever_config or {}
        
        if retriever_name == "lightrag":
            return LightRAGRetriever(
                mode=retriever_config.get('mode', 'hybrid'),
                rag_instance=retriever_config.get('rag_instance'),
                settings=settings,
                data_type=retriever_config.get('data_type', 'DI'),
                sup=retriever_config.get('sup', ''),
                fast_test=retriever_config.get('fast_test', False)
            )
        elif retriever_name == "csr":
            # CSR Graph Retriever
            # 注意:CSRGraphQueryEngine 不是標準的 Retriever,需要適配
            print("⚠️  CSR Retriever 整合待完成")
            # 暫時返回 LightRAG Retriever 作為備選
            return LightRAGRetriever(
                mode='hybrid',
                settings=settings,
                data_type=retriever_config.get('data_type', 'DI')
            )
        elif retriever_name == "neo4j":
            # Neo4j Retriever (待實作)
            raise NotImplementedError("Neo4j Retriever 尚未實作")
        else:
            raise ValueError(f"未知的 Retriever: {retriever_name}")
    
    @staticmethod
    def create_pipeline(
        preset_name: str = None,
        builder_name: str = None,
        retriever_name: str = None,
        settings: Any = None,
        documents: list = None,
        builder_config: Optional[Dict[str, Any]] = None,
        retriever_config: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        model_type: str = "small"
    ) -> ModularGraphWrapper:
        """
        建立模組化 Pipeline
        
        Args:
            preset_name: 預設組合名稱(如果指定,會覆蓋 builder_name 和 retriever_name)
            builder_name: Builder 名稱
            retriever_name: Retriever 名稱
            settings: LlamaIndex Settings
            documents: 文檔列表
            builder_config: Builder 配置
            retriever_config: Retriever 配置
            top_k: 檢索數量
            model_type: 模型類型
        
        Returns:
            ModularGraphWrapper 實例
        """
        # 處理預設組合
        if preset_name:
            if preset_name not in PRESET_PIPELINES:
                raise ValueError(f"未知的預設組合: {preset_name}")
            
            preset = PRESET_PIPELINES[preset_name]
            builder_name = preset["builder"]
            retriever_name = preset["retriever"]
            print(f"🎯 使用預設組合: {preset['description']}")
        
        # 檢查參數
        if not builder_name or not retriever_name:
            raise ValueError("必須指定 preset_name 或同時指定 builder_name 和 retriever_name")
        
        # 建立 Builder
        builder = PipelineFactory.create_builder(
            builder_name=builder_name,
            settings=settings,
            builder_config=builder_config
        )
        
        # 建立 Retriever
        retriever = PipelineFactory.create_retriever(
            retriever_name=retriever_name,
            settings=settings,
            retriever_config=retriever_config
        )
        
        # 建立 Pipeline
        pipeline_name = f"{builder.get_name()}+{retriever.get_name()}"
        
        pipeline = ModularGraphWrapper(
            name=pipeline_name,
            builder=builder,
            retriever=retriever,
            documents=documents,
            model_type=model_type,
            top_k=top_k
        )
        
        return pipeline
    
    @staticmethod
    def list_presets() -> Dict[str, str]:
        """
        列出所有預設組合
        
        Returns:
            預設組合字典 {name: description}
        """
        return {name: preset["description"] for name, preset in PRESET_PIPELINES.items()}
