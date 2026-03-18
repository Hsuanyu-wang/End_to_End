"""
模組化 Pipeline 測試

測試 Builder, Retriever 和 Factory 的基本功能
"""

import pytest
import sys
sys.path.insert(0, '/home/End_to_End_RAG')

from llama_index.core import Document


def test_graph_data_creation():
    """測試 GraphData 物件建立"""
    from src.graph_retriever.base_retriever import GraphData
    
    graph_data = GraphData(
        nodes=[{"id": "1", "label": "Test"}],
        edges=[{"source": "1", "target": "2"}],
        metadata={"test": True},
        schema_info={"entities": ["Test"]},
        storage_path="/tmp/test",
        graph_format="custom"
    )
    
    assert len(graph_data.nodes) == 1
    assert len(graph_data.edges) == 1
    assert graph_data.metadata["test"] == True
    assert graph_data.graph_format == "custom"
    
    # 測試轉換為字典
    data_dict = graph_data.to_dict()
    assert "nodes" in data_dict
    assert "edges" in data_dict
    assert "schema_info" in data_dict
    
    # 測試從字典建立
    graph_data2 = GraphData.from_dict(data_dict)
    assert len(graph_data2.nodes) == 1


def test_pipeline_factory_list_presets():
    """測試 PipelineFactory 列出預設組合"""
    from src.rag.pipeline_factory import PipelineFactory
    
    presets = PipelineFactory.list_presets()
    
    assert "autoschema_lightrag" in presets
    assert "lightrag_csr" in presets
    assert "dynamic_csr" in presets
    assert "dynamic_lightrag" in presets
    
    # 檢查描述
    assert "AutoSchemaKG" in presets["autoschema_lightrag"]
    assert "LightRAG" in presets["autoschema_lightrag"]


def test_lightrag_builder_initialization():
    """測試 LightRAG Builder 初始化"""
    from src.graph_builder.lightrag_builder import LightRAGBuilder
    
    builder = LightRAGBuilder(
        graph_store=None,
        settings=None,
        data_type="DI",
        schema_method="lightrag_default",
        fast_test=True
    )
    
    assert builder.get_name() == "LightRAG"
    assert builder.data_type == "DI"
    assert builder.fast_test == True


def test_dynamic_schema_builder_initialization():
    """測試 DynamicSchema Builder 初始化"""
    from src.graph_builder.dynamic_schema_builder import DynamicSchemaBuilder
    
    builder = DynamicSchemaBuilder(
        graph_store=None,
        settings=None,
        data_type="DI",
        fast_test=True
    )
    
    assert builder.get_name() == "DynamicSchema"
    assert builder.data_type == "DI"


def test_lightrag_retriever_initialization():
    """測試 LightRAG Retriever 初始化"""
    from src.graph_retriever.lightrag_retriever import LightRAGRetriever
    
    retriever = LightRAGRetriever(
        mode="hybrid",
        settings=None,
        data_type="DI",
        fast_test=True
    )
    
    assert retriever.get_name() == "LightRAG_Hybrid"
    assert retriever.mode == "hybrid"


def test_autoschema_builder_initialization():
    """測試 AutoSchemaKG Builder 初始化"""
    from src.graph_builder.autoschema_builder import AutoSchemaKGBuilder
    
    builder = AutoSchemaKGBuilder(
        graph_store=None,
        settings=None,
        output_dir="/tmp/autoschema_test"
    )
    
    assert builder.get_name() == "AutoSchemaKG"
    assert builder.output_dir == "/tmp/autoschema_test"


def test_base_classes_abstract():
    """測試基類是否正確定義為抽象類"""
    from src.graph_builder.base_builder import BaseGraphBuilder
    from src.graph_retriever.base_retriever import BaseGraphRetriever
    
    # 應該無法直接實例化抽象類
    with pytest.raises(TypeError):
        BaseGraphBuilder()
    
    with pytest.raises(TypeError):
        BaseGraphRetriever()


if __name__ == "__main__":
    # 執行測試
    pytest.main([__file__, "-v"])
