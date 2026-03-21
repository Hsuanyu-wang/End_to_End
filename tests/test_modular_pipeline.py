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


def test_lightrag_builder_propagates_entity_types(monkeypatch, tmp_path):
    """測試傳入 entity_types 時，會寫回 settings.lightrag_config.entity_types"""
    import os
    from llama_index.core import Document
    from src.graph_builder.lightrag_builder import LightRAGBuilder

    # dummy settings (只提供 lightrag_config.entity_types)
    class DummyLightragConfig:
        def __init__(self):
            self.entity_types = ["OldType"]

    class DummySettings:
        def __init__(self):
            self.lightrag_config = DummyLightragConfig()

    settings = DummySettings()

    # stub storage path：確保目錄存在但為空，讓 build 走到 "建圖" 分支
    storage_dir = tmp_path / "lightrag_build"
    os.makedirs(storage_dir, exist_ok=True)

    def fake_get_storage_path(*args, **kwargs):
        # 保持為空目錄，避免 build 直接跳過
        os.makedirs(storage_dir, exist_ok=True)
        return str(storage_dir)

    monkeypatch.setattr("src.storage.get_storage_path", fake_get_storage_path, raising=True)

    # stub lightrag 重建/engine，避免真的跑 LightRAG
    import src.rag.graph.lightrag as lightrag_graph

    called = {"build": False, "engine": False}

    def fake_build_lightrag_index(Settings, mode, data_type, sup, fast_build, **kwargs):
        called["build"] = True
        # build 時應該已同步好 entity_types
        assert Settings.lightrag_config.entity_types == ["Person", "Organization"]

    def fake_get_lightrag_engine(Settings, data_type="DI", sup="", fast_test=False, **kwargs):
        called["engine"] = True
        return object()

    monkeypatch.setattr(lightrag_graph, "build_lightrag_index", fake_build_lightrag_index)
    monkeypatch.setattr(lightrag_graph, "get_lightrag_engine", fake_get_lightrag_engine)

    builder = LightRAGBuilder(
        graph_store=None,
        settings=settings,
        data_type="DI",
        schema_method="lightrag_default",
        sup="",
        fast_test=False,
        entity_types=["Person", "Organization"],
    )

    result = builder.build([Document(text="hello")])

    assert called["build"] is True
    assert called["engine"] is True
    assert settings.lightrag_config.entity_types == ["Person", "Organization"]
    assert result["schema_info"]["entities"] == ["Person", "Organization"]


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
