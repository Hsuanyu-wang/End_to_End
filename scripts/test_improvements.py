#!/usr/bin/env python3
"""
快速驗證腳本

驗證改進後的核心功能是否正常運作
"""

import os
import sys

sys.path.insert(0, '/home/End_to_End_RAG')

def test_storage_manager():
    """測試 StorageManager"""
    print("=" * 60)
    print("測試 1: StorageManager")
    print("=" * 60)
    
    from src.storage import get_storage_path, get_csr_graph_path
    
    # 測試 vector_index 路徑
    path = get_storage_path("vector_index", "DI", "hybrid", fast_test=False)
    print(f"✅ Vector Index 路徑: {path}")
    assert "/storage/vector_index/DI_hybrid" in path
    
    # 測試 lightrag 路徑
    path = get_storage_path("lightrag", "DI", "lightrag_default", fast_test=True)
    print(f"✅ LightRAG 路徑 (fast_test): {path}")
    assert "/storage/lightrag/DI_lightrag_default_fast_test" in path
    
    # 測試 CSR graph 路徑
    path = get_csr_graph_path("DI", "natural_text", "khop", fast_test=False)
    print(f"✅ CSR Graph 路徑: {path}")
    assert "/storage/csr_graph/DI_natural_text_khop.pkl" in path
    
    print("✅ StorageManager 測試通過\n")


def test_chunk_id_mapper():
    """測試 ChunkIDMapper"""
    print("=" * 60)
    print("測試 2: ChunkIDMapper")
    print("=" * 60)
    
    from src.rag.graph.lightrag_id_mapper import ChunkIDMapper
    import tempfile
    
    # 建立臨時目錄
    with tempfile.TemporaryDirectory() as tmpdir:
        mapper = ChunkIDMapper(tmpdir)
        
        # 測試新增映射
        mapper.add_mapping("chunk-abc123", "963ba04d-ff07-484d-ad8e-b11e33fe8044")
        mapper.add_mapping("chunk-def456", "c40aa050-2287-467c-9d0e-25902db9da40")
        
        # 測試查詢
        original_no = mapper.get_original_no("chunk-abc123")
        print(f"✅ 映射查詢: chunk-abc123 -> {original_no}")
        assert original_no == "963ba04d-ff07-484d-ad8e-b11e33fe8044"
        
        # 測試批次查詢
        nos = mapper.get_original_nos(["chunk-abc123", "chunk-def456", "chunk-unknown"])
        print(f"✅ 批次查詢: {len(nos)} 個結果")
        assert len(nos) == 3
        
        print(f"✅ ChunkIDMapper 包含 {len(mapper)} 個映射")
    
    print("✅ ChunkIDMapper 測試通過\n")


def test_plugin_system():
    """測試插件系統"""
    print("=" * 60)
    print("測試 3: 插件系統")
    print("=" * 60)
    
    from src.plugins import PluginRegistry, get_plugin, list_available_plugins
    
    # 匯入插件（會自動註冊）
    import src.plugins.autoschema_plugin
    import src.plugins.dynamic_path_plugin
    import src.plugins.graphiti_plugin
    import src.plugins.cq_driven_plugin
    import src.plugins.neo4j_builder_plugin
    
    # 列出可用插件
    plugins = list_available_plugins()
    print(f"✅ 可用插件: {plugins}")
    assert len(plugins) >= 5
    
    # 測試取得插件
    plugin = get_plugin("autoschema")
    if plugin:
        print(f"✅ 載入插件: {plugin.get_name()} - {plugin.get_description()}")
        assert plugin.get_name() == "AutoSchemaKG"
    else:
        print("⚠️  無法載入 autoschema 插件")
    
    print("✅ 插件系統測試通過\n")


def test_data_processor():
    """測試資料處理器"""
    print("=" * 60)
    print("測試 4: 資料處理器 (doc_id)")
    print("=" * 60)
    
    from src.data.processors import DataProcessor
    from llama_index.core import Document
    
    processor = DataProcessor(mode="natural_text", data_type="DI")
    
    # 建立測試資料
    test_data = {
        "NO": "test-uuid-12345",
        "Customer": "測試客戶",
        "Engineers": "測試工程師",
        "Service Start": "2024-01-01",
        "Service End": "2024-01-02",
        "Description": "測試描述",
        "Action": "測試動作"
    }
    
    doc = processor._create_document(test_data)
    
    print(f"✅ Document doc_id: {doc.doc_id}")
    print(f"✅ Document metadata['NO']: {doc.metadata.get('NO')}")
    
    assert doc.doc_id == "test-uuid-12345"
    assert doc.metadata["NO"] == "test-uuid-12345"
    
    print("✅ 資料處理器測試通過\n")


def main():
    """主函數"""
    print("\n" + "=" * 60)
    print("End_to_End_RAG 改進功能驗證")
    print("=" * 60 + "\n")
    
    try:
        test_storage_manager()
        test_chunk_id_mapper()
        test_plugin_system()
        test_data_processor()
        
        print("=" * 60)
        print("✅ 所有測試通過！")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
