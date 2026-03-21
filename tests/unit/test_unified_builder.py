"""
UnifiedGraphBuilder 單元測試

測試統一 Graph Builder 的功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.graph_builder.unified import UnifiedGraphBuilder, GraphBuilderRegistry
from src.graph_builder.base_builder import BaseGraphBuilder
from llama_index.core import Document
import networkx as nx


class TestUnifiedGraphBuilder(unittest.TestCase):
    """測試 UnifiedGraphBuilder 的基本功能"""
    
    def setUp(self):
        """設置測試資料"""
        self.mock_settings = Mock()
        self.mock_settings.llm = Mock()
        self.mock_settings.embed_model = Mock()
        
        self.test_documents = [
            Document(text="測試文檔 1"),
            Document(text="測試文檔 2")
        ]
    
    def test_builder_initialization(self):
        """測試 Builder 初始化"""
        # 註冊一個測試 builder
        class TestBuilder(BaseGraphBuilder):
            def __init__(self, settings, **kwargs):
                super().__init__(settings)
                self.settings = settings
            
            def get_name(self):
                return "TestBuilder"
            
            def build(self, documents):
                return {
                    "nodes": [{"id": "test"}],
                    "edges": [],
                    "graph_format": "custom"
                }
        
        GraphBuilderRegistry.register("test_builder", TestBuilder)
        
        # 建立 UnifiedGraphBuilder
        builder = UnifiedGraphBuilder(
            settings=self.mock_settings,
            builder_type="test_builder"
        )
        
        self.assertEqual(builder.builder_type, "test_builder")
        self.assertEqual(builder.get_name(), "Unified[test_builder]")
    
    def test_invalid_builder_type(self):
        """測試無效的 builder 類型"""
        with self.assertRaises(ValueError):
            UnifiedGraphBuilder(
                settings=self.mock_settings,
                builder_type="invalid_builder"
            )
    
    @patch('src.graph_adapter.GraphFormatAdapter.to_networkx')
    def test_build_with_networkx_conversion(self, mock_to_networkx):
        """測試建圖並轉換為 NetworkX"""
        # 註冊測試 builder
        class TestBuilder(BaseGraphBuilder):
            def __init__(self, settings, **kwargs):
                super().__init__(settings)
                self.settings = settings
            
            def get_name(self):
                return "TestBuilder"
            
            def build(self, documents):
                return {
                    "nodes": [{"id": "A"}, {"id": "B"}],
                    "edges": [{"source": "A", "target": "B"}],
                    "graph_format": "custom",
                    "metadata": {"test": True}
                }
        
        GraphBuilderRegistry.register("test_builder_2", TestBuilder)
        
        # 模擬 NetworkX 轉換
        mock_nx_graph = nx.DiGraph()
        mock_nx_graph.add_node("A")
        mock_nx_graph.add_node("B")
        mock_nx_graph.add_edge("A", "B")
        mock_to_networkx.return_value = mock_nx_graph
        
        # 建立並執行
        builder = UnifiedGraphBuilder(
            settings=self.mock_settings,
            builder_type="test_builder_2"
        )
        
        result = builder.build(self.test_documents)
        
        # 驗證
        self.assertEqual(result["graph_format"], "networkx")
        self.assertEqual(result["source_format"], "custom")
        self.assertIn("graph_data", result)
        self.assertIn("metadata", result)
        mock_to_networkx.assert_called_once()


class TestGraphBuilderRegistry(unittest.TestCase):
    """測試 GraphBuilderRegistry"""
    
    def test_register_and_list(self):
        """測試註冊和列出 builders"""
        class DummyBuilder(BaseGraphBuilder):
            def __init__(self, settings, **kwargs):
                super().__init__(settings)
            
            def get_name(self):
                return "Dummy"
            
            def build(self, documents):
                return {}
        
        # 註冊
        GraphBuilderRegistry.register("dummy", DummyBuilder)
        
        # 檢查是否註冊成功
        self.assertTrue(GraphBuilderRegistry.is_registered("dummy"))
        self.assertIn("dummy", GraphBuilderRegistry.list_available())
    
    def test_create_builder(self):
        """測試建立 builder 實例"""
        class DummyBuilder2(BaseGraphBuilder):
            def __init__(self, settings, **kwargs):
                super().__init__(settings)
                self.settings = settings
                self.extra = kwargs.get("extra")
            
            def get_name(self):
                return "Dummy2"
            
            def build(self, documents):
                return {}
        
        GraphBuilderRegistry.register("dummy2", DummyBuilder2)
        
        # 建立實例
        mock_settings = Mock()
        builder = GraphBuilderRegistry.create("dummy2", settings=mock_settings, extra="test")
        
        self.assertIsInstance(builder, DummyBuilder2)
        self.assertEqual(builder.settings, mock_settings)
        self.assertEqual(builder.extra, "test")


if __name__ == "__main__":
    unittest.main()
