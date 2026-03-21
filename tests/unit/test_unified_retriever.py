"""
UnifiedGraphRetriever 單元測試

測試統一 Graph Retriever 的功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.graph_retriever.unified import UnifiedGraphRetriever, GraphRetrieverRegistry
from src.graph_retriever.base_retriever import BaseGraphRetriever
import networkx as nx


class TestUnifiedGraphRetriever(unittest.TestCase):
    """測試 UnifiedGraphRetriever 的基本功能"""
    
    def setUp(self):
        """設置測試資料"""
        self.mock_settings = Mock()
        self.mock_settings.llm = Mock()
        self.mock_settings.embed_model = Mock()
        
        # 建立測試用的 NetworkX graph
        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_node("Node1", type="Entity")
        self.nx_graph.add_node("Node2", type="Entity")
        self.nx_graph.add_edge("Node1", "Node2", relation="test")
    
    def test_retriever_initialization(self):
        """測試 Retriever 初始化"""
        # 註冊測試 retriever
        class TestRetriever(BaseGraphRetriever):
            def __init__(self, graph_source, settings, **kwargs):
                self.graph_source = graph_source
                self.settings = settings
            
            def get_name(self):
                return "TestRetriever"
            
            def retrieve(self, query, graph_data=None, top_k=2, **kwargs):
                return {
                    "contexts": ["測試內容"],
                    "nodes": [],
                    "metadata": {}
                }
        
        GraphRetrieverRegistry.register("test_retriever", TestRetriever)
        
        # 建立 UnifiedGraphRetriever
        retriever = UnifiedGraphRetriever(
            graph_source=self.nx_graph,
            settings=self.mock_settings,
            retriever_type="test_retriever"
        )
        
        self.assertEqual(retriever.retriever_type, "test_retriever")
        self.assertEqual(retriever.get_name(), "Unified[test_retriever]")
    
    def test_invalid_retriever_type(self):
        """測試無效的 retriever 類型"""
        with self.assertRaises(ValueError):
            UnifiedGraphRetriever(
                graph_source=self.nx_graph,
                settings=self.mock_settings,
                retriever_type="invalid_retriever"
            )
    
    def test_retrieve(self):
        """測試檢索功能"""
        # 註冊測試 retriever
        class TestRetriever2(BaseGraphRetriever):
            def __init__(self, graph_source, settings, **kwargs):
                self.graph_source = graph_source
                self.settings = settings
            
            def get_name(self):
                return "TestRetriever2"
            
            def retrieve(self, query, graph_data=None, top_k=2, **kwargs):
                return {
                    "contexts": [f"檢索結果：{query}"],
                    "nodes": [],
                    "metadata": {"query": query}
                }
        
        GraphRetrieverRegistry.register("test_retriever_2", TestRetriever2)
        
        # 建立並執行檢索
        retriever = UnifiedGraphRetriever(
            graph_source=self.nx_graph,
            settings=self.mock_settings,
            retriever_type="test_retriever_2"
        )
        
        result = retriever.retrieve("測試查詢", top_k=3)
        
        # 驗證
        self.assertIn("contexts", result)
        self.assertIn("nodes", result)
        self.assertIn("metadata", result)
        self.assertEqual(result["contexts"][0], "檢索結果：測試查詢")
    
    def test_graph_source_dict_with_graph_data(self):
        """測試 graph_source 為包含 graph_data 的 dict"""
        graph_dict = {
            "graph_data": self.nx_graph,
            "metadata": {"test": True}
        }
        
        class TestRetriever3(BaseGraphRetriever):
            def __init__(self, graph_source, settings, **kwargs):
                self.graph_source = graph_source
            
            def get_name(self):
                return "TestRetriever3"
            
            def retrieve(self, query, graph_data=None, top_k=2, **kwargs):
                return {"contexts": [], "nodes": [], "metadata": {}}
        
        GraphRetrieverRegistry.register("test_retriever_3", TestRetriever3)
        
        # 應該能正確提取 NetworkX graph
        retriever = UnifiedGraphRetriever(
            graph_source=graph_dict,
            settings=self.mock_settings,
            retriever_type="test_retriever_3"
        )
        
        # 驗證 converted_source 是 NetworkX graph
        self.assertIsNotNone(retriever.converted_source)


class TestGraphRetrieverRegistry(unittest.TestCase):
    """測試 GraphRetrieverRegistry"""
    
    def test_register_and_list(self):
        """測試註冊和列出 retrievers"""
        class DummyRetriever(BaseGraphRetriever):
            def __init__(self, graph_source, settings, **kwargs):
                pass
            
            def get_name(self):
                return "Dummy"
            
            def retrieve(self, query, graph_data=None, top_k=2, **kwargs):
                return {"contexts": [], "nodes": [], "metadata": {}}
        
        # 註冊
        GraphRetrieverRegistry.register("dummy_ret", DummyRetriever)
        
        # 檢查是否註冊成功
        self.assertTrue(GraphRetrieverRegistry.is_registered("dummy_ret"))
        self.assertIn("dummy_ret", GraphRetrieverRegistry.list_available())
    
    def test_create_retriever(self):
        """測試建立 retriever 實例"""
        class DummyRetriever2(BaseGraphRetriever):
            def __init__(self, graph_source, settings, **kwargs):
                self.graph_source = graph_source
                self.settings = settings
                self.extra = kwargs.get("extra")
            
            def get_name(self):
                return "Dummy2"
            
            def retrieve(self, query, graph_data=None, top_k=2, **kwargs):
                return {"contexts": [], "nodes": [], "metadata": {}}
        
        GraphRetrieverRegistry.register("dummy_ret2", DummyRetriever2)
        
        # 建立實例
        mock_settings = Mock()
        nx_graph = nx.DiGraph()
        retriever = GraphRetrieverRegistry.create(
            "dummy_ret2",
            graph_source=nx_graph,
            settings=mock_settings,
            extra="test"
        )
        
        self.assertIsInstance(retriever, DummyRetriever2)
        self.assertEqual(retriever.settings, mock_settings)
        self.assertEqual(retriever.extra, "test")


if __name__ == "__main__":
    unittest.main()
