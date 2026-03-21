"""
ModularGraphWrapper 單元測試

測試 ModularGraphWrapper 的格式轉換功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.rag.wrappers import ModularGraphWrapper
from llama_index.core import Document
import networkx as nx


class TestModularGraphWrapper(unittest.TestCase):
    """測試 ModularGraphWrapper 的基本功能"""
    
    def setUp(self):
        """設置測試資料"""
        self.mock_builder = Mock()
        self.mock_builder.get_name.return_value = "TestBuilder"
        
        self.mock_retriever = Mock()
        self.mock_retriever.get_name.return_value = "TestRetriever"
        
        # 模擬建圖結果
        self.mock_graph_result = {
            "nodes": [{"id": "A"}, {"id": "B"}],
            "edges": [{"source": "A", "target": "B"}],
            "graph_format": "custom",
            "metadata": {"test": True},
            "schema_info": {"entities": ["Entity1"]},
            "storage_path": "/test/path"
        }
        
        self.mock_builder.build.return_value = self.mock_graph_result
        
        self.test_documents = [
            Document(text="測試文檔 1"),
            Document(text="測試文檔 2")
        ]
    
    def test_wrapper_initialization_without_documents(self):
        """測試不帶文檔的初始化"""
        wrapper = ModularGraphWrapper(
            name="TestWrapper",
            builder=self.mock_builder,
            retriever=self.mock_retriever,
            enable_format_conversion=True
        )
        
        self.assertEqual(wrapper.name, "TestWrapper")
        self.assertEqual(wrapper.builder, self.mock_builder)
        self.assertEqual(wrapper.retriever, self.mock_retriever)
        self.assertTrue(wrapper.enable_format_conversion)
        self.assertIsNone(wrapper.graph_data)
    
    def test_wrapper_initialization_with_documents(self):
        """測試帶文檔的初始化（立即建圖）"""
        wrapper = ModularGraphWrapper(
            name="TestWrapper",
            builder=self.mock_builder,
            retriever=self.mock_retriever,
            documents=self.test_documents,
            enable_format_conversion=True
        )
        
        # 驗證建圖被呼叫
        self.mock_builder.build.assert_called_once()
        self.assertIsNotNone(wrapper.graph_data)
        self.assertEqual(wrapper.graph_data.graph_format, "custom")
    
    def test_format_conversion_disabled(self):
        """測試關閉格式轉換"""
        wrapper = ModularGraphWrapper(
            name="TestWrapper",
            builder=self.mock_builder,
            retriever=self.mock_retriever,
            enable_format_conversion=False
        )
        
        self.assertFalse(wrapper.enable_format_conversion)
        self.assertIsNone(wrapper.adapter)
    
    @patch('src.graph_adapter.GraphFormatAdapter')
    def test_ensure_compatible_format(self, mock_adapter_class):
        """測試格式轉換邏輯"""
        # 建立 NetworkX graph
        nx_graph = nx.DiGraph()
        nx_graph.add_node("A")
        nx_graph.add_node("B")
        nx_graph.add_edge("A", "B")
        
        # 模擬建圖結果包含 NetworkX
        graph_result_with_nx = {
            "graph_data": nx_graph,
            "graph_format": "networkx",
            "nodes": [],
            "edges": []
        }
        
        self.mock_builder.build.return_value = graph_result_with_nx
        
        # 設置 retriever 需要的格式
        self.mock_retriever.required_format = "property_graph"
        
        # 模擬 adapter 的轉換方法
        mock_adapter = Mock()
        mock_pg_index = Mock()
        mock_adapter.networkx_to_pg.return_value = mock_pg_index
        mock_adapter_class.return_value = mock_adapter
        
        # 建立 wrapper（會觸發建圖和格式轉換）
        wrapper = ModularGraphWrapper(
            name="TestWrapper",
            builder=self.mock_builder,
            retriever=self.mock_retriever,
            documents=self.test_documents,
            enable_format_conversion=True
        )
        
        # 驗證建圖被呼叫
        self.mock_builder.build.assert_called_once()
    
    def test_build_graph(self):
        """測試 _build_graph 方法"""
        wrapper = ModularGraphWrapper(
            name="TestWrapper",
            builder=self.mock_builder,
            retriever=self.mock_retriever,
            enable_format_conversion=False
        )
        
        # 手動呼叫 _build_graph
        wrapper._build_graph(self.test_documents)
        
        # 驗證
        self.mock_builder.build.assert_called_once()
        self.assertIsNotNone(wrapper.graph_data)
        self.assertEqual(len(wrapper.graph_data.nodes), 2)
        self.assertEqual(len(wrapper.graph_data.edges), 1)
        self.assertEqual(wrapper.graph_data.graph_format, "custom")


class TestModularGraphWrapperIntegration(unittest.TestCase):
    """測試 ModularGraphWrapper 的整合場景"""
    
    @patch('src.config.my_settings')
    def test_execute_query(self, mock_settings):
        """測試完整的查詢流程"""
        # 建立 mock objects
        mock_builder = Mock()
        mock_builder.get_name.return_value = "TestBuilder"
        mock_builder.build.return_value = {
            "nodes": [{"id": "A"}],
            "edges": [],
            "graph_format": "custom",
            "metadata": {},
            "schema_info": {}
        }
        
        mock_retriever = Mock()
        mock_retriever.get_name.return_value = "TestRetriever"
        mock_retriever.retrieve.return_value = {
            "contexts": ["檢索到的內容"],
            "nodes": []
        }
        
        # 模擬 LLM
        mock_llm = Mock()
        mock_llm.complete.return_value = "生成的答案"
        mock_settings.llm = mock_llm
        
        # 建立 wrapper
        wrapper = ModularGraphWrapper(
            name="TestWrapper",
            builder=mock_builder,
            retriever=mock_retriever,
            documents=[Document(text="測試")],
            enable_format_conversion=False
        )
        
        # 執行查詢（需要使用 asyncio）
        import asyncio
        result = asyncio.run(wrapper._execute_query("測試問題"))
        
        # 驗證
        self.assertIn("generated_answer", result)
        self.assertIn("retrieved_contexts", result)
        self.assertEqual(result["generated_answer"], "生成的答案")
        mock_retriever.retrieve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
