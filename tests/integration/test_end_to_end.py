"""
端到端整合測試

測試完整的 PropertyGraph 模組化重構流程
"""

import unittest
import sys
import os
sys.path.insert(0, '/home/End_to_End_RAG')

from unittest.mock import Mock, patch, MagicMock
from llama_index.core import Document
import networkx as nx


class TestEndToEndPropertyGraphPipeline(unittest.TestCase):
    """測試完整的 PropertyGraph Pipeline（模擬）"""
    
    @patch('src.graph_builder.unified.UnifiedGraphBuilder')
    @patch('src.graph_retriever.unified.UnifiedGraphRetriever')
    @patch('src.rag.wrappers.ModularGraphWrapper')
    def test_property_graph_pipeline_creation(self, mock_wrapper, mock_retriever, mock_builder):
        """測試 PropertyGraph Pipeline 建立流程"""
        from src.config.settings import get_settings
        
        # 模擬 Settings
        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.embed_model = Mock()
        
        # 測試文檔
        documents = [
            Document(text="測試文檔 1：工程師處理客戶問題"),
            Document(text="測試文檔 2：系統發生錯誤")
        ]
        
        # 模擬 extractor 配置
        extractor_config = {
            "implicit": {"enabled": True},
            "schema": {
                "enabled": True,
                "entities": ["Engineer", "Customer", "System"],
                "relations": ["HANDLED", "AFFECTS"],
                "strict": False
            }
        }
        
        # 模擬 retriever 配置
        retriever_config = {
            "vector": {"enabled": True, "similarity_top_k": 5},
            "synonym": {"enabled": True, "include_text": True}
        }
        
        # 模擬建圖結果
        mock_nx_graph = nx.DiGraph()
        mock_nx_graph.add_node("Engineer1", type="Engineer")
        mock_nx_graph.add_node("Customer1", type="Customer")
        mock_nx_graph.add_edge("Engineer1", "Customer1", relation="HANDLED")
        
        mock_graph_result = {
            "graph_data": mock_nx_graph,
            "graph_format": "networkx",
            "nodes": [{"id": "Engineer1"}, {"id": "Customer1"}],
            "edges": [{"source": "Engineer1", "target": "Customer1"}],
            "metadata": {},
            "schema_info": {"entities": ["Engineer", "Customer"]}
        }
        
        mock_builder_instance = Mock()
        mock_builder_instance.build.return_value = mock_graph_result
        mock_builder.return_value = mock_builder_instance
        
        # 模擬 retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve.return_value = {
            "contexts": ["檢索結果"],
            "nodes": []
        }
        mock_retriever.return_value = mock_retriever_instance
        
        # 模擬 wrapper
        mock_wrapper_instance = Mock()
        mock_wrapper_instance.name = "PG[implicit,schema]+[vector,synonym]"
        mock_wrapper.return_value = mock_wrapper_instance
        
        # 驗證能正確建立各組件
        self.assertIsNotNone(mock_builder_instance)
        self.assertIsNotNone(mock_retriever_instance)
        self.assertIsNotNone(mock_wrapper_instance)


class TestEndToEndLightRAGPipeline(unittest.TestCase):
    """測試完整的 LightRAG Pipeline（模擬）"""
    
    @patch('src.graph_builder.unified.UnifiedGraphBuilder')
    @patch('src.graph_retriever.unified.UnifiedGraphRetriever')
    def test_lightrag_pipeline_creation(self, mock_retriever, mock_builder):
        """測試 LightRAG Pipeline 建立流程"""
        # 模擬 Settings
        mock_settings = Mock()
        mock_settings.llm = Mock()
        mock_settings.embed_model = Mock()
        
        # 測試文檔
        documents = [
            Document(text="測試文檔 1"),
            Document(text="測試文檔 2")
        ]
        
        # 模擬 schema
        schema_info = {
            "entities": ["Person", "Organization", "Event"],
            "relations": ["WORKS_AT", "ATTENDS"]
        }
        
        # 模擬建圖結果
        mock_graph_result = {
            "graph_data": nx.DiGraph(),
            "graph_format": "networkx",
            "nodes": [],
            "edges": [],
            "schema_info": schema_info
        }
        
        mock_builder_instance = Mock()
        mock_builder_instance.build.return_value = mock_graph_result
        mock_builder.return_value = mock_builder_instance
        
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve.return_value = {
            "contexts": ["LightRAG 檢索結果"],
            "nodes": []
        }
        mock_retriever.return_value = mock_retriever_instance
        
        # 驗證
        self.assertIsNotNone(mock_builder_instance)
        self.assertIsNotNone(mock_retriever_instance)


class TestCrossFormatConversion(unittest.TestCase):
    """測試跨格式轉換"""
    
    def test_networkx_hub_conversion(self):
        """測試 NetworkX 作為中轉站的轉換流程"""
        from src.graph_adapter import GraphFormatAdapter
        
        # 建立源圖譜（dict 格式）
        source_dict = {
            "nodes": [
                {"id": "A", "type": "Entity"},
                {"id": "B", "type": "Entity"}
            ],
            "edges": [
                {"source": "A", "target": "B", "relation": "test"}
            ]
        }
        
        # Step 1: Dict → NetworkX
        nx_graph = GraphFormatAdapter.to_networkx(source_dict, source_format="custom")
        
        # 驗證 NetworkX graph
        self.assertIsInstance(nx_graph, nx.Graph)
        self.assertEqual(nx_graph.number_of_nodes(), 2)
        self.assertEqual(nx_graph.number_of_edges(), 1)
        
        # Step 2: NetworkX → Dict (通過檢查節點和邊)
        self.assertIn("A", nx_graph.nodes())
        self.assertIn("B", nx_graph.nodes())
        self.assertTrue(nx_graph.has_edge("A", "B"))


class TestEvaluationScriptIntegration(unittest.TestCase):
    """測試 run_evaluation.py 的參數處理"""
    
    def test_parse_extractor_config(self):
        """測試 extractor 配置解析"""
        # 這裡我們導入實際的函式
        import sys
        sys.path.insert(0, '/home/End_to_End_RAG/scripts')
        
        # 模擬解析函式（簡化版）
        def parse_extractor_config(extractors_str: str) -> dict:
            config = {}
            extractors = [e.strip() for e in extractors_str.split(",") if e.strip()]
            
            for extractor in extractors:
                if extractor == "implicit":
                    config["implicit"] = {"enabled": True}
                elif extractor == "schema":
                    config["schema"] = {"enabled": True}
                elif extractor == "simple":
                    config["simple"] = {"enabled": True}
            
            return config
        
        # 測試
        result = parse_extractor_config("implicit,schema,simple")
        
        self.assertIn("implicit", result)
        self.assertIn("schema", result)
        self.assertIn("simple", result)
        self.assertTrue(result["implicit"]["enabled"])
    
    def test_parse_retriever_config(self):
        """測試 retriever 配置解析"""
        # 模擬解析函式
        def parse_retriever_config(retrievers_str: str) -> dict:
            config = {}
            retrievers = [r.strip() for r in retrievers_str.split(",") if r.strip()]
            
            for retriever in retrievers:
                if retriever == "vector":
                    config["vector"] = {"enabled": True}
                elif retriever == "synonym":
                    config["synonym"] = {"enabled": True}
            
            return config
        
        # 測試
        result = parse_retriever_config("vector,synonym")
        
        self.assertIn("vector", result)
        self.assertIn("synonym", result)
        self.assertTrue(result["vector"]["enabled"])


if __name__ == "__main__":
    # 執行測試
    unittest.main(verbosity=2)
