"""
GraphFormatAdapter 單元測試

測試圖譜格式轉換功能
"""

import unittest
import networkx as nx
from src.graph_adapter import GraphFormatAdapter


class TestGraphFormatAdapter(unittest.TestCase):
    """測試 GraphFormatAdapter 的基本轉換功能"""
    
    def setUp(self):
        """設置測試資料"""
        # 建立簡單的 NetworkX graph
        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_node("Node1", type="Entity", name="測試節點1")
        self.nx_graph.add_node("Node2", type="Entity", name="測試節點2")
        self.nx_graph.add_edge("Node1", "Node2", relation="測試關係")
        
        # 建立 dict 格式的圖譜
        self.dict_graph = {
            "nodes": [
                {"id": "Node1", "type": "Entity", "name": "測試節點1"},
                {"id": "Node2", "type": "Entity", "name": "測試節點2"}
            ],
            "edges": [
                {"source": "Node1", "target": "Node2", "relation": "測試關係"}
            ]
        }
    
    def test_dict_to_networkx(self):
        """測試從 dict 轉換為 NetworkX"""
        nx_graph = GraphFormatAdapter._dict_to_networkx(self.dict_graph)
        
        self.assertIsInstance(nx_graph, nx.DiGraph)
        self.assertEqual(nx_graph.number_of_nodes(), 2)
        self.assertEqual(nx_graph.number_of_edges(), 1)
        self.assertIn("Node1", nx_graph.nodes())
        self.assertIn("Node2", nx_graph.nodes())
        self.assertTrue(nx_graph.has_edge("Node1", "Node2"))
    
    def test_to_networkx_from_dict(self):
        """測試 to_networkx 方法處理 dict 輸入"""
        nx_graph = GraphFormatAdapter.to_networkx(self.dict_graph, source_format="custom")
        
        self.assertIsInstance(nx_graph, nx.DiGraph)
        self.assertEqual(nx_graph.number_of_nodes(), 2)
        self.assertEqual(nx_graph.number_of_edges(), 1)
    
    def test_to_networkx_from_networkx(self):
        """測試 to_networkx 方法處理 NetworkX 輸入（應直接返回）"""
        result = GraphFormatAdapter.to_networkx(self.nx_graph, source_format="networkx")
        
        self.assertIs(result, self.nx_graph)  # 應該是同一個物件
    
    def test_to_networkx_with_graph_data_key(self):
        """測試 to_networkx 方法處理包含 graph_data 的 dict"""
        wrapper_dict = {
            "graph_data": self.nx_graph,
            "metadata": {"test": True}
        }
        
        result = GraphFormatAdapter.to_networkx(wrapper_dict, source_format="custom")
        
        self.assertIs(result, self.nx_graph)
    
    def test_graphml_save_and_load(self):
        """測試 GraphML 儲存與載入"""
        import tempfile
        import os
        
        # 建立臨時檔案
        with tempfile.TemporaryDirectory() as tmpdir:
            graphml_path = os.path.join(tmpdir, "test_graph.graphml")
            
            # 儲存
            GraphFormatAdapter.save_graphml(self.nx_graph, graphml_path)
            self.assertTrue(os.path.exists(graphml_path))
            
            # 載入
            loaded_graph = GraphFormatAdapter.load_graphml(graphml_path)
            
            # 驗證
            self.assertIsInstance(loaded_graph, nx.Graph)
            self.assertEqual(loaded_graph.number_of_nodes(), 2)
            self.assertEqual(loaded_graph.number_of_edges(), 1)
    
    def test_invalid_source_format(self):
        """測試無效的源格式"""
        # 使用真正無效的格式（不在支援列表中且不是 custom/autoschema）
        with self.assertRaises(ValueError):
            # 傳入非 dict 且非 NetworkX 的資料，並指定無效格式
            GraphFormatAdapter.to_networkx("invalid_data", source_format="totally_invalid_format")
    
    def test_invalid_target_format(self):
        """測試無效的目標格式"""
        with self.assertRaises(ValueError):
            GraphFormatAdapter.from_networkx(self.nx_graph, target_format="invalid_format")


class TestGraphFormatAdapterEdgeCases(unittest.TestCase):
    """測試邊界情況"""
    
    def test_empty_graph(self):
        """測試空圖譜"""
        empty_dict = {"nodes": [], "edges": []}
        nx_graph = GraphFormatAdapter._dict_to_networkx(empty_dict)
        
        self.assertEqual(nx_graph.number_of_nodes(), 0)
        self.assertEqual(nx_graph.number_of_edges(), 0)
    
    def test_nodes_only(self):
        """測試只有節點沒有邊"""
        nodes_only = {
            "nodes": [{"id": "A"}, {"id": "B"}],
            "edges": []
        }
        nx_graph = GraphFormatAdapter._dict_to_networkx(nodes_only)
        
        self.assertEqual(nx_graph.number_of_nodes(), 2)
        self.assertEqual(nx_graph.number_of_edges(), 0)
    
    def test_node_without_id(self):
        """測試節點缺少 id 的情況"""
        invalid_nodes = {
            "nodes": [{"name": "Node1"}, {"id": "Node2"}],
            "edges": []
        }
        nx_graph = GraphFormatAdapter._dict_to_networkx(invalid_nodes)
        
        # 應只添加有 id 或 name 的節點
        self.assertEqual(nx_graph.number_of_nodes(), 2)


if __name__ == "__main__":
    unittest.main()
