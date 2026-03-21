"""
GraphFormatAdapter 基類

提供圖譜格式轉換的統一介面，使用 NetworkX/GraphML 作為中轉格式
"""

import networkx as nx
from typing import Any, Dict, Optional
import os


class GraphFormatAdapter:
    """
    統一圖譜格式轉換器
    
    設計理念：使用 NetworkX 作為 Hub-and-Spoke 中心，簡化格式轉換
    - 任意格式 → NetworkX → 目標格式
    - 避免 N×(N-1) 的直接轉換組合爆炸
    """
    
    # 儲存已註冊的轉換器
    _converters: Dict[str, callable] = {}
    _reverse_converters: Dict[str, callable] = {}
    
    @classmethod
    def register_converter(cls, format_name: str):
        """
        裝飾器：註冊格式轉換器（格式 → NetworkX）
        
        用法:
        @GraphFormatAdapter.register_converter("my_format")
        def my_format_to_networkx(source):
            return nx_graph
        """
        def decorator(func):
            cls._converters[format_name] = func
            return func
        return decorator
    
    @classmethod
    def register_reverse_converter(cls, format_name: str):
        """
        裝飾器：註冊反向轉換器（NetworkX → 格式）
        
        用法:
        @GraphFormatAdapter.register_reverse_converter("my_format")
        def networkx_to_my_format(nx_graph, settings):
            return my_format_instance
        """
        def decorator(func):
            cls._reverse_converters[format_name] = func
            return func
        return decorator
    
    # === 核心轉換介面：任意格式 ⟷ NetworkX ===
    
    @staticmethod
    def to_networkx(source: Any, source_format: str) -> nx.Graph:
        """
        將任意格式轉為 NetworkX
        
        Args:
            source: 源圖譜資料（可以是 dict、PropertyGraphIndex、LightRAG instance 等）
            source_format: 源格式名稱（property_graph、lightrag、autoschema、networkx 等）
        
        Returns:
            NetworkX Graph
        """
        # 如果已經是 NetworkX，直接返回
        if isinstance(source, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            return source
        
        # 如果是 dict，嘗試提取 graph_data
        if isinstance(source, dict):
            if "graph_data" in source and isinstance(source["graph_data"], (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                return source["graph_data"]
            # PropertyGraph dict: 優先取 graph_index 用專用轉換器
            if "graph_index" in source and source.get("graph_format") == "property_graph":
                from .converters.pg_converter import pg_to_networkx
                return pg_to_networkx(source["graph_index"])
            # 如果有實際的 nodes 和 edges 資料，從字典建立
            if "nodes" in source and "edges" in source and (source["nodes"] or source["edges"]):
                return GraphFormatAdapter._dict_to_networkx(source)
        
        # 使用已註冊的轉換器
        if source_format in GraphFormatAdapter._converters:
            converter = GraphFormatAdapter._converters[source_format]
            return converter(source)
        
        # 根據格式名稱選擇轉換方法
        if source_format == "property_graph":
            from .converters.pg_converter import pg_to_networkx
            # 如果是 dict 且有 graph_index，先取出
            if isinstance(source, dict) and "graph_index" in source:
                return pg_to_networkx(source["graph_index"])
            return pg_to_networkx(source)
        elif source_format == "lightrag":
            from .converters.lightrag_converter import lightrag_to_networkx
            return lightrag_to_networkx(source)
        elif source_format in ("autoschema", "custom"):
            # AutoSchemaKG 和自定義格式通常輸出 dict 格式
            if isinstance(source, dict):
                return GraphFormatAdapter._dict_to_networkx(source)
            else:
                raise ValueError(f"AutoSchema/Custom 格式需要 dict 輸入，收到: {type(source)}")
        else:
            raise ValueError(f"未知的源格式: {source_format}。可用: {list(GraphFormatAdapter._converters.keys())}")
    
    @staticmethod
    def from_networkx(nx_graph: nx.Graph, target_format: str, settings: Any = None) -> Any:
        """
        將 NetworkX 轉為目標格式
        
        Args:
            nx_graph: NetworkX graph
            target_format: 目標格式名稱
            settings: 配置設定（某些格式需要）
        
        Returns:
            目標格式的圖譜實例
        """
        if not isinstance(nx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise ValueError(f"需要 NetworkX graph，收到: {type(nx_graph)}")
        
        # 使用已註冊的反向轉換器
        if target_format in GraphFormatAdapter._reverse_converters:
            converter = GraphFormatAdapter._reverse_converters[target_format]
            return converter(nx_graph, settings)
        
        # 根據格式名稱選擇轉換方法
        if target_format == "property_graph":
            from .converters.pg_converter import networkx_to_pg
            return networkx_to_pg(nx_graph, settings)
        elif target_format == "lightrag":
            from .converters.lightrag_converter import networkx_to_lightrag
            return networkx_to_lightrag(nx_graph, settings)
        elif target_format == "neo4j":
            from .converters.neo4j_converter import networkx_to_neo4j
            return networkx_to_neo4j(nx_graph, settings)
        else:
            raise ValueError(f"未知的目標格式: {target_format}。可用: {list(GraphFormatAdapter._reverse_converters.keys())}")
    
    # === 便捷方法：特定格式 ⟷ NetworkX ===
    
    @staticmethod
    def pg_to_networkx(pg_index) -> nx.Graph:
        """PropertyGraph → NetworkX"""
        from .converters.pg_converter import pg_to_networkx
        return pg_to_networkx(pg_index)
    
    @staticmethod
    def networkx_to_pg(nx_graph: nx.Graph, settings) -> Any:
        """NetworkX → PropertyGraph"""
        from .converters.pg_converter import networkx_to_pg
        return networkx_to_pg(nx_graph, settings)
    
    @staticmethod
    def lightrag_to_networkx(lightrag_instance) -> nx.Graph:
        """LightRAG → NetworkX"""
        from .converters.lightrag_converter import lightrag_to_networkx
        return lightrag_to_networkx(lightrag_instance)
    
    @staticmethod
    def networkx_to_lightrag(nx_graph: nx.Graph, settings) -> Any:
        """NetworkX → LightRAG"""
        from .converters.lightrag_converter import networkx_to_lightrag
        return networkx_to_lightrag(nx_graph, settings)
    
    @staticmethod
    def neo4j_to_networkx(neo4j_session) -> nx.Graph:
        """Neo4j → NetworkX"""
        from .converters.neo4j_converter import neo4j_to_networkx
        return neo4j_to_networkx(neo4j_session)
    
    @staticmethod
    def networkx_to_neo4j(nx_graph: nx.Graph, neo4j_config: dict) -> Any:
        """NetworkX → Neo4j"""
        from .converters.neo4j_converter import networkx_to_neo4j
        return networkx_to_neo4j(nx_graph, neo4j_config)
    
    # === GraphML 持久化 ===
    
    @staticmethod
    def save_graphml(nx_graph: nx.Graph, path: str):
        """
        儲存 NetworkX graph 為 GraphML 檔案
        
        Args:
            nx_graph: NetworkX graph
            path: 儲存路徑（.graphml）
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        nx.write_graphml(nx_graph, path)
        print(f"✅ GraphML 已儲存: {path}")
    
    @staticmethod
    def load_graphml(path: str) -> nx.Graph:
        """
        從 GraphML 檔案載入 NetworkX graph
        
        Args:
            path: GraphML 檔案路徑
        
        Returns:
            NetworkX graph
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"GraphML 檔案不存在: {path}")
        
        nx_graph = nx.read_graphml(path)
        print(f"✅ GraphML 已載入: {path}")
        return nx_graph
    
    # === 輔助方法 ===
    
    @staticmethod
    def _dict_to_networkx(graph_dict: Dict) -> nx.Graph:
        """
        從字典格式建立 NetworkX graph
        
        Args:
            graph_dict: 包含 nodes 和 edges 的字典
        
        Returns:
            NetworkX graph
        """
        G = nx.DiGraph()
        
        # 添加節點
        nodes = graph_dict.get("nodes", [])
        for node in nodes:
            if isinstance(node, dict):
                node_id = node.get("id") or node.get("name")
                if node_id:
                    # 將其他屬性作為節點屬性
                    attrs = {k: v for k, v in node.items() if k not in ("id", "name")}
                    G.add_node(node_id, **attrs)
        
        # 添加邊
        edges = graph_dict.get("edges", [])
        for edge in edges:
            if isinstance(edge, dict):
                source = edge.get("source")
                target = edge.get("target")
                if source and target:
                    # 將其他屬性作為邊屬性
                    attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
                    G.add_edge(source, target, **attrs)
        
        return G
