"""
GraphML 處理器

處理 NetworkX 與 GraphML 檔案之間的轉換
"""

import networkx as nx
import os


def save_graphml(nx_graph: nx.Graph, path: str):
    """
    儲存 NetworkX graph 為 GraphML 檔案
    
    Args:
        nx_graph: NetworkX graph
        path: 儲存路徑（.graphml）
    """
    if not isinstance(nx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError(f"需要 NetworkX graph，收到: {type(nx_graph)}")
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # 儲存為 GraphML
    nx.write_graphml(nx_graph, path)
    print(f"✅ GraphML 已儲存: {path}")
    print(f"   節點數: {nx_graph.number_of_nodes()}, 邊數: {nx_graph.number_of_edges()}")


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
    
    # 載入 GraphML
    nx_graph = nx.read_graphml(path)
    print(f"✅ GraphML 已載入: {path}")
    print(f"   節點數: {nx_graph.number_of_nodes()}, 邊數: {nx_graph.number_of_edges()}")
    
    return nx_graph
