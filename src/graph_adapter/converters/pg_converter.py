"""
PropertyGraph ⟷ NetworkX 轉換器

處理 LlamaIndex PropertyGraphIndex 與 NetworkX 之間的轉換
"""

import networkx as nx
from typing import Any, Optional


def pg_to_networkx(pg_index: Any) -> nx.Graph:
    """
    將 PropertyGraphIndex 轉換為 NetworkX graph
    
    Args:
        pg_index: PropertyGraphIndex 實例
    
    Returns:
        NetworkX DiGraph
    """
    try:
        from llama_index.core import PropertyGraphIndex
    except ImportError:
        raise ImportError("需要安裝 llama-index-core: pip install llama-index-core")
    
    if not isinstance(pg_index, PropertyGraphIndex):
        raise ValueError(f"需要 PropertyGraphIndex，收到: {type(pg_index)}")
    
    G = nx.DiGraph()
    
    # 從 PropertyGraphIndex 提取圖譜資料
    if not hasattr(pg_index, 'property_graph_store'):
        print("⚠️  PropertyGraphIndex 沒有 property_graph_store 屬性")
        return G
    
    graph_store = pg_index.property_graph_store
    
    try:
        # 嘗試取得所有三元組（triplets）
        # PropertyGraphStore 的介面可能因版本而異
        if hasattr(graph_store, 'get_triplets'):
            triplets = graph_store.get_triplets()
            
            for triplet in triplets:
                # 三元組格式: (subject, predicate, object)
                if hasattr(triplet, 'subject') and hasattr(triplet, 'object_'):
                    subject = triplet.subject
                    predicate = triplet.predicate if hasattr(triplet, 'predicate') else "RELATED_TO"
                    obj = triplet.object_
                    
                    # 添加節點（如果不存在）
                    if subject.name not in G:
                        G.add_node(subject.name, **{"type": subject.label if hasattr(subject, 'label') else "Entity"})
                    if obj.name not in G:
                        G.add_node(obj.name, **{"type": obj.label if hasattr(obj, 'label') else "Entity"})
                    
                    # 添加邊
                    G.add_edge(subject.name, obj.name, relation=predicate)
        
        # 如果沒有 get_triplets，嘗試其他方法
        elif hasattr(graph_store, 'get'):
            # 嘗試取得所有節點和邊
            pass
    
    except Exception as e:
        print(f"⚠️  從 PropertyGraphIndex 提取資料時出錯: {e}")
        print("   返回空圖")
    
    print(f"✅ PropertyGraph → NetworkX: {G.number_of_nodes()} 節點, {G.number_of_edges()} 邊")
    return G


def networkx_to_pg(nx_graph: nx.Graph, settings: Any) -> Any:
    """
    將 NetworkX graph 轉換為 PropertyGraphIndex
    
    注意：這是一個複雜的轉換，因為需要重建 PropertyGraphIndex 的內部結構
    
    Args:
        nx_graph: NetworkX graph
        settings: LlamaIndex Settings（需要 llm 和 embed_model）
    
    Returns:
        PropertyGraphIndex 實例
    """
    try:
        from llama_index.core import PropertyGraphIndex, Document
        from llama_index.core.graph_stores import SimplePropertyGraphStore
    except ImportError:
        raise ImportError("需要安裝 llama-index-core: pip install llama-index-core")
    
    if not isinstance(nx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError(f"需要 NetworkX graph，收到: {type(nx_graph)}")
    
    # 建立 PropertyGraphStore
    property_graph_store = SimplePropertyGraphStore()
    
    # 將 NetworkX 資料轉換為 PropertyGraph 格式
    # 注意：這是簡化實作，完整版需要處理更多細節
    
    try:
        from llama_index.core.graph_stores import EntityNode, Relation
        
        # 添加節點
        for node_id in nx_graph.nodes():
            node_data = nx_graph.nodes[node_id]
            entity_node = EntityNode(
                name=str(node_id),
                label=node_data.get("type", "Entity"),
                properties=dict(node_data)
            )
            property_graph_store.upsert_nodes([entity_node])
        
        # 添加邊
        for source, target, edge_data in nx_graph.edges(data=True):
            relation = Relation(
                source_id=str(source),
                target_id=str(target),
                label=edge_data.get("relation", "RELATED_TO"),
                properties=dict(edge_data)
            )
            property_graph_store.upsert_relations([relation])
    
    except Exception as e:
        print(f"⚠️  轉換 NetworkX 到 PropertyGraph 時出錯: {e}")
    
    # 建立 PropertyGraphIndex
    # 注意：這裡建立的是空索引，實際應用中可能需要重新索引文檔
    try:
        pg_index = PropertyGraphIndex(
            nodes=[],  # 空文檔列表
            property_graph_store=property_graph_store,
            llm=settings.llm if hasattr(settings, 'llm') else None,
            embed_model=settings.embed_model if hasattr(settings, 'embed_model') else None,
        )
        print(f"✅ NetworkX → PropertyGraph: {nx_graph.number_of_nodes()} 節點, {nx_graph.number_of_edges()} 邊")
        return pg_index
    
    except Exception as e:
        print(f"❌ 建立 PropertyGraphIndex 失敗: {e}")
        raise
