"""
Neo4j ⟷ NetworkX 轉換器

處理 Neo4j 與 NetworkX 之間的轉換
"""

import networkx as nx
from typing import Any, Dict, Optional


def neo4j_to_networkx(neo4j_session: Any) -> nx.Graph:
    """
    從 Neo4j 會話提取資料並轉換為 NetworkX graph
    
    Args:
        neo4j_session: Neo4j Driver session 或連接配置
    
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    try:
        # 如果是 dict 配置，建立連接
        if isinstance(neo4j_session, dict):
            try:
                from neo4j import GraphDatabase
            except ImportError:
                raise ImportError("需要安裝 neo4j driver: pip install neo4j")
            
            uri = neo4j_session.get("uri", "bolt://localhost:7687")
            username = neo4j_session.get("username", "neo4j")
            password = neo4j_session.get("password")
            database = neo4j_session.get("database", "neo4j")
            
            if not password:
                raise ValueError("需要提供 Neo4j 密碼")
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session(database=database) as session:
                # 查詢所有節點和關係
                result = session.run("""
                    MATCH (n)-[r]->(m)
                    RETURN n, r, m
                    LIMIT 10000
                """)
                
                for record in result:
                    node_n = record["n"]
                    relation = record["r"]
                    node_m = record["m"]
                    
                    # 添加節點
                    n_id = node_n.element_id if hasattr(node_n, 'element_id') else node_n.id
                    m_id = node_m.element_id if hasattr(node_m, 'element_id') else node_m.id
                    
                    if n_id not in G:
                        G.add_node(n_id, **dict(node_n))
                    if m_id not in G:
                        G.add_node(m_id, **dict(node_m))
                    
                    # 添加邊
                    G.add_edge(n_id, m_id, **dict(relation))
            
            driver.close()
        
        else:
            # 假設是已經建立的 session
            result = neo4j_session.run("""
                MATCH (n)-[r]->(m)
                RETURN n, r, m
                LIMIT 10000
            """)
            
            for record in result:
                node_n = record["n"]
                relation = record["r"]
                node_m = record["m"]
                
                n_id = node_n.element_id if hasattr(node_n, 'element_id') else node_n.id
                m_id = node_m.element_id if hasattr(node_m, 'element_id') else node_m.id
                
                if n_id not in G:
                    G.add_node(n_id, **dict(node_n))
                if m_id not in G:
                    G.add_node(m_id, **dict(node_m))
                
                G.add_edge(n_id, m_id, **dict(relation))
    
    except Exception as e:
        print(f"⚠️  從 Neo4j 提取資料時出錯: {e}")
        print("   返回空圖")
    
    print(f"✅ Neo4j → NetworkX: {G.number_of_nodes()} 節點, {G.number_of_edges()} 邊")
    return G


def networkx_to_neo4j(nx_graph: nx.Graph, neo4j_config: Dict) -> None:
    """
    將 NetworkX graph 匯入到 Neo4j
    
    Args:
        nx_graph: NetworkX graph
        neo4j_config: Neo4j 連接配置
            {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",
                "database": "neo4j"
            }
    
    Returns:
        None（資料直接寫入 Neo4j）
    """
    if not isinstance(nx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError(f"需要 NetworkX graph，收到: {type(nx_graph)}")
    
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("需要安裝 neo4j driver: pip install neo4j")
    
    uri = neo4j_config.get("uri", "bolt://localhost:7687")
    username = neo4j_config.get("username", "neo4j")
    password = neo4j_config.get("password")
    database = neo4j_config.get("database", "neo4j")
    
    if not password:
        raise ValueError("需要提供 Neo4j 密碼")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session(database=database) as session:
            # 清空資料庫（可選）
            if neo4j_config.get("clear_before_import", False):
                session.run("MATCH (n) DETACH DELETE n")
                print("🗑️  已清空 Neo4j 資料庫")
            
            # 批次建立節點
            node_batch = []
            for node_id in nx_graph.nodes():
                node_data = nx_graph.nodes[node_id]
                node_batch.append({
                    "id": str(node_id),
                    "properties": dict(node_data)
                })
                
                if len(node_batch) >= 1000:
                    session.run("""
                        UNWIND $nodes AS node
                        CREATE (n:Entity {id: node.id})
                        SET n += node.properties
                    """, nodes=node_batch)
                    node_batch = []
            
            if node_batch:
                session.run("""
                    UNWIND $nodes AS node
                    CREATE (n:Entity {id: node.id})
                    SET n += node.properties
                """, nodes=node_batch)
            
            # 批次建立關係
            edge_batch = []
            for source, target, edge_data in nx_graph.edges(data=True):
                edge_batch.append({
                    "source": str(source),
                    "target": str(target),
                    "relation": edge_data.get("relation", "RELATED_TO"),
                    "properties": dict(edge_data)
                })
                
                if len(edge_batch) >= 1000:
                    session.run("""
                        UNWIND $edges AS edge
                        MATCH (n:Entity {id: edge.source})
                        MATCH (m:Entity {id: edge.target})
                        CREATE (n)-[r:RELATION {type: edge.relation}]->(m)
                        SET r += edge.properties
                    """, edges=edge_batch)
                    edge_batch = []
            
            if edge_batch:
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (n:Entity {id: edge.source})
                    MATCH (m:Entity {id: edge.target})
                    CREATE (n)-[r:RELATION {type: edge.relation}]->(m)
                    SET r += edge.properties
                """, edges=edge_batch)
        
        print(f"✅ NetworkX → Neo4j: {nx_graph.number_of_nodes()} 節點, {nx_graph.number_of_edges()} 邊")
    
    except Exception as e:
        print(f"❌ 匯入 Neo4j 失敗: {e}")
        raise
    
    finally:
        driver.close()
