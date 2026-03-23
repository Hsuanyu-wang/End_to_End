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


def _sanitize_label(raw: str) -> str:
    """將原始類型字串轉為合法的 Neo4j label（英數底線，首字母大寫）"""
    import re
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(raw).strip())
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "Entity_" + cleaned
    return cleaned[:64]


def networkx_to_neo4j(nx_graph: nx.Graph, neo4j_config: Dict) -> None:
    """
    將 NetworkX graph 匯入到 Neo4j

    節點會依 entity_type / type 屬性動態設定 Neo4j label；
    關係會依 relation_type / type / label 屬性動態設定 relationship type。

    Args:
        nx_graph: NetworkX graph
        neo4j_config: Neo4j 連接配置
            {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",
                "database": "neo4j",
                "clear_before_import": false
            }
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
            if neo4j_config.get("clear_before_import", False):
                session.run("MATCH (n) DETACH DELETE n")
                print("🗑️  已清空 Neo4j 資料庫")

            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.id)")

            # 按 label 分組節點，批次寫入
            label_groups: Dict[str, list] = {}
            for node_id in nx_graph.nodes():
                data = dict(nx_graph.nodes[node_id])
                etype = data.get("entity_type") or data.get("type") or data.get("label") or ""
                neo_label = _sanitize_label(etype) if etype else "Entity"
                props = {k: v for k, v in data.items()
                         if isinstance(v, (str, int, float, bool))}
                label_groups.setdefault(neo_label, []).append({
                    "id": str(node_id),
                    "properties": props,
                })

            for label, batch in label_groups.items():
                for i in range(0, len(batch), 1000):
                    chunk = batch[i:i + 1000]
                    session.run(
                        f"UNWIND $nodes AS node "
                        f"MERGE (n:Entity {{id: node.id}}) "
                        f"SET n:{label} SET n += node.properties",
                        nodes=chunk,
                    )

            # 按 relation type 分組邊
            rel_groups: Dict[str, list] = {}
            for source, target, edge_data in nx_graph.edges(data=True):
                rtype = (edge_data.get("relation_type")
                         or edge_data.get("type")
                         or edge_data.get("label")
                         or "RELATED_TO")
                neo_rel = _sanitize_label(rtype)
                props = {k: v for k, v in edge_data.items()
                         if isinstance(v, (str, int, float, bool))}
                rel_groups.setdefault(neo_rel, []).append({
                    "source": str(source),
                    "target": str(target),
                    "properties": props,
                })

            for rtype, batch in rel_groups.items():
                for i in range(0, len(batch), 1000):
                    chunk = batch[i:i + 1000]
                    session.run(
                        f"UNWIND $edges AS edge "
                        f"MATCH (n:Entity {{id: edge.source}}) "
                        f"MATCH (m:Entity {{id: edge.target}}) "
                        f"CREATE (n)-[r:{rtype}]->(m) "
                        f"SET r += edge.properties",
                        edges=chunk,
                    )

        n_labels = len(label_groups)
        n_rtypes = len(rel_groups)
        print(f"✅ NetworkX → Neo4j: {nx_graph.number_of_nodes()} 節點 ({n_labels} 種 label), "
              f"{nx_graph.number_of_edges()} 邊 ({n_rtypes} 種 relationship type)")

    except Exception as e:
        print(f"❌ 匯入 Neo4j 失敗: {e}")
        raise

    finally:
        driver.close()
