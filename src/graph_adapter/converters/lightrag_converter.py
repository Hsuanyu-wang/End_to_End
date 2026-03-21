"""
LightRAG ⟷ NetworkX 轉換器

處理 LightRAG 與 NetworkX 之間的轉換
"""

import glob
import networkx as nx
from typing import Any, List, Optional
import os


def _resolve_lightrag_graphml_path(working_dir: str) -> Optional[str]:
    """
    解析 LightRAG working_dir 內的 GraphML 路徑。
    現行 LightRAG 預設檔名為 graph_chunk_entity_relation.graphml，舊版或自訂可能為 graph.graphml。
    """
    candidates: List[str] = [
        os.path.join(working_dir, "graph_chunk_entity_relation.graphml"),
        os.path.join(working_dir, "graph.graphml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    all_ml = sorted(glob.glob(os.path.join(working_dir, "*.graphml")))
    if not all_ml:
        return None
    if len(all_ml) == 1:
        return all_ml[0]

    basenames = {os.path.basename(p) for p in all_ml}
    for preferred in ("graph_chunk_entity_relation.graphml", "graph.graphml"):
        if preferred in basenames:
            return os.path.join(working_dir, preferred)
    return all_ml[0]


def lightrag_to_networkx(lightrag_source: Any) -> nx.Graph:
    """
    將 LightRAG 實例或儲存目錄轉換為 NetworkX graph
    
    Args:
        lightrag_source: LightRAG 實例或儲存目錄路徑
    
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    try:
        # 如果是字串，假設是儲存目錄
        if isinstance(lightrag_source, str):
            working_dir = lightrag_source
        # 如果是 LightRAG 實例
        elif hasattr(lightrag_source, 'working_dir'):
            working_dir = lightrag_source.working_dir
        else:
            # 如果是 dict，可能包含 storage_path
            if isinstance(lightrag_source, dict):
                working_dir = lightrag_source.get("storage_path") or lightrag_source.get("working_dir")
            else:
                raise ValueError(f"無法從 {type(lightrag_source)} 提取 working_dir")
        
        if not working_dir or not os.path.exists(working_dir):
            print(f"⚠️  LightRAG 儲存目錄不存在: {working_dir}")
            return G
        
        # LightRAG 將圖譜資料儲存在 working_dir 中（檔名依版本可能不同）
        graphml_path = _resolve_lightrag_graphml_path(working_dir)

        if graphml_path and os.path.isfile(graphml_path):
            G = nx.read_graphml(graphml_path)
            print(f"✅ LightRAG (GraphML) → NetworkX: {G.number_of_nodes()} 節點, {G.number_of_edges()} 邊")
            print(f"   來源: {graphml_path}")
            return G

        print("⚠️  未找到 LightRAG GraphML 檔案，返回空圖")
        print(f"   已搜尋目錄: {working_dir}")
    
    except Exception as e:
        print(f"⚠️  從 LightRAG 提取資料時出錯: {e}")
        print("   返回空圖")
    
    return G


def networkx_to_lightrag(nx_graph: nx.Graph, settings: Any) -> Any:
    """
    將 NetworkX graph 轉換為 LightRAG 實例
    
    注意：這需要重新建立 LightRAG 實例並插入資料
    
    Args:
        nx_graph: NetworkX graph
        settings: 配置設定（需要 LightRAG 相關配置）
    
    Returns:
        LightRAG 實例
    """
    if not isinstance(nx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError(f"需要 NetworkX graph，收到: {type(nx_graph)}")
    
    try:
        from lightrag import LightRAG
        from lightrag.llm import openai_complete_if_cache, openai_embedding
    except ImportError:
        raise ImportError("需要安裝 LightRAG: pip install lightrag-hku")
    
    # 取得配置
    working_dir = getattr(settings, 'lightrag_working_dir', None) or "/tmp/lightrag_from_networkx"
    os.makedirs(working_dir, exist_ok=True)
    
    # 建立 LightRAG 實例
    try:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=openai_complete_if_cache if hasattr(settings, 'llm') else None,
            embedding_func=openai_embedding if hasattr(settings, 'embed_model') else None,
        )
        
        # 將 NetworkX 資料寫入 LightRAG
        # 注意：這是簡化實作，完整版需要將圖譜資料轉換為 LightRAG 格式
        
        # 將 NetworkX 儲存為 GraphML，LightRAG 可能可以讀取
        graphml_path = os.path.join(working_dir, "graph.graphml")
        nx.write_graphml(nx_graph, graphml_path)
        
        print(f"✅ NetworkX → LightRAG: {nx_graph.number_of_nodes()} 節點, {nx_graph.number_of_edges()} 邊")
        print(f"   working_dir: {working_dir}")
        
        return rag
    
    except Exception as e:
        print(f"❌ 建立 LightRAG 實例失敗: {e}")
        raise
