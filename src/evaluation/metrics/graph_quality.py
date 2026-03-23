"""
圖譜建構品質評估指標

從 GraphML 檔案或 build() 回傳結果計算圖譜結構與實體品質指標。
"""

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import networkx as nx


class GraphQualityMetrics:
    """
    圖譜品質指標計算器

    支援兩種輸入來源：
    - compute_from_graphml(path): 離線分析已存在的 GraphML
    - compute_from_networkx(G): 直接從 NetworkX graph 計算
    """

    @staticmethod
    def compute_from_graphml(path: str) -> Dict[str, Any]:
        """從 GraphML 檔案計算品質指標"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"GraphML 檔案不存在: {path}")
        G = nx.read_graphml(path)
        return GraphQualityMetrics.compute_from_networkx(G)

    @staticmethod
    def compute_from_build(build_result: Dict[str, Any], source_format: str = "auto") -> Dict[str, Any]:
        """從 build() 回傳值計算品質指標（自動轉 NetworkX）"""
        from src.graph_adapter.base_adapter import GraphFormatAdapter

        graphml_path = build_result.get("graphml_path")
        if graphml_path and os.path.exists(graphml_path):
            return GraphQualityMetrics.compute_from_graphml(graphml_path)

        try:
            G = GraphFormatAdapter.to_networkx(build_result, source_format)
            return GraphQualityMetrics.compute_from_networkx(G)
        except Exception as e:
            return {"error": str(e), "node_count": 0, "edge_count": 0}

    @staticmethod
    def compute_from_networkx(G: nx.Graph) -> Dict[str, Any]:
        """核心計算：從 NetworkX graph 產出所有品質指標"""
        is_directed = G.is_directed()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "max_degree": 0,
                "num_connected_components": 0,
                "largest_component_ratio": 0.0,
                "orphan_node_count": 0,
                "orphan_node_ratio": 0.0,
                "avg_clustering_coefficient": 0.0,
                "entity_type_distribution": {},
                "relation_type_distribution": {},
                "is_directed": is_directed,
            }

        density = nx.density(G)
        degrees = [d for _, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0

        UG = G.to_undirected() if is_directed else G
        components = list(nx.connected_components(UG))
        num_components = len(components)
        largest_ratio = max(len(c) for c in components) / n_nodes if components else 0.0

        orphan_count = sum(1 for d in degrees if d == 0)
        orphan_ratio = orphan_count / n_nodes if n_nodes else 0.0

        try:
            avg_cc = nx.average_clustering(UG)
        except Exception:
            avg_cc = 0.0

        entity_types: Counter = Counter()
        for _, data in G.nodes(data=True):
            etype = data.get("entity_type") or data.get("type") or data.get("label") or "UNKNOWN"
            entity_types[str(etype)] += 1

        relation_types: Counter = Counter()
        for _, _, data in G.edges(data=True):
            rtype = data.get("relation_type") or data.get("type") or data.get("label") or "UNKNOWN"
            relation_types[str(rtype)] += 1

        return {
            "node_count": n_nodes,
            "edge_count": n_edges,
            "density": round(density, 6),
            "avg_degree": round(avg_degree, 4),
            "max_degree": max_degree,
            "num_connected_components": num_components,
            "largest_component_ratio": round(largest_ratio, 4),
            "orphan_node_count": orphan_count,
            "orphan_node_ratio": round(orphan_ratio, 4),
            "avg_clustering_coefficient": round(avg_cc, 4),
            "entity_type_distribution": dict(entity_types.most_common()),
            "relation_type_distribution": dict(relation_types.most_common()),
            "is_directed": is_directed,
        }

    @staticmethod
    def save_report(metrics: Dict[str, Any], path: str) -> None:
        """將品質指標儲存為 JSON"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"✅ 圖譜品質報告已儲存: {path}")

    @staticmethod
    def summary_columns() -> List[str]:
        """回傳可加入 global_summary 的數值欄位名稱"""
        return [
            "node_count",
            "edge_count",
            "density",
            "avg_degree",
            "orphan_node_ratio",
            "largest_component_ratio",
            "num_connected_components",
            "avg_clustering_coefficient",
        ]
