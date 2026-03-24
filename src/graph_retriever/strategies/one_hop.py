"""
1-Hop Strategy

包裝 LightRAG 原始的 1-hop edge expansion 行為作為 baseline。
對每個 seed entity 取其直接鄰居（1-hop），按 (degree, weight) 排序。
"""

import logging
from typing import Any, Dict, List

import networkx as nx

from src.graph_retriever.strategies.base import (
    BaseTraversalStrategy,
    TraversalResult,
    register_strategy,
)

logger = logging.getLogger(__name__)


@register_strategy("one_hop")
class OneHopStrategy(BaseTraversalStrategy):

    def get_name(self) -> str:
        return "OneHop"

    def traverse(
        self,
        nx_graph: nx.DiGraph,
        seed_entities: List[Dict[str, Any]],
        query: str,
        top_k: int = 20,
        **kwargs,
    ) -> TraversalResult:
        seed_names = {
            e["entity_name"] for e in seed_entities if e["entity_name"] in nx_graph
        }
        if not seed_names:
            return TraversalResult(metadata={"strategy": "one_hop", "seeds_found": 0})

        collected_nodes: Dict[str, Dict[str, Any]] = {}
        collected_edges: List[Dict[str, Any]] = []
        seen_edges: set = set()

        for name in seed_names:
            node_data = dict(nx_graph.nodes[name])
            node_data["entity_name"] = name
            node_data["rank"] = nx_graph.degree(name)
            collected_nodes[name] = node_data

            for u, v, data in nx_graph.edges(name, data=True):
                edge_key = (min(u, v), max(u, v))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    collected_edges.append({
                        "src_tgt": (u, v),
                        "weight": float(data.get("weight", 1.0)),
                        **{k: v_ for k, v_ in data.items() if k != "weight"},
                        "rank": nx_graph.degree(u) + nx_graph.degree(v),
                    })

                neighbor = v if u == name else u
                if neighbor not in collected_nodes and neighbor in nx_graph:
                    nb_data = dict(nx_graph.nodes[neighbor])
                    nb_data["entity_name"] = neighbor
                    nb_data["rank"] = nx_graph.degree(neighbor)
                    collected_nodes[neighbor] = nb_data

            if nx_graph.is_directed():
                for u, v, data in nx_graph.in_edges(name, data=True):
                    edge_key = (min(u, v), max(u, v))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        collected_edges.append({
                            "src_tgt": (u, v),
                            "weight": float(data.get("weight", 1.0)),
                            **{k: v_ for k, v_ in data.items() if k != "weight"},
                            "rank": nx_graph.degree(u) + nx_graph.degree(v),
                        })
                    if u not in collected_nodes and u in nx_graph:
                        nb_data = dict(nx_graph.nodes[u])
                        nb_data["entity_name"] = u
                        nb_data["rank"] = nx_graph.degree(u)
                        collected_nodes[u] = nb_data

        collected_edges.sort(key=lambda x: (x["rank"], x["weight"]), reverse=True)

        nodes_list = list(collected_nodes.values())[:top_k]

        return TraversalResult(
            nodes=nodes_list,
            edges=collected_edges,
            metadata={
                "strategy": "one_hop",
                "seeds_found": len(seed_names),
                "total_nodes": len(nodes_list),
                "total_edges": len(collected_edges),
            },
        )
