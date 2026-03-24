"""
K-Hop Strategy

從 seed entities 進行分層擴展，最多擴展到 k-hop。
節點排序維持 degree 導向，邊排序維持 (rank, weight)。
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


@register_strategy("k_hop")
class KHopStrategy(BaseTraversalStrategy):

    def __init__(self, hop_k: int = 1):
        self.hop_k = max(1, int(hop_k))

    def get_name(self) -> str:
        return "KHop"

    def traverse(
        self,
        nx_graph: nx.DiGraph,
        seed_entities: List[Dict[str, Any]],
        query: str,
        top_k: int = 20,
        **kwargs,
    ) -> TraversalResult:
        hop_k = max(1, int(kwargs.get("hop_k", self.hop_k)))
        seed_names = {
            e["entity_name"] for e in seed_entities if e["entity_name"] in nx_graph
        }
        if not seed_names:
            return TraversalResult(
                metadata={"strategy": "k_hop", "seeds_found": 0, "hop_k": hop_k}
            )

        collected_nodes: Dict[str, Dict[str, Any]] = {}
        collected_edges: List[Dict[str, Any]] = []
        seen_edges: set = set()
        visited: set = set()
        current_frontier = set(seed_names)
        layer_stats: List[Dict[str, int]] = []

        for name in seed_names:
            node_data = dict(nx_graph.nodes[name])
            node_data["entity_name"] = name
            node_data["rank"] = nx_graph.degree(name)
            node_data["hop"] = 0
            collected_nodes[name] = node_data
            visited.add(name)

        for hop in range(1, hop_k + 1):
            next_frontier = set()
            layer_nodes_before = len(collected_nodes)
            layer_edges_before = len(collected_edges)
            for name in current_frontier:
                if name not in nx_graph:
                    continue

                for u, v, data in nx_graph.edges(name, data=True):
                    edge_key = (min(u, v), max(u, v))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        collected_edges.append({
                            "src_tgt": (u, v),
                            "weight": float(data.get("weight", 1.0)),
                            **{k: v_ for k, v_ in data.items() if k != "weight"},
                            "rank": nx_graph.degree(u) + nx_graph.degree(v),
                            "hop": hop,
                        })

                    neighbor = v if u == name else u
                    if neighbor not in visited and neighbor in nx_graph:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        nb_data = dict(nx_graph.nodes[neighbor])
                        nb_data["entity_name"] = neighbor
                        nb_data["rank"] = nx_graph.degree(neighbor)
                        nb_data["hop"] = hop
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
                                "hop": hop,
                            })
                        if u not in visited and u in nx_graph:
                            visited.add(u)
                            next_frontier.add(u)
                            nb_data = dict(nx_graph.nodes[u])
                            nb_data["entity_name"] = u
                            nb_data["rank"] = nx_graph.degree(u)
                            nb_data["hop"] = hop
                            collected_nodes[u] = nb_data

            layer_stats.append({
                "hop": hop,
                "new_nodes": len(collected_nodes) - layer_nodes_before,
                "new_edges": len(collected_edges) - layer_edges_before,
            })
            if not next_frontier:
                break
            current_frontier = next_frontier

        collected_edges.sort(key=lambda x: (x["rank"], x["weight"]), reverse=True)

        nodes_list = sorted(
            collected_nodes.values(),
            key=lambda x: (x.get("rank", 0), -x.get("hop", 0)),
            reverse=True,
        )[:top_k]

        return TraversalResult(
            nodes=nodes_list,
            edges=collected_edges,
            metadata={
                "strategy": "k_hop",
                "seeds_found": len(seed_names),
                "hop_k": hop_k,
                "layer_stats": layer_stats,
                "total_nodes": len(nodes_list),
                "total_edges": len(collected_edges),
            },
        )
