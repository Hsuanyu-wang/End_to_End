"""
Prize-Collecting Steiner Tree (PCST) Strategy

從 seed entities 出發，將 entity linking 分數作為 prize，
edge weight 的倒數作為 cost，求解 PCST 得到連接 seed nodes 的最小成本子圖。

優先使用 pcst_fast 套件；fallback 到 NetworkX Steiner tree 近似。
"""

import logging
from typing import Any, Dict, List, Literal

import networkx as nx
import numpy as np

from src.graph_retriever.strategies.base import (
    BaseTraversalStrategy,
    TraversalResult,
    register_strategy,
)

logger = logging.getLogger(__name__)

CostMode = Literal["inverse_weight", "uniform", "inverse_log_weight"]

try:
    import pcst_fast
    HAS_PCST_FAST = True
except ImportError:
    HAS_PCST_FAST = False
    logger.info("pcst_fast 未安裝，PCST 將使用 NetworkX Steiner tree fallback")


@register_strategy("pcst")
class PCSTStrategy(BaseTraversalStrategy):

    def __init__(
        self,
        prize_scale: float = 1.0,
        cost_mode: CostMode = "inverse_weight",
        num_clusters: int = 1,
        pruning: str = "gw",
        min_nodes: int = 5,
    ):
        """
        Args:
            prize_scale: seed entity prize 的縮放因子
            cost_mode: edge cost 計算方式
            num_clusters: PCST 回傳的連通分量數（預設 1 = 單一 Steiner tree）
            pruning: pcst_fast pruning method ("none", "simple", "gw", "strong")
            min_nodes: 結果最少節點數（不足時用 1-hop 補充）
        """
        self.prize_scale = prize_scale
        self.cost_mode = cost_mode
        self.num_clusters = num_clusters
        self.pruning = pruning
        self.min_nodes = min_nodes

    def get_name(self) -> str:
        return f"PCST(cost={self.cost_mode})"

    def traverse(
        self,
        nx_graph: nx.DiGraph,
        seed_entities: List[Dict[str, Any]],
        query: str,
        top_k: int = 20,
        **kwargs,
    ) -> TraversalResult:
        valid_seeds = [
            e for e in seed_entities if e["entity_name"] in nx_graph
        ]
        if not valid_seeds:
            return TraversalResult(metadata={"strategy": "pcst", "seeds_found": 0})

        undirected = nx_graph.to_undirected() if nx_graph.is_directed() else nx_graph

        if HAS_PCST_FAST:
            selected_nodes, selected_edges = self._solve_pcst_fast(
                undirected, valid_seeds, top_k
            )
        else:
            selected_nodes, selected_edges = self._solve_steiner_fallback(
                undirected, valid_seeds, top_k
            )

        # 若節點太少，用 1-hop 補充
        if len(selected_nodes) < self.min_nodes:
            selected_nodes, selected_edges = self._augment_with_one_hop(
                nx_graph, selected_nodes, selected_edges, valid_seeds, top_k
            )

        nodes_list = []
        for name in selected_nodes:
            if name not in nx_graph:
                continue
            nd = dict(nx_graph.nodes[name])
            nd["entity_name"] = name
            nd["rank"] = nx_graph.degree(name)
            nodes_list.append(nd)

        edges_list = []
        for u, v in selected_edges:
            if nx_graph.has_edge(u, v):
                data = dict(nx_graph.edges[u, v])
            elif nx_graph.has_edge(v, u):
                data = dict(nx_graph.edges[v, u])
            else:
                data = {}
            edges_list.append({
                "src_tgt": (u, v),
                "weight": float(data.get("weight", 1.0)),
                "description": data.get("description", ""),
                "keywords": data.get("keywords", ""),
            })

        return TraversalResult(
            nodes=nodes_list,
            edges=edges_list,
            metadata={
                "strategy": "pcst",
                "cost_mode": self.cost_mode,
                "seeds_found": len(valid_seeds),
                "total_nodes": len(nodes_list),
                "total_edges": len(edges_list),
                "solver": "pcst_fast" if HAS_PCST_FAST else "steiner_fallback",
            },
        )

    # ------------------------------------------------------------------
    # pcst_fast solver
    # ------------------------------------------------------------------

    def _solve_pcst_fast(
        self,
        G: nx.Graph,
        seeds: List[Dict[str, Any]],
        top_k: int,
    ) -> tuple:
        node_list = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        n = len(node_list)

        prizes = np.zeros(n, dtype=np.float64)
        seed_set = set()
        for s in seeds:
            name = s["entity_name"]
            if name in node_to_idx:
                score = float(s.get("vdb_score", 1.0))
                prizes[node_to_idx[name]] = max(score, 0.01) * self.prize_scale
                seed_set.add(name)

        edge_list = list(G.edges())
        m = len(edge_list)
        edges_arr = np.zeros((m, 2), dtype=np.int64)
        costs_arr = np.zeros(m, dtype=np.float64)

        for idx, (u, v) in enumerate(edge_list):
            edges_arr[idx, 0] = node_to_idx[u]
            edges_arr[idx, 1] = node_to_idx[v]
            costs_arr[idx] = self._compute_cost(G, u, v)

        root = -1
        if seeds:
            first_seed = seeds[0]["entity_name"]
            if first_seed in node_to_idx:
                root = node_to_idx[first_seed]

        selected_v, selected_e = pcst_fast.pcst_fast(
            edges_arr, prizes, costs_arr,
            root, self.num_clusters, self.pruning, 0
        )

        selected_nodes = {node_list[i] for i in selected_v}
        selected_edges = []
        for e_idx in selected_e:
            u, v = edge_list[e_idx]
            selected_edges.append((u, v))

        return selected_nodes, selected_edges

    # ------------------------------------------------------------------
    # NetworkX Steiner tree fallback
    # ------------------------------------------------------------------

    def _solve_steiner_fallback(
        self,
        G: nx.Graph,
        seeds: List[Dict[str, Any]],
        top_k: int,
    ) -> tuple:
        terminal_nodes = [
            s["entity_name"] for s in seeds if s["entity_name"] in G
        ]

        if len(terminal_nodes) < 2:
            selected_nodes = set(terminal_nodes)
            return selected_nodes, []

        # 確保在同一個連通分量
        largest_cc = max(nx.connected_components(G), key=len)
        terminals_in_cc = [t for t in terminal_nodes if t in largest_cc]

        if len(terminals_in_cc) < 2:
            selected_nodes = set(terminal_nodes)
            return selected_nodes, []

        subgraph_cc = G.subgraph(largest_cc).copy()
        for u, v, data in subgraph_cc.edges(data=True):
            data["steiner_cost"] = self._compute_cost(subgraph_cc, u, v)

        try:
            steiner = nx.approximation.steiner_tree(
                subgraph_cc, terminals_in_cc, weight="steiner_cost"
            )
            selected_nodes = set(steiner.nodes())
            selected_edges = list(steiner.edges())
        except Exception as e:
            logger.warning("Steiner tree 求解失敗: %s，回退到 seed nodes only", e)
            selected_nodes = set(terminal_nodes)
            selected_edges = []

        return selected_nodes, selected_edges

    # ------------------------------------------------------------------
    # 補充用 1-hop
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_with_one_hop(
        G: nx.DiGraph,
        nodes: set,
        edges: list,
        seeds: List[Dict[str, Any]],
        top_k: int,
    ) -> tuple:
        """當 PCST 結果節點不足時，從 seed 做 1-hop 擴展。"""
        new_nodes = set(nodes)
        new_edges = list(edges)
        edge_set = {(min(u, v), max(u, v)) for u, v in edges}

        for s in seeds:
            name = s["entity_name"]
            if name not in G:
                continue
            for _, nb in G.edges(name):
                if len(new_nodes) >= top_k:
                    break
                new_nodes.add(nb)
                ek = (min(name, nb), max(name, nb))
                if ek not in edge_set:
                    edge_set.add(ek)
                    new_edges.append((name, nb))
            if G.is_directed():
                for nb, _ in G.in_edges(name):
                    if len(new_nodes) >= top_k:
                        break
                    new_nodes.add(nb)
                    ek = (min(name, nb), max(name, nb))
                    if ek not in edge_set:
                        edge_set.add(ek)
                        new_edges.append((nb, name))

        return new_nodes, new_edges

    # ------------------------------------------------------------------
    # edge cost
    # ------------------------------------------------------------------

    def _compute_cost(self, G: nx.Graph, u: str, v: str) -> float:
        data = G.edges[u, v] if G.has_edge(u, v) else {}
        w = float(data.get("weight", 1.0))

        if self.cost_mode == "inverse_weight":
            return 1.0 / max(w, 0.01)
        elif self.cost_mode == "inverse_log_weight":
            import math
            return 1.0 / max(math.log1p(w), 0.01)
        elif self.cost_mode == "uniform":
            return 1.0
        else:
            return 1.0 / max(w, 0.01)
