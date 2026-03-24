"""
Personalized PageRank (PPR) Strategy

從 seed entities 出發，利用 Personalized PageRank 在 KG 上擴散，
取 PPR score 最高的 top-k 節點及其 induced subgraph。

支援三種 edge weight 模式：
- semantic: 使用 LightRAG 原始的 semantic weight
- degree: 使用 topological degree 作為 weight
- combined: semantic * log(1 + degree)
"""

import logging
import math
from typing import Any, Dict, List, Literal

import networkx as nx

from src.graph_retriever.strategies.base import (
    BaseTraversalStrategy,
    TraversalResult,
    register_strategy,
)

logger = logging.getLogger(__name__)

WeightMode = Literal["semantic", "degree", "combined"]


@register_strategy("ppr")
class PPRStrategy(BaseTraversalStrategy):

    def __init__(
        self,
        alpha: float = 0.85,
        weight_mode: WeightMode = "semantic",
        include_seeds: bool = True,
    ):
        """
        Args:
            alpha: PageRank damping factor（越高越集中在 seed 附近）
            weight_mode: edge weight 計算方式
            include_seeds: 是否強制將 seed entities 加入結果
        """
        self.alpha = alpha
        self.weight_mode = weight_mode
        self.include_seeds = include_seeds

    def get_name(self) -> str:
        return f"PPR(alpha={self.alpha}, weight={self.weight_mode})"

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
            return TraversalResult(metadata={"strategy": "ppr", "seeds_found": 0})

        weighted_graph = self._prepare_weighted_graph(nx_graph)

        personalization = self._build_personalization(weighted_graph, valid_seeds)

        try:
            ppr_scores = nx.pagerank(
                weighted_graph,
                alpha=self.alpha,
                personalization=personalization,
                weight="ppr_weight",
                max_iter=100,
                tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PPR 未收斂，使用預設 max_iter=200 重試")
            ppr_scores = nx.pagerank(
                weighted_graph,
                alpha=self.alpha,
                personalization=personalization,
                weight="ppr_weight",
                max_iter=200,
                tol=1e-4,
            )

        ranked = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)

        selected_names = set()
        if self.include_seeds:
            for s in valid_seeds:
                selected_names.add(s["entity_name"])

        for name, score in ranked:
            if len(selected_names) >= top_k:
                break
            selected_names.add(name)

        subgraph = nx_graph.subgraph(selected_names)

        nodes_list = []
        for name in selected_names:
            if name not in nx_graph:
                continue
            node_data = dict(nx_graph.nodes[name])
            node_data["entity_name"] = name
            node_data["ppr_score"] = ppr_scores.get(name, 0.0)
            node_data["rank"] = nx_graph.degree(name)
            nodes_list.append(node_data)

        nodes_list.sort(key=lambda x: x["ppr_score"], reverse=True)

        edges_list = []
        for u, v, data in subgraph.edges(data=True):
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
                "strategy": "ppr",
                "alpha": self.alpha,
                "weight_mode": self.weight_mode,
                "seeds_found": len(valid_seeds),
                "total_nodes": len(nodes_list),
                "total_edges": len(edges_list),
                "top_ppr_score": ranked[0][1] if ranked else 0.0,
            },
        )

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    def _prepare_weighted_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        """根據 weight_mode 為 edge 設定 ppr_weight。"""
        H = G.copy()

        for u, v, data in H.edges(data=True):
            semantic_w = float(data.get("weight", 1.0))

            if self.weight_mode == "semantic":
                data["ppr_weight"] = max(semantic_w, 0.01)
            elif self.weight_mode == "degree":
                deg = H.degree(u) + H.degree(v)
                data["ppr_weight"] = max(float(deg), 1.0)
            elif self.weight_mode == "combined":
                deg = H.degree(u) + H.degree(v)
                data["ppr_weight"] = max(
                    semantic_w * math.log1p(deg), 0.01
                )
            else:
                data["ppr_weight"] = max(semantic_w, 0.01)

        return H

    @staticmethod
    def _build_personalization(
        G: nx.DiGraph, seeds: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """建構 PPR personalization vector：以 vdb_score 為權重。"""
        personalization: Dict[str, float] = {}
        for s in seeds:
            name = s["entity_name"]
            score = float(s.get("vdb_score", 1.0))
            personalization[name] = max(score, 0.01)

        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v / total for k, v in personalization.items()}

        return personalization
