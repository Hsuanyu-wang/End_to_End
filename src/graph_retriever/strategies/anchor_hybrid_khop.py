"""
Anchor Hybrid K-Hop Strategy

第 1 步：以 seed(anchor) 做 1-hop 擴展，使用 hybrid 分數排序：
- vector similarity（query 與候選文字 embedding cosine）
- BM25（query token 與候選 token）

第 2 步：後續 hop 僅擴展上一層分數前 top-k 候選。
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import networkx as nx

from src.graph_retriever.strategies.base import (
    BaseTraversalStrategy,
    TraversalResult,
    register_strategy,
)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@register_strategy("anchor_hybrid_khop")
class AnchorHybridKHopStrategy(BaseTraversalStrategy):

    def __init__(self, hop_k: int = 2, vector_weight: float = 0.5, bm25_weight: float = 0.5, embed_model: Any = None):
        self.hop_k = max(1, int(hop_k))
        self.vector_weight = float(vector_weight)
        self.bm25_weight = float(bm25_weight)
        self.embed_model = embed_model

    def get_name(self) -> str:
        return "AnchorHybridKHop"

    @staticmethod
    def _candidate_text(name: str, node_data: Dict[str, Any]) -> str:
        desc = str(node_data.get("description", "")).split("<SEP>")[0].strip()
        return f"{name} {desc}".strip()

    def _vector_scores(self, query: str, candidates: Dict[str, str]) -> Dict[str, float]:
        if not candidates:
            return {}
        if self.embed_model is None:
            return {k: 0.0 for k in candidates}

        q_vec = self.embed_model.get_text_embedding(query)
        q_norm = math.sqrt(_dot(q_vec, q_vec)) if q_vec else 0.0
        if q_norm <= 0:
            return {k: 0.0 for k in candidates}

        keys = list(candidates.keys())
        texts = [candidates[k] for k in keys]
        if hasattr(self.embed_model, "get_text_embedding_batch"):
            doc_vecs = self.embed_model.get_text_embedding_batch(texts)
        else:
            doc_vecs = [self.embed_model.get_text_embedding(t) for t in texts]

        out: Dict[str, float] = {}
        for k, vec in zip(keys, doc_vecs):
            d_norm = math.sqrt(_dot(vec, vec)) if vec else 0.0
            if d_norm <= 0:
                out[k] = 0.0
            else:
                out[k] = _dot(q_vec, vec) / (q_norm * d_norm)
        return out

    @staticmethod
    def _bm25_scores(query: str, candidates: Dict[str, str], k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        if not candidates:
            return {}

        doc_tokens = {k: _tokenize(v) for k, v in candidates.items()}
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {k: 0.0 for k in candidates}

        n_docs = len(doc_tokens)
        avgdl = sum(len(toks) for toks in doc_tokens.values()) / max(n_docs, 1)
        df = Counter()
        for toks in doc_tokens.values():
            for t in set(toks):
                df[t] += 1

        scores: Dict[str, float] = {}
        qtf = Counter(query_tokens)
        for doc_id, toks in doc_tokens.items():
            tf = Counter(toks)
            dl = len(toks) if toks else 1
            score = 0.0
            for term, _ in qtf.items():
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                n_q = df.get(term, 0)
                idf = math.log(1 + (n_docs - n_q + 0.5) / (n_q + 0.5))
                denom = f + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
                score += idf * (f * (k1 + 1) / max(denom, 1e-9))
            scores[doc_id] = score
        return scores

    def _collect_one_hop_candidates(
        self,
        nx_graph: nx.DiGraph,
        anchors: List[str],
        visited: set,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        node_candidates: Dict[str, Dict[str, Any]] = {}
        edge_candidates: List[Dict[str, Any]] = []
        seen_edges = set()

        for name in anchors:
            if name not in nx_graph:
                continue
            for u, v, data in nx_graph.edges(name, data=True):
                edge_key = (min(u, v), max(u, v))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge_candidates.append({
                        "src_tgt": (u, v),
                        "weight": float(data.get("weight", 1.0)),
                        **{k: v_ for k, v_ in data.items() if k != "weight"},
                    })
                nei = v if u == name else u
                if nei in nx_graph and nei not in visited:
                    nd = dict(nx_graph.nodes[nei])
                    nd["entity_name"] = nei
                    nd["rank"] = nx_graph.degree(nei)
                    node_candidates[nei] = nd

            if nx_graph.is_directed():
                for u, v, data in nx_graph.in_edges(name, data=True):
                    edge_key = (min(u, v), max(u, v))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edge_candidates.append({
                            "src_tgt": (u, v),
                            "weight": float(data.get("weight", 1.0)),
                            **{k: v_ for k, v_ in data.items() if k != "weight"},
                        })
                    if u in nx_graph and u not in visited:
                        nd = dict(nx_graph.nodes[u])
                        nd["entity_name"] = u
                        nd["rank"] = nx_graph.degree(u)
                        node_candidates[u] = nd
        return node_candidates, edge_candidates

    def traverse(
        self,
        nx_graph: nx.DiGraph,
        seed_entities: List[Dict[str, Any]],
        query: str,
        top_k: int = 20,
        **kwargs,
    ) -> TraversalResult:
        hop_k = max(1, int(kwargs.get("hop_k", self.hop_k)))
        top_k = max(1, int(top_k))
        seed_names = [e["entity_name"] for e in seed_entities if e["entity_name"] in nx_graph]
        if not seed_names:
            return TraversalResult(
                metadata={"strategy": "anchor_hybrid_khop", "seeds_found": 0, "hop_k": hop_k}
            )

        collected_nodes: Dict[str, Dict[str, Any]] = {}
        collected_edges: List[Dict[str, Any]] = []
        visited = set(seed_names)

        for s in seed_names:
            d = dict(nx_graph.nodes[s])
            d["entity_name"] = s
            d["rank"] = nx_graph.degree(s)
            d["hop"] = 0
            collected_nodes[s] = d

        # Step 1: anchor 1-hop + hybrid ranking
        one_hop_nodes, one_hop_edges = self._collect_one_hop_candidates(nx_graph, seed_names, visited)
        candidate_texts = {
            name: self._candidate_text(name, data)
            for name, data in one_hop_nodes.items()
        }
        vec_scores = self._vector_scores(query, candidate_texts)
        bm25_scores = self._bm25_scores(query, candidate_texts)
        vec_norm = _minmax_normalize(vec_scores)
        bm25_norm = _minmax_normalize(bm25_scores)

        ranked_one_hop = sorted(
            one_hop_nodes.items(),
            key=lambda kv: (
                self.vector_weight * vec_norm.get(kv[0], 0.0)
                + self.bm25_weight * bm25_norm.get(kv[0], 0.0),
                kv[1].get("rank", 0),
            ),
            reverse=True,
        )
        first_frontier = [name for name, _ in ranked_one_hop[:top_k]]

        for name, data in ranked_one_hop:
            data["hop"] = 1
            data["hybrid_score"] = (
                self.vector_weight * vec_norm.get(name, 0.0)
                + self.bm25_weight * bm25_norm.get(name, 0.0)
            )
            data["vector_score"] = vec_scores.get(name, 0.0)
            data["bm25_score"] = bm25_scores.get(name, 0.0)
            collected_nodes[name] = data
        collected_edges.extend(one_hop_edges)
        visited.update(one_hop_nodes.keys())

        layer_stats = [{
            "hop": 1,
            "candidates": len(one_hop_nodes),
            "expanded_top_k": len(first_frontier),
        }]

        frontier = first_frontier
        for hop in range(2, hop_k + 1):
            if not frontier:
                break
            next_nodes, next_edges = self._collect_one_hop_candidates(nx_graph, frontier, visited)
            ranked_next = sorted(
                next_nodes.items(),
                key=lambda kv: kv[1].get("rank", 0),
                reverse=True,
            )
            next_frontier = [name for name, _ in ranked_next[:top_k]]

            for name, data in ranked_next:
                data["hop"] = hop
                collected_nodes[name] = data
            collected_edges.extend(next_edges)
            visited.update(next_nodes.keys())

            layer_stats.append({
                "hop": hop,
                "candidates": len(next_nodes),
                "expanded_top_k": len(next_frontier),
            })
            frontier = next_frontier

        unique_edges = {}
        for e in collected_edges:
            src_tgt = e.get("src_tgt")
            if not src_tgt:
                continue
            k = (min(src_tgt[0], src_tgt[1]), max(src_tgt[0], src_tgt[1]))
            if k not in unique_edges:
                unique_edges[k] = e
        edges_list = sorted(
            unique_edges.values(),
            key=lambda x: (x.get("weight", 0.0), str(x.get("src_tgt", ""))),
            reverse=True,
        )
        nodes_list = sorted(
            collected_nodes.values(),
            key=lambda x: (x.get("hybrid_score", 0.0), x.get("rank", 0)),
            reverse=True,
        )[:top_k]

        return TraversalResult(
            nodes=nodes_list,
            edges=edges_list,
            metadata={
                "strategy": "anchor_hybrid_khop",
                "seeds_found": len(seed_names),
                "hop_k": hop_k,
                "layer_stats": layer_stats,
                "vector_weight": self.vector_weight,
                "bm25_weight": self.bm25_weight,
                "total_nodes": len(nodes_list),
                "total_edges": len(edges_list),
            },
        )
