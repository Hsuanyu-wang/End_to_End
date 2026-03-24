"""
Think-on-Graph (ToG) Strategy

基於 ToG 論文的 LLM-guided iterative graph traversal：
1. 從 seed entities 出發
2. 每輪取鄰居 → LLM 評分 path relevance → beam search 選 top-k paths
3. LLM 判斷資訊是否充分 → 不足則繼續迭代

直接操作 NetworkX graph，不依賴外部 graph_index。
"""

import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from src.graph_retriever.strategies.base import (
    BaseTraversalStrategy,
    TraversalResult,
    register_strategy,
)

logger = logging.getLogger(__name__)


@register_strategy("tog_refine")
class ToGRefineStrategy(BaseTraversalStrategy):

    def __init__(
        self,
        llm: Optional[Any] = None,
        max_iterations: int = 3,
        beam_width: int = 5,
        prune_top_n: int = 3,
        use_llm_scoring: bool = True,
        use_llm_sufficiency: bool = True,
    ):
        """
        Args:
            llm: LLM 實例（需有 complete(prompt) 方法；None 時退化為 weight-based）
            max_iterations: 最大迭代輪數
            beam_width: 每輪保留的最佳 path 數量
            use_llm_scoring: 是否用 LLM 對 path 評分（False 時以 edge weight 排序）
            use_llm_sufficiency: 是否用 LLM 判斷資訊充分性
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.beam_width = beam_width
        self.use_llm_scoring = use_llm_scoring and (llm is not None)
        self.use_llm_sufficiency = use_llm_sufficiency and (llm is not None)

    def get_name(self) -> str:
        return f"ToGRefine(iter={self.max_iterations}, beam={self.beam_width}), prune={self.prune_top_n})"

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
            return TraversalResult(metadata={"strategy": "tog_refine", "seeds_found": 0})

        # beam: 每條 path 是一個 list of entity_name
        beam: List[tuple] = [
            ([s["entity_name"]], 10.0) for s in valid_seeds[:self.beam_width]
        ]

        all_visited_nodes: set = {s["entity_name"] for s in valid_seeds}
        all_visited_edges: set = set()
        actual_iterations = 0

        for iteration in range(self.max_iterations):
            actual_iterations = iteration + 1
            logger.info("ToGRefine 迭代 %d/%d, beam size=%d", iteration + 1, self.max_iterations, len(beam))

            candidate_paths: List[tuple] = []  # (path, score)

            for path, current_score in beam:
                tail = path[-1]
                all_neighbors = self._get_neighbors(nx_graph, tail)
                
                # 過濾掉已經在當前路徑中的節點（避免迴圈）
                valid_neighbors = [(nb, data) for nb, data in all_neighbors if nb not in set(path)]

                if not valid_neighbors:
                    continue

                # 2. 【核心優化】使用 LLM 進行 Batch Pruning（關係修剪）
                # 讓 LLM 從所有鄰居中挑出最相關的 top-N，並給予評分
                pruned_neighbors = self._llm_prune_neighbors(
                    tail, valid_neighbors, query, top_n=self.prune_top_n
                )

                for nb, edge_data, step_score in pruned_neighbors:
                    new_path = path + [nb]
                    # 累積路徑分數（可以使用平均或衰減，這裡採用相加後平均的概念）
                    new_score = (current_score * len(path) + step_score) / len(new_path)
                    
                    candidate_paths.append((new_path, new_score))
                    all_visited_nodes.add(nb)
                    edge_key = (min(tail, nb), max(tail, nb))
                    all_visited_edges.add(edge_key)

            if not candidate_paths:
                logger.info("ToGRefine 迭代 %d: 無新候選 path，提前結束", iteration + 1)
                break

            candidate_paths.sort(key=lambda x: x[1], reverse=True)
            beam = candidate_paths[:self.beam_width]

            beam_paths_only = [p for p, _ in beam]
            if self._check_sufficiency(nx_graph, beam_paths_only, query):
                logger.info("ToG 迭代 %d: 資訊充分，提前結束", actual_iterations)
                break

        # 收集結果
        for path in beam:
            for name in path:
                all_visited_nodes.add(name)
            for i in range(len(path) - 1):
                ek = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                all_visited_edges.add(ek)

        # 限制 top_k
        if len(all_visited_nodes) > top_k:
            # 以 beam path 中出現頻率排序，保留 top_k
            freq: Dict[str, int] = {}
            for path in beam:
                for name in path:
                    freq[name] = freq.get(name, 0) + 1
            for s in valid_seeds:
                freq[s["entity_name"]] = freq.get(s["entity_name"], 0) + 100

            sorted_nodes = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
            all_visited_nodes = set(sorted_nodes[:top_k])

        nodes_list = []
        for name in all_visited_nodes:
            if name not in nx_graph:
                continue
            nd = dict(nx_graph.nodes[name])
            nd["entity_name"] = name
            nd["rank"] = nx_graph.degree(name)
            nodes_list.append(nd)

        edges_list = []
        for u, v in all_visited_edges:
            if u not in all_visited_nodes or v not in all_visited_nodes:
                continue
            if nx_graph.has_edge(u, v):
                data = dict(nx_graph.edges[u, v])
            elif nx_graph.has_edge(v, u):
                data = dict(nx_graph.edges[v, u])
            else:
                continue
            edges_list.append({
                "src_tgt": (u, v),
                "weight": float(data.get("weight", 1.0)),
                "description": data.get("description", ""),
                "keywords": data.get("keywords", ""),
            })

        beam_paths_only = [p for p, _ in beam]
        return TraversalResult(
            nodes=nodes_list,
            edges=edges_list,
            metadata={
                "strategy": "tog_refine",
                "iterations": actual_iterations,
                "beam_width": self.beam_width,
                "seeds_found": len(valid_seeds),
                "beam_paths": beam_paths_only,
            },
        )

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _get_neighbors(
        G: nx.DiGraph, node: str
    ) -> List[tuple]:
        """取得 node 的所有鄰居（含 edge data），同時考慮 in/out edges。"""
        neighbors = []
        seen = set()

        for _, nb, data in G.edges(node, data=True):
            if nb not in seen:
                seen.add(nb)
                neighbors.append((nb, data))

        if G.is_directed():
            for nb, _, data in G.in_edges(node, data=True):
                if nb not in seen:
                    seen.add(nb)
                    neighbors.append((nb, data))

        return neighbors

    def _score_path(
        self,
        G: nx.DiGraph,
        path: List[str],
        last_edge_data: Dict[str, Any],
        query: str,
    ) -> float:
        """對一條 path 評分。"""
        if self.use_llm_scoring:
            return self._llm_score_path(G, path, query)

        # Fallback: edge weight 加總
        total_weight = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                total_weight += float(G.edges[u, v].get("weight", 1.0))
            elif G.has_edge(v, u):
                total_weight += float(G.edges[v, u].get("weight", 1.0))
            else:
                total_weight += 0.1

        # 正規化：平均 weight per hop
        n_hops = max(len(path) - 1, 1)
        return total_weight / n_hops

    def _llm_score_path(
        self,
        G: nx.DiGraph,
        path: List[str],
        query: str,
    ) -> float:
        """用 LLM 對 path 的 query-relevance 評分（0-10）。"""
        path_desc_parts = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                desc = G.edges[u, v].get("description", "")
            elif G.has_edge(v, u):
                desc = G.edges[v, u].get("description", "")
            else:
                desc = ""
            # 只取 description 的第一段（避免過長）
            short_desc = desc.split("<SEP>")[0][:200] if desc else ""
            path_desc_parts.append(f"{u} -> {v}: {short_desc}")

        path_text = "\n".join(path_desc_parts)

        prompt = (
            f"Query: {query}\n\n"
            f"Graph path:\n{path_text}\n\n"
            f"Rate how relevant this path is to the query on a scale of 0-10. "
            f"Return ONLY a single number."
        )

        try:
            response = self.llm.complete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
            # 嘗試提取數字
            for token in text.strip().split():
                try:
                    score = float(token)
                    return min(max(score, 0.0), 10.0)
                except ValueError:
                    continue
            return 5.0
        except Exception as e:
            logger.warning("ToGRefine LLM scoring 失敗: %s", e)
            return 5.0

    def _check_sufficiency(
        self,
        G: nx.DiGraph,
        beam: List[List[str]],
        query: str,
    ) -> bool:
        """判斷 beam 中收集到的資訊是否足以回答 query。"""
        if not self.use_llm_sufficiency:
            return False

        # 收集 beam paths 中所有 entity descriptions
        all_descs = []
        for path in beam[:3]:
            for name in path:
                if name in G:
                    desc = G.nodes[name].get("description", "")
                    if desc:
                        all_descs.append(f"{name}: {desc[:150]}")

        if not all_descs:
            return False

        context = "\n".join(all_descs[:10])
        prompt = (
            f"Query: {query}\n\n"
            f"Collected context:\n{context}\n\n"
            f"Is the above context sufficient to answer the query? "
            f"Answer only YES or NO."
        )

        try:
            response = self.llm.complete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
            return "YES" in text.upper()
        except Exception as e:
            logger.warning("ToGRefine sufficiency check 失敗: %s", e)
            return False

    def _llm_prune_neighbors(
        self,
        tail: str,
        neighbors: List[tuple],
        query: str,
        top_n: int
    ) -> List[tuple]:
        """
        【ToG 降維核心】將所有鄰居打包進一個 Prompt，讓 LLM 一次性挑選並評分。
        回傳: List of (neighbor_name, edge_data, score)
        """
        if not self.use_llm_scoring or len(neighbors) <= top_n:
            # 如果不使用 LLM 或鄰居數量本來就很少，直接給預設分數回傳
            return [(nb, data, 5.0) for nb, data in neighbors[:top_n]]

        # 構建選擇題 Prompt
        options_text = ""
        for i, (nb, data) in enumerate(neighbors):
            desc = data.get("description", "")[:100]  # 限制長度
            options_text += f"[{i}] Entity: {nb} | Relation Context: {desc}\n"

        prompt = (
            f"Question: {query}\n\n"
            f"Current Entity: {tail}\n\n"
            f"Available connected entities to explore next:\n{options_text}\n"
            f"Task: Select the top {top_n} most relevant entities to explore to answer the question. "
            f"Output your choice strictly as a JSON list of objects with 'index' (int) and 'score' (0-10 float).\n"
            f"Example: [{{\"index\": 0, \"score\": 8.5}}, {{\"index\": 2, \"score\": 6.0}}]"
        )

        try:
            response = self.llm.complete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
            
            # 這裡簡單使用字串解析，實務上建議用 json 模組解析 LLM 的 JSON 輸出
            import json
            import re
            
            # 嘗試用 Regex 抓出 JSON 陣列
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                selections = json.loads(json_str)
                
                pruned = []
                for sel in selections:
                    idx = int(sel.get("index", -1))
                    score = float(sel.get("score", 5.0))
                    if 0 <= idx < len(neighbors):
                        nb, data = neighbors[idx]
                        pruned.append((nb, data, score))
                
                # 依照分數排序並取 top_n
                pruned.sort(key=lambda x: x[2], reverse=True)
                return pruned[:top_n]

        except Exception as e:
            logger.warning("ToG Neighbor Pruning 失敗，退回預設邏輯: %s", e)
        
        # 發生錯誤時的 Fallback
        return [(nb, data, 5.0) for nb, data in neighbors[:top_n]]