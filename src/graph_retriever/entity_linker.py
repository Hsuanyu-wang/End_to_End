"""
LightRAG Entity Linker

封裝 LightRAG 的 Entity Linking 邏輯（keyword extraction + VDB search + round-robin merge），
為下游 graph traversal strategy 提供 seed entities。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lightrag.lightrag import QueryParam

logger = logging.getLogger(__name__)


@dataclass
class SeedEntity:
    """Entity linking 的結果：一個被連結到 KG 的 seed entity。"""
    entity_name: str
    vdb_score: float
    node_data: Dict[str, Any] = field(default_factory=dict)
    degree: int = 0
    source: str = "local"  # "local" | "global"


@dataclass
class EntityLinkingResult:
    """Entity Linker 的完整回傳結果。"""
    seeds: List[SeedEntity]
    hl_keywords: List[str] = field(default_factory=list)
    ll_keywords: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)


class LightRAGEntityLinker:
    """
    利用 LightRAG 的 entities_vdb / relationships_vdb 執行 Entity Linking。

    流程與 LightRAG 內部 `_perform_kg_search` 一致：
    1. LLM keyword extraction → hl_keywords, ll_keywords
    2. entities_vdb.query (local) / relationships_vdb.query (global)
    3. round-robin merge (hybrid/mix)
    """

    def __init__(
        self,
        rag_instance: Any,
        mode: str = "hybrid",
        top_k: int = 10,
    ):
        self.rag = rag_instance
        self.mode = mode
        self.top_k = top_k

    async def alink(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> EntityLinkingResult:
        """非同步 entity linking。"""
        from dataclasses import asdict
        from lightrag.operate import get_keywords_from_query

        use_mode = mode or self.mode
        use_top_k = top_k or self.top_k

        global_config = asdict(self.rag)
        global_config["embedding_func"] = self.rag.embedding_func

        param = QueryParam(mode=use_mode, top_k=use_top_k)

        hl_keywords, ll_keywords = await get_keywords_from_query(
            query, param, global_config, self.rag.llm_response_cache
        )

        logger.info(
            "Entity linking keywords: hl=%s, ll=%s", hl_keywords, ll_keywords
        )

        local_entities: List[Dict[str, Any]] = []
        global_entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []

        kg = self.rag.chunk_entity_relation_graph

        # Local: entities_vdb → node data
        if use_mode in ("local", "hybrid", "mix") and ll_keywords:
            ll_str = ", ".join(ll_keywords)
            local_entities, local_relations = await self._query_entities(
                ll_str, use_top_k, kg
            )
            relations.extend(local_relations)

        # Global: relationships_vdb → edge → source/target nodes
        if use_mode in ("global", "hybrid", "mix") and hl_keywords:
            hl_str = ", ".join(hl_keywords)
            global_entities, global_relations = await self._query_via_relationships(
                hl_str, use_top_k, kg
            )
            relations.extend(global_relations)

        seeds = self._round_robin_merge(local_entities, global_entities)

        return EntityLinkingResult(
            seeds=seeds,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            relations=relations,
        )

    def link(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> EntityLinkingResult:
        """同步 entity linking wrapper。"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.alink(query, top_k, mode))

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    async def _query_entities(
        self, keywords_str: str, top_k: int, kg: Any
    ) -> tuple:
        """透過 entities_vdb 搜尋 entity nodes（等同 LightRAG _get_node_data 的前半段）。"""
        results = await self.rag.entities_vdb.query(keywords_str, top_k=top_k)
        if not results:
            return [], []

        node_ids = [r["entity_name"] for r in results]

        nodes_dict, degrees_dict = await asyncio.gather(
            kg.get_nodes_batch(node_ids),
            kg.node_degrees_batch(node_ids),
        )

        entities = []
        for r in results:
            name = r["entity_name"]
            node = nodes_dict.get(name)
            if node is None:
                continue
            entities.append({
                **node,
                "entity_name": name,
                "rank": degrees_dict.get(name, 0),
                "vdb_score": r.get("distance", r.get("score", 0.0)),
                "_source": "local",
            })

        # 取得 1-hop edges 資訊供下游參考
        node_names = [e["entity_name"] for e in entities]
        batch_edges_dict = await kg.get_nodes_edges_batch(node_names)
        all_edge_pairs = set()
        for node_name in node_names:
            for e in batch_edges_dict.get(node_name, []):
                all_edge_pairs.add(tuple(sorted(e)))

        edge_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edge_pairs]
        edge_data_dict = await kg.get_edges_batch(edge_dicts) if edge_dicts else {}

        relations = []
        for pair in all_edge_pairs:
            props = edge_data_dict.get(pair)
            if props:
                relations.append({"src_tgt": pair, **props})

        return entities, relations

    async def _query_via_relationships(
        self, keywords_str: str, top_k: int, kg: Any
    ) -> tuple:
        """透過 relationships_vdb 搜尋 edges，再取 source/target nodes。"""
        results = await self.rag.relationships_vdb.query(keywords_str, top_k=top_k)
        if not results:
            return [], []

        edge_pairs = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
        edge_data_dict = await kg.get_edges_batch(edge_pairs)

        relations = []
        entity_names_seen = set()
        entity_names_ordered = []

        for r in results:
            pair = (r["src_id"], r["tgt_id"])
            props = edge_data_dict.get(pair)
            if props:
                if "weight" not in props:
                    props["weight"] = 1.0
                relations.append({
                    "src_tgt": pair,
                    "src_id": r["src_id"],
                    "tgt_id": r["tgt_id"],
                    **props,
                })
            for name in (r["src_id"], r["tgt_id"]):
                if name not in entity_names_seen:
                    entity_names_seen.add(name)
                    entity_names_ordered.append(name)

        nodes_dict = await kg.get_nodes_batch(entity_names_ordered)

        entities = []
        for name in entity_names_ordered:
            node = nodes_dict.get(name)
            if node is None:
                continue
            entities.append({
                **node,
                "entity_name": name,
                "rank": 0,
                "vdb_score": 0.0,
                "_source": "global",
            })

        return entities, relations

    @staticmethod
    def _round_robin_merge(
        local_entities: List[Dict[str, Any]],
        global_entities: List[Dict[str, Any]],
    ) -> List[SeedEntity]:
        """Round-robin merge local + global entities，與 LightRAG hybrid 行為一致。"""
        merged: List[SeedEntity] = []
        seen: set = set()
        max_len = max(len(local_entities), len(global_entities), 0)

        for i in range(max_len):
            if i < len(local_entities):
                ent = local_entities[i]
                name = ent.get("entity_name", "")
                if name and name not in seen:
                    seen.add(name)
                    merged.append(SeedEntity(
                        entity_name=name,
                        vdb_score=float(ent.get("vdb_score", 0.0)),
                        node_data=ent,
                        degree=int(ent.get("rank", 0)),
                        source="local",
                    ))

            if i < len(global_entities):
                ent = global_entities[i]
                name = ent.get("entity_name", "")
                if name and name not in seen:
                    seen.add(name)
                    merged.append(SeedEntity(
                        entity_name=name,
                        vdb_score=float(ent.get("vdb_score", 0.0)),
                        node_data=ent,
                        degree=int(ent.get("rank", 0)),
                        source="global",
                    ))

        return merged
