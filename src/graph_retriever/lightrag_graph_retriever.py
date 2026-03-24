"""
LightRAG Graph Retriever

組合 LightRAGEntityLinker + TraversalStrategy 的主 Retriever。
保留 LightRAG 的 Entity Linking（keyword extraction + VDB + round-robin），
將 graph traversal 委派給可插拔的策略（one_hop / ppr / pcst / tog）。
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Iterable

from src.graph_retriever.base_retriever import BaseGraphRetriever
from src.graph_retriever.entity_linker import (
    LightRAGEntityLinker,
    EntityLinkingResult,
    SeedEntity,
)
from src.graph_retriever.strategies import (
    TraversalStrategyRegistry,
    BaseTraversalStrategy,
    TraversalResult,
)
from src.graph_adapter.converters.lightrag_converter import lightrag_to_networkx
from src.rag.graph.lightrag_id_mapper import ChunkIDMapper

logger = logging.getLogger(__name__)


class LightRAGGraphRetriever(BaseGraphRetriever):
    """
    使用 LightRAG Entity Linking + 可插拔 graph traversal strategy 的 Retriever。

    支援策略：one_hop, ppr, pcst, tog
    """

    def __init__(
        self,
        rag_instance: Any = None,
        strategy: str = "ppr",
        mode: str = "hybrid",
        top_k: int = 10,
        settings: Any = None,
        storage_path: str = None,
        **strategy_kwargs,
    ):
        """
        Args:
            rag_instance: LightRAG 實例
            strategy: traversal 策略名稱
            mode: entity linking 模式 (local/global/hybrid/mix)
            top_k: entity linking 的 VDB top_k
            settings: ModelSettings（用於延遲初始化 LightRAG）
            storage_path: LightRAG 儲存路徑
            **strategy_kwargs: 傳給策略的額外參數
        """
        self.rag_instance = rag_instance
        self.strategy_name = strategy
        self.mode = mode
        self.el_top_k = top_k
        self.settings = settings
        self._storage_path = storage_path
        self.strategy_kwargs = strategy_kwargs

        self._entity_linker: Optional[LightRAGEntityLinker] = None
        self._nx_graph = None
        self._strategy: Optional[BaseTraversalStrategy] = None
        self._chunk_id_mapper: Optional[ChunkIDMapper] = None

    def get_name(self) -> str:
        return f"LightRAG_{self.strategy_name}_{self.mode}"

    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """延遲初始化：載入 LightRAG 實例、NetworkX 圖、策略。"""
        config = config or {}

        if self.rag_instance is None:
            storage_path = config.get("storage_path") or self._storage_path
            if not storage_path:
                raise ValueError("需要提供 rag_instance 或 storage_path")

            if not os.path.exists(storage_path):
                raise ValueError(f"LightRAG 索引不存在: {storage_path}")

            from src.rag.graph.lightrag import _create_lightrag_at_path
            use_settings = self.settings or config.get("settings")
            if use_settings is None:
                from src.config.settings import get_settings
                use_settings = get_settings()

            self.rag_instance = _create_lightrag_at_path(
                use_settings, working_dir=storage_path
            )
            logger.info("LightRAG 實例載入完成: %s", storage_path)

        self._entity_linker = LightRAGEntityLinker(
            self.rag_instance, mode=self.mode, top_k=self.el_top_k
        )
        self._nx_graph = lightrag_to_networkx(self.rag_instance)
        working_dir = getattr(self.rag_instance, "working_dir", None)
        if working_dir:
            self._chunk_id_mapper = ChunkIDMapper(working_dir)
        self._strategy = TraversalStrategyRegistry.create(
            self.strategy_name, **self.strategy_kwargs
        )

        logger.info(
            "LightRAGGraphRetriever 初始化完成: strategy=%s, mode=%s, graph=%d nodes/%d edges",
            self._strategy.get_name(),
            self.mode,
            self._nx_graph.number_of_nodes(),
            self._nx_graph.number_of_edges(),
        )

    def _ensure_initialized(self, graph_data: Optional[Dict[str, Any]] = None):
        if self._strategy is not None:
            return
        init_config: Dict[str, Any] = {}
        if graph_data:
            if "storage_path" in graph_data:
                init_config["storage_path"] = graph_data["storage_path"]
            if "settings" in graph_data:
                init_config["settings"] = graph_data["settings"]
        self.initialize(init_config)

    # ------------------------------------------------------------------
    # 同步 retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ) -> Dict[str, Any]:
        """同步檢索：Entity Linking → Graph Traversal → Context Extraction。"""
        self._ensure_initialized(graph_data)

        logger.info("[%s] 開始檢索: %s", self.get_name(), query[:80])

        el_result = self._entity_linker.link(query)
        if not el_result.seeds:
            logger.warning("Entity linking 未找到任何 seed entity")
            return {
                "contexts": [],
                "nodes": [],
                "metadata": {"strategy": self.strategy_name, "seeds_found": 0},
            }

        seed_dicts = [
            {
                "entity_name": s.entity_name,
                "vdb_score": s.vdb_score,
                "degree": s.degree,
                "source": s.source,
            }
            for s in el_result.seeds
        ]

        traversal_result = self._strategy.traverse(
            self._nx_graph, seed_dicts, query, top_k=top_k, **kwargs
        )

        contexts = self._extract_contexts(traversal_result, el_result)
        retrieved_ids = self._extract_retrieved_ids(
            traversal_result=traversal_result, top_k=top_k
        )

        return {
            "contexts": contexts,
            "nodes": traversal_result.nodes,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                **traversal_result.metadata,
                "hl_keywords": el_result.hl_keywords,
                "ll_keywords": el_result.ll_keywords,
                "retriever": self.get_name(),
            },
        }

    # ------------------------------------------------------------------
    # 非同步 aretrieve
    # ------------------------------------------------------------------

    async def aretrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ) -> Dict[str, Any]:
        """非同步檢索。"""
        self._ensure_initialized(graph_data)

        logger.info("[%s] 開始非同步檢索: %s", self.get_name(), query[:80])

        el_result = await self._entity_linker.alink(query)
        if not el_result.seeds:
            logger.warning("Entity linking 未找到任何 seed entity")
            return {
                "contexts": [],
                "nodes": [],
                "metadata": {"strategy": self.strategy_name, "seeds_found": 0},
            }

        seed_dicts = [
            {
                "entity_name": s.entity_name,
                "vdb_score": s.vdb_score,
                "degree": s.degree,
                "source": s.source,
            }
            for s in el_result.seeds
        ]

        traversal_result = self._strategy.traverse(
            self._nx_graph, seed_dicts, query, top_k=top_k, **kwargs
        )

        contexts = self._extract_contexts(traversal_result, el_result)
        retrieved_ids = self._extract_retrieved_ids(
            traversal_result=traversal_result, top_k=top_k
        )

        return {
            "contexts": contexts,
            "nodes": traversal_result.nodes,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                **traversal_result.metadata,
                "hl_keywords": el_result.hl_keywords,
                "ll_keywords": el_result.ll_keywords,
                "retriever": self.get_name(),
            },
        }

    @staticmethod
    def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
        """去重且保留第一次出現順序。"""
        seen = set()
        out: List[str] = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    @staticmethod
    def _parse_chunk_ids_from_source_id(source_id_value: Any) -> List[str]:
        """
        從 graphml 的 source_id 欄位解析 chunk id。

        LightRAG graphml:
        - node.source_id 可能是 "chunk-a<SEP>chunk-b..."
        - edge.source_id 也可能是單一 "chunk-x"
        """
        if not source_id_value:
            return []

        if isinstance(source_id_value, list):
            # 容錯：若讀到的是 list（理論上 graphml 讀回通常是 str）
            candidates = []
            for v in source_id_value:
                candidates.extend(
                    LightRAGGraphRetriever._parse_chunk_ids_from_source_id(v)
                )
            return candidates

        text = str(source_id_value)
        if not text.strip():
            return []

        parts = [p.strip() for p in text.split("<SEP>")]
        chunk_ids: List[str] = []
        for p in parts:
            if not p:
                continue
            # 只接受 chunk-*（避免把 entity 名稱/雜訊誤當成 id）
            if p.startswith("chunk-"):
                chunk_ids.append(p)
        return chunk_ids

    def _get_edge_source_id(self, u: Any, v: Any) -> Optional[Any]:
        if self._nx_graph is None:
            return None
        if self._nx_graph.has_edge(u, v):
            data = self._nx_graph.edges[u, v]
            return data.get("source_id")
        if self._nx_graph.has_edge(v, u):
            data = self._nx_graph.edges[v, u]
            return data.get("source_id")
        return None

    def _extract_retrieved_ids(
        self,
        traversal_result: TraversalResult,
        top_k: int,
    ) -> List[str]:
        """
        把 traversal nodes/edges 對應到 chunk id，再映射回 QA 的 REF(uuid)。

        Notes:
        - strategy-based wrapper 原本沒有 chunk mapping，所以 retrieved_ids 為空；
          這裡補上讓 retrieval 指標可用。
        """
        if self._chunk_id_mapper is None:
            return []

        # 取得 chunk id（先 nodes，再 edges）
        chunk_ids: List[str] = []

        # 為了讓 MRR 更穩定：對 nodes 做可預期排序
        nodes = traversal_result.nodes or []
        nodes_sorted = sorted(
            nodes,
            key=lambda n: (
                float(n.get("ppr_score", n.get("rank", 0.0))) if n else 0.0,
                str(n.get("entity_name", "")) if n else "",
            ),
            reverse=True,
        )
        for node in nodes_sorted:
            src = node.get("source_id", "")
            chunk_ids.extend(self._parse_chunk_ids_from_source_id(src))

        edges = traversal_result.edges or []
        # edges 的 order 也用 weight/端點做穩定排序（邊權重是 float/缺省 0）
        edges_sorted = sorted(
            edges,
            key=lambda e: (
                float(e.get("weight", 0.0)) if e else 0.0,
                str(e.get("src_tgt", "")) if e else "",
            ),
            reverse=True,
        )
        for edge in edges_sorted:
            src_tgt = edge.get("src_tgt")
            if not src_tgt or not isinstance(src_tgt, tuple) or len(src_tgt) != 2:
                continue
            u, v = src_tgt
            edge_source_id = self._get_edge_source_id(u, v)
            chunk_ids.extend(self._parse_chunk_ids_from_source_id(edge_source_id))

        # chunk id -> REF(uuid)
        retrieved_ids: List[str] = []
        seen: set = set()
        for chunk_id in chunk_ids:
            # chunk_id_mapping.json 內的 key 是 chunk-*
            ref = self._chunk_id_mapper.get_original_no(chunk_id)
            rid = ref if ref else chunk_id
            if rid in seen:
                continue
            seen.add(rid)
            retrieved_ids.append(str(rid))

        # 再保險去重（理論上不需要，但維持一致性）
        return self._dedupe_preserve_order(retrieved_ids)

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_contexts(
        traversal: TraversalResult,
        el_result: EntityLinkingResult,
    ) -> List[str]:
        """
        從 traversal result 中提取 contexts，格式與 LightRAGRetriever 相容。

        優先順序：entity descriptions > edge descriptions > chunk references
        """
        contexts: List[str] = []
        seen: set = set()

        # 1. Entity descriptions
        for node in traversal.nodes:
            desc = node.get("description", "")
            name = node.get("entity_name", "")
            if desc and desc not in seen:
                seen.add(desc)
                # 只取第一段 description（<SEP> 分隔的多段合併描述）
                primary_desc = desc.split("<SEP>")[0].strip()
                if primary_desc:
                    contexts.append(primary_desc)

        # 2. Edge descriptions
        for edge in traversal.edges:
            desc = edge.get("description", "")
            if desc and desc not in seen:
                seen.add(desc)
                primary_desc = desc.split("<SEP>")[0].strip()
                if primary_desc:
                    contexts.append(primary_desc)

        # 3. Entity linking 階段取得的 relation descriptions（補充）
        for rel in el_result.relations:
            desc = rel.get("description", "")
            if desc and desc not in seen:
                seen.add(desc)
                primary_desc = desc.split("<SEP>")[0].strip()
                if primary_desc:
                    contexts.append(primary_desc)

        return contexts
