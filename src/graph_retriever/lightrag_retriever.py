"""
LightRAG Retriever

分離 LightRAG 的檢索邏輯作為 Retriever 模組，
使用官方 query_data() / aquery_data() 取得結構化 context（而非 LLM 生成答案）。
"""

import os
from typing import Dict, Any, Optional, List
from src.graph_retriever.base_retriever import BaseGraphRetriever, GraphData


class LightRAGRetriever(BaseGraphRetriever):
    """
    LightRAG Retriever
    
    提供 LightRAG 的多種檢索模式：local/global/hybrid/mix/naive/bypass
    """
    
    def __init__(
        self,
        mode: str = "hybrid",
        rag_instance=None,
        settings: Any = None,
        data_type: str = "DI",
        sup: str = "",
        fast_test: bool = False,
        graph_source: Any = None,
        storage_path: str = None,
    ):
        self.mode = mode
        self.settings = settings
        self.data_type = data_type
        self.sup = sup
        self.fast_test = fast_test
        self._storage_path = storage_path

        if rag_instance is not None:
            self.rag_instance = rag_instance
        elif graph_source is not None and callable(getattr(graph_source, "query", None)):
            self.rag_instance = graph_source
        elif isinstance(graph_source, dict):
            self.rag_instance = graph_source.get("lightrag_instance")
            if self._storage_path is None:
                self._storage_path = graph_source.get("storage_path")
        else:
            self.rag_instance = None
    
    def get_name(self) -> str:
        return f"LightRAG_{self.mode.capitalize()}"
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """載入或建立 LightRAG 實例"""
        if self.rag_instance is not None:
            return
        
        config = config or {}
        storage_path = config.get('storage_path') or self._storage_path
        
        if not storage_path:
            from src.storage import get_storage_path
            schema_method = config.get('schema_method', 'lightrag_default')
            full_sup = f"{self.sup}_{schema_method}" if self.sup else schema_method
            storage_path = get_storage_path(
                storage_type="lightrag",
                data_type=self.data_type,
                method=full_sup,
                mode=self.mode,
                fast_test=self.fast_test
            )
        
        if not os.path.exists(storage_path):
            raise ValueError(f"LightRAG 索引不存在: {storage_path}")
        
        print(f"📂 載入 LightRAG 索引: {storage_path}")
        self._storage_path = storage_path
        
        from src.rag.graph.lightrag import _create_lightrag_at_path
        self.rag_instance = _create_lightrag_at_path(
            Settings=self.settings,
            working_dir=storage_path
        )
        print(f"✅ LightRAG Retriever 初始化完成 (模式: {self.mode})")

    # ------------------------------------------------------------------
    # 核心：從 aquery_data / query_data 取得結構化 context
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_query_data(raw_data: dict) -> Dict[str, Any]:
        """
        解析 LightRAG query_data() 回傳的結構化資料，
        提取 entities / relationships / chunks 作為 context。
        """
        contexts: List[str] = []
        entity_contexts: List[str] = []
        relation_contexts: List[str] = []
        chunk_contexts: List[str] = []

        if not raw_data or not isinstance(raw_data, dict):
            return {"contexts": [], "entity_contexts": [], "relation_contexts": [], "chunk_contexts": []}

        data_content = raw_data.get("data", {})

        # entities
        for ent in data_content.get("entities", []):
            if isinstance(ent, dict):
                desc = ent.get("description", "") or ent.get("entity_name", "")
                if desc:
                    entity_contexts.append(str(desc))
            elif isinstance(ent, str) and ent:
                entity_contexts.append(ent)

        # relationships
        for rel in data_content.get("relationships", []):
            if isinstance(rel, dict):
                desc = rel.get("description", "") or ""
                src = rel.get("src_id", "")
                tgt = rel.get("tgt_id", "")
                text = desc if desc else f"{src} -> {tgt}"
                if text:
                    relation_contexts.append(str(text))
            elif isinstance(rel, str) and rel:
                relation_contexts.append(rel)

        # chunks（最重要的 context 來源）
        for chunk in data_content.get("chunks", []):
            if isinstance(chunk, dict):
                content = chunk.get("content", "")
                if content:
                    chunk_contexts.append(content)
            elif isinstance(chunk, str) and chunk:
                chunk_contexts.append(chunk)

        contexts = entity_contexts + relation_contexts + chunk_contexts

        return {
            "contexts": contexts,
            "entity_contexts": entity_contexts,
            "relation_contexts": relation_contexts,
            "chunk_contexts": chunk_contexts,
        }

    def _ensure_initialized(self, graph_data: Optional[Dict[str, Any]] = None):
        """確保 rag_instance 已就緒"""
        if self.rag_instance is not None:
            return
        init_config: Dict[str, Any] = {}
        if graph_data:
            if "storage_path" in graph_data:
                init_config["storage_path"] = graph_data["storage_path"]
            si = graph_data.get("schema_info")
            if isinstance(si, dict):
                init_config["schema_method"] = si.get("method", "lightrag_default")
        self.initialize(init_config)

    def retrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """同步檢索：使用 query_data() 取得結構化 context"""
        self._ensure_initialized(graph_data)

        print(f"🔍 [LightRAG-{self.mode}] 開始檢索（query_data）...")
        try:
            from lightrag.lightrag import QueryParam
            raw_data = self.rag_instance.query_data(
                query,
                param=QueryParam(mode=self.mode)
            )
            parsed = self._parse_query_data(raw_data)
            contexts = parsed["contexts"]
        except Exception as e:
            print(f"⚠️  LightRAG 檢索失敗: {e}")
            import traceback; traceback.print_exc()
            contexts = []

        return {
            "contexts": contexts,
            "nodes": [],
            "metadata": {"mode": self.mode, "retriever": "lightrag"}
        }

    async def aretrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """非同步檢索：使用 aquery_data() 取得結構化 context"""
        self._ensure_initialized(graph_data)

        print(f"🔍 [LightRAG-{self.mode}] 開始非同步檢索（aquery_data）...")
        try:
            from lightrag.lightrag import QueryParam
            raw_data = await self.rag_instance.aquery_data(
                query,
                param=QueryParam(mode=self.mode)
            )
            parsed = self._parse_query_data(raw_data)
            contexts = parsed["contexts"]
        except Exception as e:
            print(f"⚠️  LightRAG 非同步檢索失敗: {e}")
            import traceback; traceback.print_exc()
            contexts = []

        return {
            "contexts": contexts,
            "nodes": [],
            "metadata": {"mode": self.mode, "retriever": "lightrag"}
        }
