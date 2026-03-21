import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.schema import Document as LlamaDocument

from .anchor_builder import AnchorBuilder
from .cache_utils import GraphCacheSpec, csr_graph_cache_path, load_pickle, save_pickle


def _ensure_csr_rag_importable():
    # 讓 `import CSR_RAG_v4` 在直接跑 evaluation.py 時可用
    home_dir = "/home"
    if home_dir not in sys.path:
        sys.path.insert(0, home_dir)


def _llama_docs_to_langchain_docs(docs: Sequence[Any]) -> List[Any]:
    """
    CSR_RAG_v4 使用 langchain_core.documents.Document；這裡做最小轉接。
    """
    try:
        from langchain_core.documents import Document as LCDocument
    except Exception as e:
        raise ImportError("需要安裝 langchain_core 才能使用 CSR graph methods") from e

    out: List[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        text = getattr(d, "text", None)
        if text is None:
            # llama_index Document 通常有 get_content()
            if hasattr(d, "get_content"):
                text = d.get_content()
            else:
                text = str(d)
        out.append(LCDocument(page_content=str(text or ""), metadata=dict(meta)))
    return out


@dataclass
class _SourceNode:
    text: str
    metadata: Dict[str, Any]

    def get_content(self) -> str:
        return self.text


@dataclass
class _CSRResponse:
    answer: str
    source_nodes: List[_SourceNode]

    def __str__(self) -> str:
        return self.answer


class CSRGraphQueryEngine:
    """
    CSR 的 query engine wrapper：
    - graph cache（只 cache 建圖結果）
    - anchors：query entity 為主，vector docs/doc entities 可選補強
    - expand：khop / bridge
    - 生成：用 Settings.llm 產回答
    """

    def __init__(
        self,
        settings: Any,
        *,
        method_name: str,
        data_mode: str,
        data_type: str,
        documents: Sequence[Any],
        fast_build: bool = False,
        top_k: int = 2,
        use_vector_docs: bool = False,
        doc_entity_mode: str = "metadata_only",
        use_schema_hint: bool = False,
        bridge_strategy: str = "shortest_path",
        max_edges: int = 120,
        k_hop: int = 2,
    ):
        self.settings = settings
        self.method_name = method_name
        self.data_mode = data_mode
        self.data_type = data_type
        self.fast_build = bool(fast_build)
        self.top_k = int(top_k)
        self.bridge_strategy = bridge_strategy
        self.max_edges = int(max_edges)
        self.k_hop = int(k_hop)

        docs = list(documents)
        if self.fast_build:
            docs = docs[:2]
        self._documents = docs

        self.anchor_builder = AnchorBuilder(
            settings,
            use_vector_docs=use_vector_docs,
            doc_entity_mode=doc_entity_mode,
            use_schema_hint=use_schema_hint,
        )
        self._use_vector_docs = bool(use_vector_docs)
        self._doc_entity_mode = doc_entity_mode
        self._use_schema_hint = bool(use_schema_hint)

        self._graph = None
        self._resolve_anchors_from_entity_list = None
        self._vector_index = None

    def _ensure_graph_loaded(self) -> Any:
        if self._graph is not None:
            return self._graph

        spec = GraphCacheSpec(
            method_name=self.method_name,
            data_type=self.data_type,
            data_mode=self.data_mode,
            fast_build=self.fast_build,
        )
        cache_path = csr_graph_cache_path(spec)

        cached = load_pickle(cache_path)
        if cached is not None:
            print(f"✅ [CSRGraph] 從 cache 讀取建圖結果: {cache_path}")
            self._graph = cached
            return self._graph

        print(f"⏳ [CSRGraph] 建立新圖譜（並寫入 cache）: {cache_path}")

        _ensure_csr_rag_importable()
        from CSR_RAG_v4.graph.build import build_graph_from_docs_schema

        lc_docs = _llama_docs_to_langchain_docs(self._documents)
        graph = build_graph_from_docs_schema(lc_docs, llm_client=None, use_llm_extract=False, as_networkx=True)

        save_pickle(cache_path, graph)
        self._graph = graph
        return self._graph

    def _ensure_expand_fns(self):
        if self._resolve_anchors_from_entity_list is not None:
            return
        _ensure_csr_rag_importable()
        from CSR_RAG_v4.graph.expand import (
            graph_expand_k_hop,
            graph_expand_bridge_context_updated,
            resolve_anchors_from_entity_list,
        )

        self._graph_expand_k_hop = graph_expand_k_hop
        self._graph_expand_bridge = graph_expand_bridge_context_updated
        self._resolve_anchors_from_entity_list = resolve_anchors_from_entity_list

    async def aquery(self, user_query: str) -> _CSRResponse:
        graph = self._ensure_graph_loaded()
        self._ensure_expand_fns()

        vector_docs: Optional[Sequence[Any]] = None
        if self._use_vector_docs:
            try:
                from llama_index.core import VectorStoreIndex
                if self._vector_index is None:
                    # 用同一批 documents 建 vector index（一次即可，避免每個 query 重建）
                    self._vector_index = VectorStoreIndex.from_documents(
                        list(self._documents),
                        embed_model=getattr(self.settings, "embed_model", None),
                    )
                retriever = self._vector_index.as_retriever(similarity_top_k=self.top_k)
                nodes = retriever.retrieve(user_query)
                vector_docs = [n.node for n in nodes if getattr(n, "node", None) is not None]
            except Exception as e:
                print(f"⚠️ [CSRGraph] vector docs 取得失敗，將只用 query entities: {e}")
                vector_docs = None

        schema_hint = None
        if self._use_schema_hint:
            # 先以既有 schema 檔作為提示；若不存在就略過
            try:
                import json
                if os.path.exists("/home/End_to_End_RAG/kg_schema.json"):
                    with open("/home/End_to_End_RAG/kg_schema.json", "r", encoding="utf-8") as f:
                        schema_hint = json.load(f)
            except Exception:
                schema_hint = None

        anchors_res = await self.anchor_builder.build(
            query=user_query,
            graph=graph,
            documents_for_vector=vector_docs,
            top_k=self.top_k,
            schema_hint=schema_hint,
            resolve_anchors_fn=self._resolve_anchors_from_entity_list,
        )

        anchor_ids = anchors_res.anchor_node_ids
        if not anchor_ids:
            anchor_ids = anchors_res.vector_doc_ids

        if self.method_name == "csr_khop":
            expanded = self._graph_expand_k_hop(graph, anchor_ids, k=self.k_hop, max_edges=self.max_edges)
        elif self.method_name == "csr_bridge":
            expanded = self._graph_expand_bridge(graph, anchor_ids, max_bridge_hops=2, max_paths_per_pair=5)
        else:
            raise ValueError(f"未知 CSR graph method: {self.method_name}")

        triplets = expanded.get("triplets", []) or []
        # contexts 組裝：以 triplets 為主，並嘗試補上節點內容（若節點帶 content）
        g = getattr(graph, "_graph", graph)
        lines: List[str] = []
        for t in triplets[: self.max_edges]:
            s = str(t.get("source", ""))
            r = str(t.get("relation", "RELATED_TO"))
            o = str(t.get("target", ""))
            lines.append(f"{s} -[{r}]-> {o}")
        context_text = "\n".join(lines[:200])

        prompt = (
            "請根據以下『知識圖譜擴展三元組』回答使用者問題。"
            "若無法從內容推導答案，請回答「不知道」。\n\n"
            f"【使用者問題】\n{user_query}\n\n"
            f"【圖譜三元組】\n{context_text}\n"
        )

        if hasattr(self.settings.llm, "acomplete"):
            llm_res = await self.settings.llm.acomplete(prompt)
            answer = getattr(llm_res, "text", None) or str(llm_res)
        else:
            llm_res = self.settings.llm.complete(prompt)
            answer = getattr(llm_res, "text", None) or str(llm_res)

        # source_nodes：塞入 contexts，並盡量附上 NO 讓 evaluation 可以算 retrieval 指標（若 anchors 是 doc_id）
        source_nodes: List[_SourceNode] = []
        for a in expanded.get("anchors", [])[:10]:
            meta = {"NO": str(a)}
            # 若節點有 content，附上；否則用節點 id
            content = ""
            try:
                if g.has_node(a):
                    content = (g.nodes[a].get("content") or g.nodes[a].get("summary") or "")[:2000]
            except Exception:
                content = ""
            source_nodes.append(_SourceNode(text=content or str(a), metadata=meta))

        if not source_nodes:
            source_nodes = [_SourceNode(text=context_text[:2000], metadata={"NO": ""})]

        return _CSRResponse(answer=answer, source_nodes=source_nodes)

