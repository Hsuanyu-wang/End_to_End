"""
Modular Graph Pipeline Wrapper

組合 Graph Builder + Graph Retriever 的模組化 Pipeline。
支援 context_first（自訂 prompt 生成）和 native（原生端到端）兩種生成模式。
"""

from typing import Dict, Any, Optional, List
import networkx as nx
from src.rag.wrappers.base_wrapper import BaseRAGWrapper
from src.graph_builder.base_builder import BaseGraphBuilder
from src.graph_retriever.base_retriever import BaseGraphRetriever, GraphData
from src.utils.token_counter import get_token_counter
from llama_index.core import Document


class ModularGraphWrapper(BaseRAGWrapper):
    """
    模組化 Graph Pipeline Wrapper

    組合 Builder + Retriever 實現靈活的 Graph RAG Pipeline。
    支援自動格式轉換（透過 GraphFormatAdapter）。

    generation_mode:
        - "context_first": 以 Retriever 取 contexts，再由自訂 prompt + LLM 生成答案
        - "native": 由 Retriever 直接生成答案（如 LightRAG 官方 aquery）
    """

    def __init__(
        self,
        name: str,
        builder: BaseGraphBuilder,
        retriever: BaseGraphRetriever,
        documents: list = None,
        model_type: str = "small",
        schema_info: Dict[str, Any] = None,
        top_k: int = 2,
        enable_format_conversion: bool = True,
        generation_mode: str = "context_first",
    ):
        super().__init__(name, schema_info=schema_info)
        self.builder = builder
        self.retriever = retriever
        self.model_type = model_type
        self.top_k = top_k
        self.graph_data = None
        self.enable_format_conversion = enable_format_conversion
        self.generation_mode = generation_mode
        self._graph_quality: Dict[str, Any] = {}

        if self.enable_format_conversion:
            from src.graph_adapter import GraphFormatAdapter
            self.adapter = GraphFormatAdapter()
        else:
            self.adapter = None

        if documents:
            self._build_graph(documents)

    def _build_graph(self, documents: list):
        """使用 Builder 建立圖譜"""
        print(f"🔨 [{self.builder.get_name()}] 開始建立圖譜...")

        if documents and not isinstance(documents[0], Document):
            documents = [Document(text=str(doc)) for doc in documents]

        graph_dict = self.builder.build(documents)

        if self.enable_format_conversion and isinstance(graph_dict.get("graph_data"), nx.Graph):
            graph_dict = self._ensure_compatible_format(graph_dict, self.retriever)

        self.graph_data = GraphData(
            nodes=graph_dict.get("nodes", []),
            edges=graph_dict.get("edges", []),
            metadata=graph_dict.get("metadata", {}),
            schema_info=graph_dict.get("schema_info", {}),
            storage_path=graph_dict.get("storage_path"),
            graph_format=graph_dict.get("graph_format", "custom"),
        )

        if self.graph_data.schema_info:
            self.schema_info = self.graph_data.schema_info

        print(f"✅ [{self.builder.get_name()}] 圖譜建立完成")
        print(f"📊 節點數: {len(self.graph_data.nodes)}, 邊數: {len(self.graph_data.edges)}")

    def _ensure_compatible_format(
        self, graph_data: Dict[str, Any], retriever: BaseGraphRetriever
    ) -> Dict[str, Any]:
        """自動轉換圖譜格式以匹配 retriever 需求"""
        if not self.enable_format_conversion or not self.adapter:
            return graph_data

        retriever_format = getattr(retriever, "required_format", None)
        current_format = graph_data.get("graph_format", "networkx")

        if retriever_format is None or retriever_format == current_format:
            return graph_data

        print(f"🔄 轉換圖譜格式: {current_format} → {retriever_format}")

        nx_graph = graph_data.get("graph_data")
        if not isinstance(nx_graph, nx.Graph):
            print("⚠️  無法轉換：graph_data 不是 NetworkX graph")
            return graph_data

        try:
            if retriever_format == "property_graph":
                from src.config.settings import get_settings
                settings = get_settings(model_type=self.model_type)
                converted = self.adapter.networkx_to_pg(nx_graph, settings)
                graph_data["graph_data"] = converted
                graph_data["graph_format"] = "property_graph"
            elif retriever_format == "lightrag":
                from src.config.settings import get_settings
                settings = get_settings(model_type=self.model_type)
                converted = self.adapter.networkx_to_lightrag(nx_graph, settings)
                graph_data["graph_data"] = converted
                graph_data["graph_format"] = "lightrag"
            elif retriever_format == "networkx":
                pass
            else:
                print(f"⚠️  不支援的目標格式: {retriever_format}")
        except Exception as e:
            print(f"⚠️  格式轉換失敗: {e}，將使用原始格式繼續")

        return graph_data

    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """執行模組化查詢（根據 generation_mode 分派）"""
        if self.graph_data is None:
            raise ValueError("圖譜尚未建立，請先呼叫 _build_graph() 或在初始化時提供 documents")

        print(f"🔍 [{self.retriever.get_name()}] 開始檢索...")

        try:
            if hasattr(self.retriever, "aretrieve"):
                retrieval_result = await self.retriever.aretrieve(
                    query=query,
                    graph_data=self.graph_data.to_dict() if self.graph_data else None,
                    top_k=self.top_k,
                )
            else:
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    pass
                retrieval_result = self.retriever.retrieve(
                    query=query,
                    graph_data=self.graph_data.to_dict() if self.graph_data else None,
                    top_k=self.top_k,
                )
        except Exception as e:
            import traceback
            print("❌ Retriever 錯誤詳情:")
            traceback.print_exc()
            raise

        contexts = retrieval_result.get("contexts", [])
        nodes = retrieval_result.get("nodes", [])

        contexts = self._truncate_contexts_by_tokens(contexts)

        from src.config.settings import get_settings
        settings = get_settings(model_type=self.model_type)

        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""基於以下檢索到的上下文資訊回答問題:

{context_str}

問題: {query}

請直接回答問題,使用繁體中文。"""

        response = settings.llm.complete(prompt)
        generated_answer = str(response)

        retrieved_ids = retrieval_result.get("retrieved_ids", [])
        if not retrieved_ids and nodes:
            for node in nodes:
                meta = getattr(node, "metadata", None) or {}
                doc_id = meta.get("NO") or meta.get("doc_id") or meta.get("original_no")
                if doc_id:
                    retrieved_ids.append(str(doc_id).strip())
                else:
                    ref = getattr(node, "ref_doc_id", None)
                    if ref:
                        retrieved_ids.append(str(ref).strip())

        # Token 統計
        token_counter = get_token_counter()
        total_tokens = token_counter.count_tokens(context_str) if context_str else 0

        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "source_nodes": nodes,
            "context_tokens": total_tokens,
            "context_token_details": {
                "total_tokens": total_tokens,
                "num_chunks": len(contexts),
            },
            "metadata": {
                "builder": self.builder.get_name(),
                "retriever": self.retriever.get_name(),
                "graph_format": self.graph_data.graph_format if self.graph_data else None,
                "generation_mode": self.generation_mode,
                "graph_quality": self._graph_quality,
            },
        }
