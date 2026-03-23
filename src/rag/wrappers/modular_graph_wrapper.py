"""
Modular Graph Pipeline Wrapper

組合 Graph Builder + Graph Retriever 的模組化 Pipeline
"""

from typing import Dict, Any, Optional
import networkx as nx
from src.rag.wrappers.base_wrapper import BaseRAGWrapper
from src.graph_builder.base_builder import BaseGraphBuilder
from src.graph_retriever.base_retriever import BaseGraphRetriever, GraphData
from llama_index.core import Document


class ModularGraphWrapper(BaseRAGWrapper):
    """
    模組化 Graph Pipeline Wrapper
    
    組合 Builder + Retriever 實現靈活的 Graph RAG Pipeline
    支援自動格式轉換（透過 GraphFormatAdapter）
    
    Attributes:
        name: Wrapper 名稱
        builder: Graph Builder 實例
        retriever: Graph Retriever 實例
        graph_data: 建立的圖譜資料
        model_type: 模型類型
        enable_format_conversion: 是否啟用自動格式轉換
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
        enable_format_conversion: bool = True
    ):
        """
        初始化模組化 Graph Pipeline
        
        Args:
            name: Wrapper 名稱
            builder: Graph Builder 實例
            retriever: Graph Retriever 實例
            documents: 文檔列表(如需要立即建圖)
            model_type: 模型類型
            schema_info: Schema 資訊(可選,會被 builder 輸出覆蓋)
            top_k: 檢索數量
            enable_format_conversion: 是否啟用自動格式轉換（預設 True）
        """
        super().__init__(name, schema_info=schema_info)
        self.builder = builder
        self.retriever = retriever
        self.model_type = model_type
        self.top_k = top_k
        self.graph_data = None
        self.enable_format_conversion = enable_format_conversion
        
        # 初始化 GraphFormatAdapter（如果啟用）
        if self.enable_format_conversion:
            from src.graph_adapter import GraphFormatAdapter
            self.adapter = GraphFormatAdapter()
        else:
            self.adapter = None
        
        # 如果提供了文檔,立即建圖
        if documents:
            self._build_graph(documents)
    
    def _build_graph(self, documents: list):
        """
        使用 Builder 建立圖譜
        
        Args:
            documents: 文檔列表
        """
        print(f"🔨 [{self.builder.get_name()}] 開始建立圖譜...")
        
        # 轉換為 LlamaIndex Document 格式(如果需要)
        if documents and not isinstance(documents[0], Document):
            documents = [Document(text=str(doc)) for doc in documents]
        
        # 建立圖譜
        graph_dict = self.builder.build(documents)
        
        # 如果啟用格式轉換且 builder 輸出的是 NetworkX
        if self.enable_format_conversion and isinstance(graph_dict.get("graph_data"), nx.Graph):
            # 自動轉換為 retriever 需要的格式
            graph_dict = self._ensure_compatible_format(graph_dict, self.retriever)
        
        # 轉換為 GraphData 物件
        self.graph_data = GraphData(
            nodes=graph_dict.get("nodes", []),
            edges=graph_dict.get("edges", []),
            metadata=graph_dict.get("metadata", {}),
            schema_info=graph_dict.get("schema_info", {}),
            storage_path=graph_dict.get("storage_path"),
            graph_format=graph_dict.get("graph_format", "custom")
        )
        
        # 更新 schema_info
        if self.graph_data.schema_info:
            self.schema_info = self.graph_data.schema_info
        
        print(f"✅ [{self.builder.get_name()}] 圖譜建立完成")
        print(f"📊 節點數: {len(self.graph_data.nodes)}, 邊數: {len(self.graph_data.edges)}")
    
    def _ensure_compatible_format(self, graph_data: Dict[str, Any], retriever: BaseGraphRetriever) -> Dict[str, Any]:
        """
        自動轉換圖譜格式以匹配 retriever 需求
        
        Args:
            graph_data: Builder 輸出的圖譜資料
            retriever: Retriever 實例
        
        Returns:
            轉換後的圖譜資料
        """
        if not self.enable_format_conversion or not self.adapter:
            return graph_data
        
        # 取得 retriever 需要的格式
        retriever_format = getattr(retriever, 'required_format', None)
        current_format = graph_data.get("graph_format", "networkx")
        
        # 如果格式已匹配，直接返回
        if retriever_format is None or retriever_format == current_format:
            return graph_data
        
        print(f"🔄 轉換圖譜格式: {current_format} → {retriever_format}")
        
        # 取得 NetworkX graph
        nx_graph = graph_data.get("graph_data")
        if not isinstance(nx_graph, nx.Graph):
            print(f"⚠️  無法轉換：graph_data 不是 NetworkX graph")
            return graph_data
        
        try:
            # 根據 retriever 需要的格式進行轉換
            if retriever_format == "property_graph":
                from src.config.settings import get_settings
                settings = get_settings(model_type=self.model_type)
                converted = self.adapter.networkx_to_pg(nx_graph, settings)
                graph_data["graph_data"] = converted
                graph_data["graph_format"] = "property_graph"
                print(f"✅ 已轉換為 PropertyGraph 格式")
            
            elif retriever_format == "lightrag":
                from src.config.settings import get_settings
                settings = get_settings(model_type=self.model_type)
                converted = self.adapter.networkx_to_lightrag(nx_graph, settings)
                graph_data["graph_data"] = converted
                graph_data["graph_format"] = "lightrag"
                print(f"✅ 已轉換為 LightRAG 格式")
            
            elif retriever_format == "networkx":
                # 已經是 NetworkX，不需轉換
                pass
            
            else:
                print(f"⚠️  不支援的目標格式: {retriever_format}")
        
        except Exception as e:
            print(f"⚠️  格式轉換失敗: {e}")
            print(f"   將使用原始格式繼續")
        
        return graph_data
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        執行模組化查詢
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果字典
        """
        if self.graph_data is None:
            raise ValueError("圖譜尚未建立,請先呼叫 _build_graph() 或在初始化時提供 documents")
        
        print(f"🔍 [{self.retriever.get_name()}] 開始檢索...")
        
        # 使用 Retriever 檢索（優先使用異步方法）
        try:
            if hasattr(self.retriever, 'aretrieve'):
                retrieval_result = await self.retriever.aretrieve(
                    query=query,
                    graph_data=self.graph_data.to_dict() if self.graph_data else None,
                    top_k=self.top_k
                )
            else:
                # 如果沒有異步方法，使用同步方法但需要處理 nested async
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    pass
                
                retrieval_result = self.retriever.retrieve(
                    query=query,
                    graph_data=self.graph_data.to_dict() if self.graph_data else None,
                    top_k=self.top_k
                )
        except Exception as e:
            import traceback
            print(f"❌ Retriever 錯誤詳情:")
            traceback.print_exc()
            raise
        
        # 提取 contexts
        contexts = retrieval_result.get("contexts", [])
        nodes = retrieval_result.get("nodes", [])
        
        # 應用 retrieval token 限制
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
        
        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "source_nodes": nodes,
            "metadata": {
                "builder": self.builder.get_name(),
                "retriever": self.retriever.get_name(),
                "graph_format": self.graph_data.graph_format if self.graph_data else None
            }
        }
