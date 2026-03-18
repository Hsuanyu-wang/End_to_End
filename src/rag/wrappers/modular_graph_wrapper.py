"""
Modular Graph Pipeline Wrapper

組合 Graph Builder + Graph Retriever 的模組化 Pipeline
"""

from typing import Dict, Any, Optional
from src.rag.wrappers.base_wrapper import BaseRAGWrapper
from src.graph_builder.base_builder import BaseGraphBuilder
from src.graph_retriever.base_retriever import BaseGraphRetriever, GraphData
from llama_index.core import Document


class ModularGraphWrapper(BaseRAGWrapper):
    """
    模組化 Graph Pipeline Wrapper
    
    組合 Builder + Retriever 實現靈活的 Graph RAG Pipeline
    
    Attributes:
        name: Wrapper 名稱
        builder: Graph Builder 實例
        retriever: Graph Retriever 實例
        graph_data: 建立的圖譜資料
        model_type: 模型類型
    """
    
    def __init__(
        self,
        name: str,
        builder: BaseGraphBuilder,
        retriever: BaseGraphRetriever,
        documents: list = None,
        model_type: str = "small",
        schema_info: Dict[str, Any] = None,
        top_k: int = 2
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
        """
        super().__init__(name, schema_info=schema_info)
        self.builder = builder
        self.retriever = retriever
        self.model_type = model_type
        self.top_k = top_k
        self.graph_data = None
        
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
        
        # 使用 Retriever 檢索
        retrieval_result = self.retriever.retrieve(
            query=query,
            graph_data=self.graph_data.to_dict() if self.graph_data else None,
            top_k=self.top_k
        )
        
        # 提取 contexts
        contexts = retrieval_result.get("contexts", [])
        nodes = retrieval_result.get("nodes", [])
        
        # 應用 retrieval token 限制
        contexts = self._truncate_contexts_by_tokens(contexts)
        
        # 使用 LLM 生成答案
        from src.config.settings import get_settings
        Settings = get_settings(model_type=self.model_type)
        
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""基於以下檢索到的上下文資訊回答問題:

{context_str}

問題: {query}

請直接回答問題,使用繁體中文。"""
        
        # 不限制生成 token
        response = Settings.llm.complete(prompt)
        generated_answer = str(response)
        
        # 提取 chunk IDs
        retrieved_ids = []
        if nodes:
            for node in nodes:
                if hasattr(node, 'node_id'):
                    retrieved_ids.append(node.node_id)
                elif hasattr(node, 'id_'):
                    retrieved_ids.append(node.id_)
        
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
