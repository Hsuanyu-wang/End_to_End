"""
DynamicSchema End-to-End Wrapper

包裝 LlamaIndex PropertyGraph + DynamicLLMPathExtractor 的端到端 Pipeline
"""

from typing import Dict, Any, Optional
from src.rag.wrappers.base_wrapper import BaseRAGWrapper
from src.graph_builder.dynamic_schema_builder import DynamicSchemaBuilder
from llama_index.core import Document


class DynamicSchemaWrapper(BaseRAGWrapper):
    """
    DynamicSchema 端到端 Wrapper
    
    使用 LlamaIndex PropertyGraph 與 DynamicLLMPathExtractor
    
    Attributes:
        name: Wrapper 名稱
        builder: DynamicSchema Builder 實例
        query_engine: PropertyGraph Query Engine
    """
    
    def __init__(
        self,
        name: str = "DynamicSchema",
        documents: list = None,
        model_type: str = "small",
        data_type: str = "DI",
        fast_test: bool = False,
        schema_info: Dict[str, Any] = None,
        top_k: int = 2,
        settings: Any = None
    ):
        """
        初始化 DynamicSchema Wrapper
        
        Args:
            name: Wrapper 名稱
            documents: 文檔列表
            model_type: 模型類型
            data_type: 資料類型
            fast_test: 是否快速測試
            schema_info: Schema 資訊
            top_k: 檢索數量
            settings: LlamaIndex Settings
        """
        super().__init__(name, schema_info=schema_info)
        self.model_type = model_type
        self.data_type = data_type
        self.fast_test = fast_test
        self.top_k = top_k
        self.query_engine = None
        
        # 取得或建立 Settings
        if settings is None:
            from src.config.settings import get_settings
            settings = get_settings(model_type=model_type)
        self.settings = settings
        
        # 建立 Builder
        self.builder = DynamicSchemaBuilder(
            graph_store=None,
            settings=self.settings,
            data_type=data_type,
            fast_test=fast_test
        )
        
        # 如果提供了文檔,立即建圖
        if documents:
            self._build_graph(documents)
    
    def _build_graph(self, documents: list):
        """
        使用 DynamicSchema Builder 建立圖譜
        
        Args:
            documents: 文檔列表
        """
        print(f"🔨 [DynamicSchema] 開始建立知識圖譜...")
        
        # 轉換為 LlamaIndex Document 格式
        if documents and not isinstance(documents[0], Document):
            documents = [Document(text=str(doc)) for doc in documents]
        
        # 建立圖譜
        graph_data = self.builder.build(documents)
        
        # 更新 schema_info
        if graph_data.get("schema_info"):
            self.schema_info = graph_data["schema_info"]
        
        # 建立 Query Engine
        if self.builder.graph_index:
            self.query_engine = self.builder.graph_index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
                similarity_top_k=self.top_k
            )
            print(f"✅ [DynamicSchema] Query Engine 建立完成")
        
        print(f"✅ [DynamicSchema] 知識圖譜建立完成")
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        執行 DynamicSchema 查詢
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果字典
        """
        if self.query_engine is None:
            return {
                "generated_answer": "Query Engine 尚未建立",
                "retrieved_contexts": [],
                "retrieved_ids": [],
                "source_nodes": []
            }
        
        print(f"🔍 [DynamicSchema] 開始檢索...")
        
        try:
            # 使用 PropertyGraph Query Engine 查詢
            response = self.query_engine.query(query)
            
            # 提取資訊
            generated_answer = str(response)
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            
            # 提取 contexts
            contexts = []
            retrieved_ids = []
            
            for node in source_nodes:
                if hasattr(node, 'text'):
                    contexts.append(node.text)
                elif hasattr(node, 'node') and hasattr(node.node, 'text'):
                    contexts.append(node.node.text)
                
                # 提取 node ID
                if hasattr(node, 'node_id'):
                    retrieved_ids.append(node.node_id)
                elif hasattr(node, 'node') and hasattr(node.node, 'node_id'):
                    retrieved_ids.append(node.node.node_id)
            
            # 應用 retrieval token 限制
            contexts = self._truncate_contexts_by_tokens(contexts)
            
            return {
                "generated_answer": generated_answer,
                "retrieved_contexts": contexts,
                "retrieved_ids": retrieved_ids,
                "source_nodes": source_nodes,
                "metadata": {
                    "method": "dynamic_schema",
                    "num_contexts": len(contexts)
                }
            }
            
        except Exception as e:
            print(f"⚠️  DynamicSchema 查詢失敗: {e}")
            return {
                "generated_answer": f"查詢失敗: {str(e)}",
                "retrieved_contexts": [],
                "retrieved_ids": [],
                "source_nodes": []
            }
