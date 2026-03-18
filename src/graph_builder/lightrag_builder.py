"""
LightRAG Builder

包裝 LightRAG 的建圖邏輯作為 Builder 模組
"""

import os
from typing import List, Dict, Any
from llama_index.core import Document
from src.graph_builder.base_builder import BaseGraphBuilder


class LightRAGBuilder(BaseGraphBuilder):
    """
    LightRAG Graph Builder
    
    使用 LightRAG 進行圖譜建構
    
    Attributes:
        settings: LlamaIndex Settings
        data_type: 資料類型
        schema_method: Schema 生成方法
        storage_path: 儲存路徑
    """
    
    def __init__(
        self,
        graph_store: Any = None,
        settings: Any = None,
        data_type: str = "DI",
        schema_method: str = "lightrag_default",
        sup: str = "",
        fast_test: bool = False
    ):
        """
        初始化 LightRAG Builder
        
        Args:
            graph_store: 圖譜儲存後端(可選)
            settings: LlamaIndex Settings
            data_type: 資料類型
            schema_method: Schema 生成方法
            sup: 附加標籤
            fast_test: 是否快速測試模式
        """
        super().__init__(graph_store, settings)
        self.data_type = data_type
        self.schema_method = schema_method
        self.sup = sup
        self.fast_test = fast_test
        self.storage_path = None
        self.lightrag_instance = None
    
    def get_name(self) -> str:
        return "LightRAG"
    
    def build(self, documents: List[Document]) -> Dict[str, Any]:
        """
        使用 LightRAG 建立知識圖譜
        
        Args:
            documents: 文檔列表
            
        Returns:
            標準化的圖譜資料字典
        """
        from src.rag.graph.lightrag import build_lightrag_index, get_lightrag_engine
        from src.storage import get_storage_path
        
        # 取得儲存路徑
        full_sup = f"{self.sup}_{self.schema_method}" if self.sup else self.schema_method
        self.storage_path = get_storage_path(
            storage_type="lightrag",
            data_type=self.data_type,
            method=full_sup,
            fast_test=self.fast_test
        )
        
        print(f"📂 LightRAG 儲存路徑: {self.storage_path}")
        
        # 檢查是否已存在
        if os.path.exists(self.storage_path) and os.listdir(self.storage_path):
            print(f"✅ LightRAG 索引已存在,跳過建圖")
        else:
            print(f"🔨 開始建立 LightRAG 索引...")
            
            # 準備文本列表
            texts = []
            for doc in documents:
                if hasattr(doc, 'text'):
                    texts.append(doc.text)
                else:
                    texts.append(str(doc))
            
            # 限制數量(如果是快速測試)
            if self.fast_test and len(texts) > 2:
                texts = texts[:2]
                print(f"⚡ 快速測試模式:僅處理前 2 筆文本")
            
            # 建立索引
            try:
                # 注意:這裡假設 entity_types 已經在 Settings 中設定好
                from src.rag.graph.lightrag import build_lightrag_index
                
                build_lightrag_index(
                    Settings=self.settings,
                    mode="unstructured_text",
                    data_type=self.data_type,
                    sup=full_sup,
                    fast_build=self.fast_test
                )
                print(f"✅ LightRAG 索引建立完成")
            except Exception as e:
                print(f"⚠️  LightRAG 建圖失敗: {e}")
                raise
        
        # 取得 LightRAG 實例
        from src.rag.graph.lightrag import get_lightrag_engine
        self.lightrag_instance = get_lightrag_engine(
            Settings=self.settings,
            data_type=self.data_type,
            sup=full_sup,
            fast_test=self.fast_test
        )
        
        # 提取 schema 資訊
        schema_info = {
            "entities": getattr(self.settings, 'lightrag_entity_types', []),
            "relations": [],
            "method": self.schema_method
        }
        
        return {
            "nodes": [],  # LightRAG 不直接返回節點列表
            "edges": [],  # LightRAG 不直接返回邊列表
            "metadata": {
                "num_documents": len(documents),
                "schema_method": self.schema_method,
                "fast_test": self.fast_test
            },
            "schema_info": schema_info,
            "storage_path": self.storage_path,
            "graph_format": "lightrag"
        }
