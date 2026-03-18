# graph_builder/baseline_builder.py
import os
from typing import List, Dict, Any
from llama_index.core import Document, PropertyGraphIndex, StorageContext, load_index_from_storage
from .base_builder import BaseGraphBuilder

class BaselineGraphBuilder(BaseGraphBuilder):
    """
    Baseline Graph Builder
    
    使用 LlamaIndex PropertyGraph 的預設建圖邏輯
    支援持久化與 cache 檢查
    """
    
    def __init__(
        self,
        graph_store: Any = None,
        settings: Any = None,
        data_type: str = "DI",
        fast_test: bool = False
    ):
        """
        初始化 Baseline Builder
        
        Args:
            graph_store: 圖譜儲存後端(可選)
            settings: 配置設定(可選)
            data_type: 資料類型
            fast_test: 是否為快速測試模式
        """
        super().__init__(graph_store, settings)
        self.data_type = data_type
        self.fast_test = fast_test
        self.storage_path = None
        self.graph_index = None
    
    def get_name(self) -> str:
        return "Baseline"
    
    def build(self, documents: List[Document]) -> Dict[str, Any]:
        """
        使用 PropertyGraph 預設邏輯建立知識圖譜
        
        Args:
            documents: 文檔列表
            
        Returns:
            標準化的圖譜資料字典
        """
        from src.storage import get_storage_path
        
        # 取得儲存路徑
        self.storage_path = get_storage_path(
            storage_type="graph_index",
            data_type=self.data_type,
            method="baseline",
            fast_test=self.fast_test
        )
        
        print(f"📂 Baseline 儲存路徑: {self.storage_path}")
        
        # 檢查是否已存在
        if os.path.exists(self.storage_path):
            print(f"✅ Baseline 索引已存在,載入現有索引...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
            self.graph_index = load_index_from_storage(storage_context)
        else:
            print(f"🔨 [Baseline] 開始建立 PropertyGraph 索引...")
            
            # 限制文檔數量(如果是快速測試)
            if self.fast_test and len(documents) > 2:
                documents = documents[:2]
                print(f"⚡ 快速測試模式:僅處理前 2 筆文檔")
            
            # 建立 PropertyGraph 索引(使用預設 extractor)
            self.graph_index = PropertyGraphIndex.from_documents(
                documents,
                property_graph_store=self.graph_store,
                llm=self.settings.llm,
                embed_model=self.settings.embed_model,
                show_progress=True
            )
            
            # 持久化
            self.graph_index.storage_context.persist(persist_dir=self.storage_path)
            print(f"✅ [Baseline] 索引建立完成")
        
        # 提取 schema 資訊(PropertyGraph 不直接提供)
        entities = set()
        relations = set()
        
        # 可選:從 graph_store 提取資訊
        if hasattr(self.graph_index, 'property_graph_store'):
            graph_store = self.graph_index.property_graph_store
            # 具體實作取決於 PropertyGraphStore 的介面
        
        schema_info = {
            "entities": list(entities),
            "relations": list(relations),
            "method": "baseline",
            "note": "PropertyGraph 預設建圖邏輯"
        }
        
        return {
            "nodes": [],  # PropertyGraph 不直接返回節點列表
            "edges": [],  # PropertyGraph 不直接返回邊列表
            "metadata": {
                "num_documents": len(documents),
                "fast_test": self.fast_test,
                "cached": os.path.exists(self.storage_path)
            },
            "schema_info": schema_info,
            "storage_path": self.storage_path,
            "graph_format": "property_graph"
        }
