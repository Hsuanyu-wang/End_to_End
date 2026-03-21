"""
DynamicSchema Builder

包裝 LlamaIndex DynamicLLMPathExtractor 的建圖邏輯
"""

import os
from typing import List, Dict, Any, Optional
from llama_index.core import Document, PropertyGraphIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from src.graph_builder.base_builder import BaseGraphBuilder


class DynamicSchemaBuilder(BaseGraphBuilder):
    """
    DynamicSchema Graph Builder
    
    使用 LlamaIndex 的 DynamicLLMPathExtractor 進行動態 Schema 建圖
    
    Attributes:
        settings: LlamaIndex Settings
        storage_path: 儲存路徑
        max_triplets_per_chunk: 每個 chunk 最多抽取的三元組數量
        num_workers: 並發工作數
    """
    
    def __init__(
        self,
        graph_store: Any = None,
        settings: Any = None,
        data_type: str = "DI",
        fast_test: bool = False,
        max_triplets_per_chunk: int = 20,
        num_workers: int = 4,
        seed_types: Optional[List[str]] = None,
        use_cache: bool = True,
    ):
        """
        初始化 DynamicSchema Builder
        
        Args:
            graph_store: 圖譜儲存後端(可選)
            settings: LlamaIndex Settings
            data_type: 資料類型
            fast_test: 是否快速測試模式
            max_triplets_per_chunk: 每個 chunk 最多抽取的三元組數量
            num_workers: 並發工作數
            seed_types: 種子實體類型列表（引導抽取）
            use_cache: 是否使用 Schema Cache
        """
        super().__init__(graph_store, settings)
        self.data_type = data_type
        self.fast_test = fast_test
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.num_workers = num_workers
        self.storage_path = None
        self.graph_index = None
        
        # 設定種子類型
        if seed_types is None:
            self.seed_types = ["Person", "Organization", "Location", "Event", "System"]
        else:
            self.seed_types = seed_types
        
        self.use_cache = use_cache
        
        print(f"🔧 初始化 DynamicSchema Builder")
        print(f"   - 種子類型: {self.seed_types}")
        print(f"   - Max triplets: {max_triplets_per_chunk}")
        print(f"   - 使用快取: {self.use_cache}")
    
    def get_name(self) -> str:
        return "DynamicSchema"

    @staticmethod
    def _has_complete_llamaindex_cache(persist_dir: str) -> bool:
        """檢查 LlamaIndex persist 目錄是否完整。"""
        required_files = ["docstore.json", "index_store.json"]
        return all(os.path.isfile(os.path.join(persist_dir, f)) for f in required_files)
    
    def build(self, documents: List[Document]) -> Dict[str, Any]:
        """
        使用 DynamicLLMPathExtractor 建立知識圖譜
        
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
            method="dynamic_schema",
            fast_test=self.fast_test
        )
        
        print(f"📂 DynamicSchema 儲存路徑: {self.storage_path}")
        
        # 檢查是否已存在
        if os.path.exists(self.storage_path) and self._has_complete_llamaindex_cache(self.storage_path):
            print(f"✅ DynamicSchema 索引已存在,載入現有索引...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
            self.graph_index = load_index_from_storage(storage_context)
        else:
            if os.path.exists(self.storage_path):
                print(f"⚠️ DynamicSchema 快取不完整，將重建索引: {self.storage_path}")
            print(f"🔨 開始建立 DynamicSchema 索引...")
            
            # 限制文檔數量(如果是快速測試)
            if self.fast_test and len(documents) > 2:
                documents = documents[:2]
                print(f"⚡ 快速測試模式:僅處理前 2 筆文檔")
            
            # 建立 DynamicLLMPathExtractor
            print(f">> 初始化 DynamicLLMPathExtractor...")
            dynamic_extractor = DynamicLLMPathExtractor(
                llm=self.settings.builder_llm,
                max_triplets_per_chunk=self.max_triplets_per_chunk,
                num_workers=self.num_workers,
                allowed_entity_types=self.seed_types,
            )
            
            # 建立 PropertyGraph 索引
            self.graph_index = PropertyGraphIndex.from_documents(
                documents,
                llm=self.settings.builder_llm,
                embed_model=self.settings.embed_model,
                show_progress=True,
                embed_batch_size=2,
                kg_extractors=[dynamic_extractor],
            )
            
            # 持久化
            self.graph_index.storage_context.persist(persist_dir=self.storage_path)
            print(f"✅ DynamicSchema 索引建立完成")
        
        # 提取 schema 資訊(從圖譜中推斷)
        # PropertyGraph 不直接提供 schema,需要從圖譜中統計
        entities = set()
        relations = set()
        
        # 嘗試從 graph_store 提取資訊
        if hasattr(self.graph_index, 'property_graph_store'):
            graph_store = self.graph_index.property_graph_store
            # 這裡可以進一步提取 entity types 和 relation types
            # 具體實作取決於 PropertyGraphStore 的介面
        
        schema_info = {
            "entities": list(entities),
            "relations": list(relations),
            "method": "dynamic_schema",
            "note": "動態推斷,無預定義 Schema"
        }
        
        return {
            "nodes": [],  # PropertyGraph 不直接返回節點列表
            "edges": [],  # PropertyGraph 不直接返回邊列表
            "metadata": {
                "num_documents": len(documents),
                "fast_test": self.fast_test,
                "max_triplets_per_chunk": self.max_triplets_per_chunk
            },
            "schema_info": schema_info,
            "storage_path": self.storage_path,
            "graph_format": "property_graph"
        }
