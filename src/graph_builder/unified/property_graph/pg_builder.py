"""
PropertyGraph Builder

使用 LlamaIndex PropertyGraphIndex 的建圖實作，支援多 extractor 並行
"""

import os
from typing import List, Dict, Any
from llama_index.core import Document
from src.graph_builder.base_builder import BaseGraphBuilder
from .extractor_factory import PropertyGraphExtractorFactory


class PropertyGraphBuilder(BaseGraphBuilder):
    """
    PropertyGraph Builder
    
    使用 LlamaIndex PropertyGraphIndex 與多個 extractors 建圖
    """
    
    def __init__(
        self,
        graph_store: Any = None,
        settings: Any = None,
        data_type: str = "DI",
        fast_test: bool = False,
        extractor_config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        初始化 PropertyGraph Builder
        
        Args:
            graph_store: 圖譜儲存後端(可選)
            settings: LlamaIndex Settings
            data_type: 資料類型
            fast_test: 是否快速測試模式
            extractor_config: Extractor 配置
        """
        super().__init__(graph_store, settings)
        self.data_type = data_type
        self.fast_test = fast_test
        self.extractor_config = extractor_config or {}
        self.storage_path = None
        self.graph_index = None
        
        # 建立 extractors
        self.extractors = PropertyGraphExtractorFactory.create_extractors(
            settings,
            self.extractor_config
        )
        
        print(f"🔧 初始化 PropertyGraphBuilder")
        print(f"   - 資料類型: {data_type}")
        print(f"   - 快速測試: {fast_test}")
        print(f"   - Extractors: {len(self.extractors)}")
    
    def get_name(self) -> str:
        return "PropertyGraph"
    
    def build(self, documents: List[Document]) -> Dict[str, Any]:
        """
        使用 PropertyGraph 與多 extractors 建立知識圖譜
        
        Args:
            documents: 文檔列表
        
        Returns:
            標準化的圖譜資料字典
        """
        try:
            from llama_index.core import PropertyGraphIndex, StorageContext, load_index_from_storage
        except ImportError:
            raise ImportError("需要安裝 llama-index-core: pip install llama-index-core")
        
        from src.storage import get_storage_path
        
        # 取得儲存路徑
        method_name = f"property_graph_{'_'.join([k for k, v in self.extractor_config.items() if (v.get('enabled') if isinstance(v, dict) else v)])}"
        self.storage_path = get_storage_path(
            storage_type="graph_index",
            data_type=self.data_type,
            method=method_name,
            fast_test=self.fast_test
        )
        
        print(f"📂 PropertyGraph 儲存路徑: {self.storage_path}")
        
        # 檢查快取
        if os.path.exists(self.storage_path) and self._has_complete_cache(self.storage_path):
            print(f"✅ PropertyGraph 索引已存在，載入現有索引...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
                self.graph_index = load_index_from_storage(storage_context)
            except Exception as e:
                print(f"⚠️  載入索引失敗: {e}，將重新建圖")
                self.graph_index = None
        
        if self.graph_index is None:
            print(f"🔨 開始建立 PropertyGraph 索引...")
            
            # 快速測試模式
            if self.fast_test and len(documents) > 2:
                documents = documents[:2]
                print(f"⚡ 快速測試模式：僅處理前 2 筆文檔")
            
            # 建立 PropertyGraphIndex
            try:
                llm = getattr(self.settings, 'builder_llm', None) or getattr(self.settings, 'llm', None)
                embed_model = getattr(self.settings, 'embed_model', None)
                
                self.graph_index = PropertyGraphIndex.from_documents(
                    documents,
                    property_graph_store=self.graph_store,
                    kg_extractors=self.extractors if self.extractors else None,
                    llm=llm,
                    embed_model=embed_model,
                    show_progress=True,
                    embed_batch_size=2
                )
                
                # 持久化
                self.graph_index.storage_context.persist(persist_dir=self.storage_path)
                print(f"✅ PropertyGraph 索引建立完成")
            
            except Exception as e:
                print(f"❌ PropertyGraph 建圖失敗: {e}")
                raise
        
        # 提取 schema 資訊
        entities = set()
        relations = set()
        
        # 嘗試從配置中提取
        if "schema" in self.extractor_config and isinstance(self.extractor_config["schema"], dict):
            schema_cfg = self.extractor_config["schema"]
            entities.update(schema_cfg.get("entities", []))
            relations.update(schema_cfg.get("relations", []))
        
        schema_info = {
            "entities": list(entities),
            "relations": list(relations),
            "method": "property_graph",
            "extractors": [k for k, v in self.extractor_config.items() if (v.get("enabled") if isinstance(v, dict) else v)]
        }
        
        return {
            "nodes": [],
            "edges": [],
            "metadata": {
                "num_documents": len(documents),
                "fast_test": self.fast_test,
                "extractors": schema_info["extractors"]
            },
            "schema_info": schema_info,
            "storage_path": self.storage_path,
            "graph_format": "property_graph",
            "graph_index": self.graph_index  # 保留原始 index 供後續使用
        }
    
    @staticmethod
    def _has_complete_cache(persist_dir: str) -> bool:
        """檢查 LlamaIndex persist 目錄是否完整"""
        required_files = ["docstore.json", "index_store.json"]
        return all(os.path.isfile(os.path.join(persist_dir, f)) for f in required_files)
