"""
LightRAG Retriever

分離 LightRAG 的檢索邏輯作為 Retriever 模組
"""

import os
from typing import Dict, Any, Optional, List
from src.graph_retriever.base_retriever import BaseGraphRetriever, GraphData


class LightRAGRetriever(BaseGraphRetriever):
    """
    LightRAG Retriever
    
    提供 LightRAG 的多種檢索模式
    
    Attributes:
        mode: 檢索模式(local/global/hybrid/mix/naive/bypass)
        rag_instance: LightRAG 實例
        settings: LlamaIndex Settings
    """
    
    def __init__(
        self,
        mode: str = "hybrid",
        rag_instance = None,
        settings: Any = None,
        data_type: str = "DI",
        sup: str = "",
        fast_test: bool = False
    ):
        """
        初始化 LightRAG Retriever
        
        Args:
            mode: 檢索模式
            rag_instance: LightRAG 實例(如果已有)
            settings: LlamaIndex Settings
            data_type: 資料類型
            sup: 附加標籤
            fast_test: 是否快速測試模式
        """
        self.mode = mode
        self.rag_instance = rag_instance
        self.settings = settings
        self.data_type = data_type
        self.sup = sup
        self.fast_test = fast_test
    
    def get_name(self) -> str:
        return f"LightRAG_{self.mode.capitalize()}"
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Retriever,載入 LightRAG 實例
        
        Args:
            config: 配置參數,可包含:
                - storage_path: 儲存路徑
                - schema_method: Schema 方法
        """
        if self.rag_instance is not None:
            return
        
        config = config or {}
        
        # 如果提供了 graph_data,從中取得儲存路徑
        storage_path = config.get('storage_path')
        
        if not storage_path:
            # 自動推斷儲存路徑
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
        
        # 載入 LightRAG 實例
        from src.rag.graph.lightrag import get_lightrag_engine
        
        self.rag_instance = get_lightrag_engine(
            Settings=self.settings,
            data_type=self.data_type,
            sup=self.sup,
            mode=self.mode,
            fast_test=self.fast_test
        )
        
        print(f"✅ LightRAG Retriever 初始化完成 (模式: {self.mode})")
    
    def retrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 LightRAG 檢索
        
        Args:
            query: 使用者查詢
            graph_data: 圖譜資料(包含 storage_path)
            top_k: 檢索數量(LightRAG 內部控制)
            **kwargs: 額外參數
        
        Returns:
            檢索結果字典
        """
        # 如果未初始化,先初始化
        if self.rag_instance is None:
            init_config = {}
            if graph_data and 'storage_path' in graph_data:
                init_config['storage_path'] = graph_data['storage_path']
            if graph_data and 'schema_info' in graph_data:
                init_config['schema_method'] = graph_data['schema_info'].get('method', 'lightrag_default')
            
            self.initialize(init_config)
        
        print(f"🔍 [LightRAG-{self.mode}] 開始檢索...")
        
        try:
            # 使用 LightRAG 的 query 方法
            from lightrag.lightrag import QueryParam
            
            response = self.rag_instance.query(
                query,
                param=QueryParam(mode=self.mode)
            )
            
            # LightRAG 返回的是字串,需要解析成 contexts
            # 簡化處理:將整個回應作為一個 context
            contexts = [str(response)] if response else []
            
        except Exception as e:
            print(f"⚠️  LightRAG 檢索失敗: {e}")
            contexts = []
        
        return {
            "contexts": contexts,
            "nodes": [],
            "metadata": {
                "mode": self.mode,
                "retriever": "lightrag"
            }
        }
