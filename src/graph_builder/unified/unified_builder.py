"""
Unified Graph Builder

統一的 Graph Builder Wrapper，透過 Registry 系統動態載入不同的 Builder 實作
"""

from typing import List, Dict, Any
from llama_index.core import Document
from src.graph_builder.base_builder import BaseGraphBuilder
from .builder_registry import GraphBuilderRegistry


class UnifiedGraphBuilder(BaseGraphBuilder):
    """
    統一的 Graph Builder Wrapper
    
    透過 Registry 系統動態載入不同的 Builder 實作
    統一輸出 NetworkX 格式
    """
    
    def __init__(
        self,
        settings: Any,
        builder_type: str,
        builder_config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        初始化 Unified Builder
        
        Args:
            settings: LlamaIndex Settings
            builder_type: Builder 類型（從 Registry 中選擇）
            builder_config: Builder 配置
            **kwargs: 額外參數傳遞給實際 builder
        """
        super().__init__(settings=settings)
        self.builder_type = builder_type
        self.builder_config = builder_config or {}
        
        # 特殊處理：PropertyGraph 需要 extractor_config
        if builder_type == "property_graph":
            # 先註冊 PropertyGraph Builder（如果尚未註冊）
            if not GraphBuilderRegistry.is_registered("property_graph"):
                from .property_graph import PropertyGraphBuilder
                GraphBuilderRegistry.register("property_graph", PropertyGraphBuilder)
            
            # 確保有 extractor_config
            if "extractor_config" not in kwargs and "extractors" in self.builder_config:
                kwargs["extractor_config"] = self.builder_config.get("extractors", {})
        
        # 從 Registry 建立實際的 builder
        try:
            self.actual_builder = GraphBuilderRegistry.create(
                builder_type,
                settings=settings,
                **self.builder_config,
                **kwargs
            )
            print(f"🔧 UnifiedGraphBuilder 使用: {builder_type}")
        except Exception as e:
            print(f"❌ 建立 {builder_type} builder 失敗: {e}")
            raise
    
    def get_name(self) -> str:
        return f"Unified[{self.builder_type}]"
    
    def build(self, documents: List[Document]) -> Dict[str, Any]:
        """
        建圖並統一轉換為 NetworkX 格式
        
        Args:
            documents: 文檔列表
        
        Returns:
            標準化輸出:
            {
                "graph_data": NetworkX graph (統一格式),
                "original_output": 原始 builder 的輸出,
                "graph_format": "networkx",
                "source_format": 原始格式名稱,
                ...
            }
        """
        print(f"🔨 [{self.builder_type}] 開始建圖...")
        
        # 呼叫實際 builder 的 build 方法
        try:
            original_output = self.actual_builder.build(documents)
        except Exception as e:
            print(f"❌ [{self.builder_type}] 建圖失敗: {e}")
            raise
        
        source_format = original_output.get("graph_format", self.builder_type)
        
        # 轉換為 NetworkX（透過 GraphFormatAdapter）
        from src.graph_adapter import GraphFormatAdapter
        
        try:
            nx_graph = GraphFormatAdapter.to_networkx(
                original_output,
                source_format=source_format
            )
            
            print(f"✅ [{self.builder_type}] 建圖完成，已轉為 NetworkX")
            print(f"   節點數: {nx_graph.number_of_nodes()}, 邊數: {nx_graph.number_of_edges()}")
        
        except Exception as e:
            print(f"⚠️  [{self.builder_type}] 轉換為 NetworkX 失敗: {e}")
            print("   將返回原始輸出")
            # 如果轉換失敗，至少返回原始輸出
            return {
                "graph_data": None,
                "original_output": original_output,
                "graph_format": source_format,
                "source_format": source_format,
                "metadata": original_output.get("metadata", {}),
                "schema_info": original_output.get("schema_info", {}),
                "storage_path": original_output.get("storage_path"),
                "graph_index": original_output.get("graph_index"),  # 保留原始 index（PropertyGraph 需要）
                "conversion_error": str(e)
            }
        
        return {
            "graph_data": nx_graph,
            "original_output": original_output,
            "graph_format": "networkx",
            "source_format": source_format,
            "metadata": original_output.get("metadata", {}),
            "schema_info": original_output.get("schema_info", {}),
            "storage_path": original_output.get("storage_path"),
            "graph_index": original_output.get("graph_index"),  # 保留原始 index（PropertyGraph 需要）
        }
    
    async def abuild(self, documents: List[Document]) -> Dict[str, Any]:
        """
        非同步建圖（如果 builder 支援）
        
        Args:
            documents: 文檔列表
        
        Returns:
            標準化輸出（同 build）
        """
        print(f"🔨 [{self.builder_type}] 開始非同步建圖...")
        
        # 檢查 builder 是否支援非同步
        if hasattr(self.actual_builder, 'abuild'):
            original_output = await self.actual_builder.abuild(documents)
        else:
            print(f"⚠️  {self.builder_type} 不支援非同步建圖，使用同步方法")
            original_output = self.actual_builder.build(documents)
        
        # 轉換為 NetworkX
        from src.graph_adapter import GraphFormatAdapter
        source_format = original_output.get("graph_format", self.builder_type)
        
        try:
            nx_graph = GraphFormatAdapter.to_networkx(original_output, source_format)
            
            print(f"✅ [{self.builder_type}] 非同步建圖完成，已轉為 NetworkX")
            print(f"   節點數: {nx_graph.number_of_nodes()}, 邊數: {nx_graph.number_of_edges()}")
        
        except Exception as e:
            print(f"⚠️  轉換為 NetworkX 失敗: {e}")
            nx_graph = None
        
        return {
            "graph_data": nx_graph,
            "original_output": original_output,
            "graph_format": "networkx",
            "source_format": source_format,
            "metadata": original_output.get("metadata", {}),
            "schema_info": original_output.get("schema_info", {}),
            "storage_path": original_output.get("storage_path"),
            "graph_index": original_output.get("graph_index"),  # 保留原始 index（PropertyGraph 需要）
        }
