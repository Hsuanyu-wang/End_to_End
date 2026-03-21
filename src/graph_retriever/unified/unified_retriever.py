"""
Unified Graph Retriever

統一的 Graph Retriever Wrapper，透過 Registry 系統動態載入不同的 Retriever 實作
"""

from typing import Dict, Any, Optional
import networkx as nx
from src.graph_retriever.base_retriever import BaseGraphRetriever
from .retriever_registry import GraphRetrieverRegistry


class UnifiedGraphRetriever(BaseGraphRetriever):
    """
    統一的 Graph Retriever Wrapper
    
    透過 Registry 系統動態載入不同的 Retriever 實作
    統一接受 NetworkX 輸入（自動轉換為 retriever 需要的格式）
    """
    
    def __init__(
        self,
        graph_source: Any,
        settings: Any,
        retriever_type: str,
        retriever_config: Dict[str, Any] = None,
        combination_mode: str = "ensemble",
        **kwargs
    ):
        """
        初始化 Unified Retriever
        
        Args:
            graph_source: 圖譜來源（NetworkX graph、dict 或原始格式）
            settings: LlamaIndex Settings
            retriever_type: Retriever 類型（從 Registry 中選擇）
            retriever_config: Retriever 配置
            combination_mode: 組合模式（僅 PropertyGraph 使用）
            **kwargs: 額外參數傳遞給實際 retriever
        """
        self.settings = settings
        self.retriever_type = retriever_type
        self.retriever_config = retriever_config or {}
        self.graph_source = graph_source
        self.combination_mode = combination_mode
        
        # 自動轉換格式（如果需要）
        self._prepare_graph_source()
        
        # 從 Registry 建立實際的 retriever
        try:
            # PropertyGraph 需要特殊處理
            if retriever_type == "property_graph":
                self._create_property_graph_retriever(**kwargs)
            else:
                self.actual_retriever = GraphRetrieverRegistry.create(
                    retriever_type,
                    graph_source=self.converted_source,
                    settings=settings,
                    **self.retriever_config,
                    **kwargs
                )
            
            print(f"🔧 UnifiedGraphRetriever 使用: {retriever_type}")
        
        except Exception as e:
            print(f"❌ 建立 {retriever_type} retriever 失敗: {e}")
            raise
    
    def _prepare_graph_source(self):
        """
        根據 retriever_type 自動轉換圖譜格式
        
        - PropertyGraph retriever 需要 PropertyGraphIndex
        - LightRAG retriever 需要 LightRAG 實例
        - 其他 retriever 可能需要 NetworkX
        """
        from src.graph_adapter import GraphFormatAdapter
        
        # 如果輸入已經是 NetworkX
        if isinstance(self.graph_source, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            nx_graph = self.graph_source
        # 如果是 dict，嘗試提取 graph_data 或 graph_index
        elif isinstance(self.graph_source, dict):
            # 優先檢查是否已有 PropertyGraphIndex（適用於 PropertyGraph builder）
            if "graph_index" in self.graph_source:
                try:
                    from llama_index.core import PropertyGraphIndex
                    graph_index = self.graph_source["graph_index"]
                    if isinstance(graph_index, PropertyGraphIndex):
                        if self.retriever_type == "property_graph":
                            self.converted_source = graph_index
                            print(f"✅ 直接使用 dict 中的 PropertyGraphIndex")
                            return
                except ImportError:
                    pass
            
            # 嘗試提取 graph_data
            nx_graph = self.graph_source.get("graph_data")
            if not isinstance(nx_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                # 嘗試從 dict 建立 NetworkX
                try:
                    nx_graph = GraphFormatAdapter._dict_to_networkx(self.graph_source)
                except Exception as e:
                    print(f"⚠️  無法從 dict 建立 NetworkX: {e}")
                    nx_graph = None
        else:
            nx_graph = None
        
        # 根據 retriever_type 轉換
        if self.retriever_type == "property_graph":
            # 需要 PropertyGraphIndex
            if nx_graph:
                try:
                    self.converted_source = GraphFormatAdapter.networkx_to_pg(nx_graph, self.settings)
                    print(f"🔄 已轉換為 PropertyGraph 格式")
                except Exception as e:
                    print(f"⚠️  轉換為 PropertyGraph 失敗: {e}")
                    self.converted_source = self.graph_source
            else:
                self.converted_source = self.graph_source
        
        elif self.retriever_type == "lightrag":
            # LightRAG 使用自己的 storage，不需要 NetworkX 轉換。
            # 優先從 builder 回傳的 dict 中取得已建立的 LightRAG 實例。
            if isinstance(self.graph_source, dict):
                lr_inst = self.graph_source.get("lightrag_instance")
                orig = self.graph_source.get("original_output", {})
                if lr_inst is None and isinstance(orig, dict):
                    lr_inst = orig.get("lightrag_instance")
                sp = self.graph_source.get("storage_path")
                if sp is None and isinstance(orig, dict):
                    sp = orig.get("storage_path")
                self.converted_source = {
                    "lightrag_instance": lr_inst,
                    "storage_path": sp,
                }
                if lr_inst is not None:
                    print(f"✅ 直接使用 Builder 回傳的 LightRAG 實例")
                else:
                    print(f"📂 將從 storage_path 載入 LightRAG: {sp}")
            else:
                self.converted_source = self.graph_source
        
        else:
            # 其他 retriever 預設使用 NetworkX 或原始格式
            self.converted_source = nx_graph if nx_graph else self.graph_source
    
    def _create_property_graph_retriever(self, **kwargs):
        """建立 PropertyGraph Retriever（支援多 sub_retriever 組合）"""
        from src.graph_retriever.unified.property_graph import PropertyGraphRetrieverFactory
        
        # 確保有 PropertyGraphIndex
        pg_index = self.converted_source
        
        # 建立 sub_retrievers
        sub_retrievers = PropertyGraphRetrieverFactory.create_retrievers(
            pg_index=pg_index,
            settings=self.settings,
            retriever_config=self.retriever_config
        )
        
        print(f"📦 建立了 {len(sub_retrievers)} 個 PropertyGraph sub_retrievers")
        
        # 根據 combination_mode 建立 retriever
        if self.combination_mode == "ensemble":
            # 使用 ensemble 模式
            from llama_index.core.indices.property_graph import PGRetriever
            self.actual_retriever = PGRetriever(
                sub_retrievers=sub_retrievers,
                include_text=True
            )
        elif self.combination_mode == "cascade":
            # cascade 模式：依序呼叫 sub_retrievers，結果不足時才用下一個補充
            self._cascade_sub_retrievers = sub_retrievers
            self.actual_retriever = None
        elif self.combination_mode == "single":
            # 單一 retriever
            if sub_retrievers:
                self.actual_retriever = sub_retrievers[0]
            else:
                raise ValueError("無可用的 sub_retriever")
        else:
            raise ValueError(f"未知的組合模式: {self.combination_mode}")
    
    def get_name(self) -> str:
        return f"Unified[{self.retriever_type}]"
    
    def retrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        同步檢索
        
        Args:
            query: 查詢字串
            graph_data: 圖譜資料（可選）
            top_k: 檢索數量
            **kwargs: 額外參數
        
        Returns:
            檢索結果字典
        """
        # PropertyGraph retriever 使用不同的介面
        if self.retriever_type == "property_graph":
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                print("⚠️  建議安裝 nest_asyncio: pip install nest_asyncio")
            
            nodes = self._pg_retrieve_sync(query, top_k)
            
            contexts = []
            for node in nodes:
                if hasattr(node, 'text'):
                    contexts.append(node.text)
                elif hasattr(node, 'node') and hasattr(node.node, 'text'):
                    contexts.append(node.node.text)
            
            return {
                "contexts": contexts[:top_k],
                "nodes": nodes[:top_k],
                "metadata": {
                    "retriever_type": self.retriever_type,
                    "combination_mode": self.combination_mode
                }
            }
        
        return self.actual_retriever.retrieve(
            query=query,
            graph_data=graph_data,
            top_k=top_k,
            **kwargs
        )
    
    def _pg_retrieve_sync(self, query: str, top_k: int = 2) -> list:
        """PropertyGraph 同步檢索，支援 cascade 模式"""
        if self.combination_mode == "cascade" and hasattr(self, '_cascade_sub_retrievers'):
            return self._cascade_retrieve_sync(query, top_k)
        
        try:
            return self.actual_retriever.retrieve(query)
        except Exception as e:
            import traceback
            print(f"❌ PGRetriever.retrieve() 錯誤:")
            traceback.print_exc()
            raise
    
    def _cascade_retrieve_sync(self, query: str, top_k: int) -> list:
        """Cascade 模式：依序呼叫 sub_retrievers，結果不足時用下一個補充"""
        all_nodes = []
        seen_ids = set()
        
        for i, retriever in enumerate(self._cascade_sub_retrievers):
            if len(all_nodes) >= top_k:
                break
            try:
                nodes = retriever.retrieve(query)
                for node in nodes:
                    node_id = getattr(node, 'node_id', None) or id(node)
                    if node_id not in seen_ids:
                        seen_ids.add(node_id)
                        all_nodes.append(node)
                print(f"  🔗 Cascade [{i+1}/{len(self._cascade_sub_retrievers)}]: 累計 {len(all_nodes)} 個結果")
            except Exception as e:
                print(f"  ⚠️  Cascade sub_retriever [{i+1}] 失敗: {e}")
                continue
        
        return all_nodes
    
    async def aretrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        非同步檢索（如果 retriever 支援）
        
        Args:
            query: 查詢字串
            graph_data: 圖譜資料（可選）
            top_k: 檢索數量
            **kwargs: 額外參數
        
        Returns:
            檢索結果字典
        """
        if self.retriever_type == "property_graph":
            # cascade 模式走同步 fallback
            if self.combination_mode == "cascade" and hasattr(self, '_cascade_sub_retrievers'):
                nodes = self._cascade_retrieve_sync(query, top_k)
            elif self.actual_retriever is not None and hasattr(self.actual_retriever, 'aretrieve'):
                nodes = await self.actual_retriever.aretrieve(query)
            elif self.actual_retriever is not None:
                nodes = self.actual_retriever.retrieve(query)
            else:
                nodes = []
            
            contexts = []
            for node in nodes:
                if hasattr(node, 'text'):
                    contexts.append(node.text)
                elif hasattr(node, 'node') and hasattr(node.node, 'text'):
                    contexts.append(node.node.text)
            
            return {
                "contexts": contexts[:top_k],
                "nodes": nodes[:top_k],
                "metadata": {
                    "retriever_type": self.retriever_type,
                    "combination_mode": self.combination_mode
                }
            }
        
        if hasattr(self.actual_retriever, 'aretrieve'):
            return await self.actual_retriever.aretrieve(query, graph_data, top_k, **kwargs)
        else:
            return self.retrieve(query, graph_data, top_k, **kwargs)
