"""
PropertyGraph Retriever

使用 PropertyGraphIndex 的檢索實作，支援多 sub_retriever 組合
"""

from typing import Dict, Any, List, Optional
from src.graph_retriever.base_retriever import BaseGraphRetriever
from .retriever_factory import PropertyGraphRetrieverFactory

_DOC_ID_KEYS = ("NO", "doc_id", "original_no", "file_name", "source")


class PropertyGraphRetriever(BaseGraphRetriever):
    """
    PropertyGraph Retriever
    
    支援組合多個 sub_retrievers (VectorContext + LLMSynonym + Text2Cypher)
    """
    
    def __init__(
        self,
        graph_source: Any,
        settings: Any,
        retriever_config: Dict[str, Any] = None,
        combination_mode: str = "ensemble",
        **kwargs
    ):
        """
        初始化 PropertyGraph Retriever
        
        Args:
            graph_source: PropertyGraphIndex 實例
            settings: LlamaIndex Settings
            retriever_config: Retriever 配置
            combination_mode: 組合模式（ensemble/cascade/single）
        """
        self.graph_source = graph_source
        self.settings = settings
        self.retriever_config = retriever_config or {}
        self.combination_mode = combination_mode
        
        # 建立 sub_retrievers
        self.sub_retrievers = PropertyGraphRetrieverFactory.create_retrievers(
            graph_source,
            settings,
            self.retriever_config
        )
        
        print(f"🔧 初始化 PropertyGraphRetriever")
        print(f"   - 組合模式: {combination_mode}")
        print(f"   - Sub-retrievers: {len(self.sub_retrievers)}")
    
    def get_name(self) -> str:
        return f"PropertyGraph_{self.combination_mode}"
    
    @staticmethod
    def _extract_doc_ids(nodes: list) -> List[str]:
        """從 LlamaIndex 節點中提取文件層級 ID（去重保序）"""
        seen: set = set()
        doc_ids: List[str] = []
        for node in nodes:
            meta = getattr(node, "metadata", None) or {}
            doc_id = None
            for key in _DOC_ID_KEYS:
                val = meta.get(key)
                if val is not None and str(val).strip():
                    doc_id = str(val).strip()
                    break
            if doc_id is None:
                ref = getattr(node, "ref_doc_id", None)
                if ref and str(ref).strip():
                    doc_id = str(ref).strip()
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
        return doc_ids

    def retrieve(
        self,
        query: str,
        graph_data: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        檢索
        
        Args:
            query: 查詢字串
            graph_data: 圖譜資料（可選）
            top_k: 檢索數量
        
        Returns:
            檢索結果
        """
        print(f"🔍 [PropertyGraph-{self.combination_mode}] 開始檢索...")
        
        if not self.sub_retrievers:
            print("⚠️  沒有可用的 sub_retriever")
            return {
                "contexts": [],
                "nodes": [],
                "retrieved_ids": [],
                "metadata": {"error": "no_sub_retrievers"}
            }
        
        try:
            if self.combination_mode == "ensemble":
                return self._retrieve_ensemble(query, top_k)
            elif self.combination_mode == "cascade":
                return self._retrieve_cascade(query, top_k)
            else:  # single
                return self._retrieve_single(query, top_k)
        
        except Exception as e:
            print(f"❌ 檢索失敗: {e}")
            return {
                "contexts": [],
                "nodes": [],
                "retrieved_ids": [],
                "metadata": {"error": str(e)}
            }
    
    def _retrieve_ensemble(self, query: str, top_k: int) -> Dict[str, Any]:
        """Ensemble 模式：合併所有 retriever 結果"""
        all_nodes = []
        all_contexts = []
        
        for retriever in self.sub_retrievers:
            try:
                nodes = retriever.retrieve(query)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"⚠️  Sub-retriever 失敗: {e}")
        
        for node in all_nodes[:top_k]:
            if hasattr(node, 'text'):
                all_contexts.append(node.text)
            elif hasattr(node, 'get_content'):
                all_contexts.append(node.get_content())
        
        trimmed = all_nodes[:top_k]
        return {
            "contexts": all_contexts,
            "nodes": trimmed,
            "retrieved_ids": self._extract_doc_ids(trimmed),
            "metadata": {
                "mode": "ensemble",
                "num_sub_retrievers": len(self.sub_retrievers)
            }
        }
    
    def _retrieve_cascade(self, query: str, top_k: int) -> Dict[str, Any]:
        """Cascade 模式：依序執行，前者不足才用後者"""
        all_nodes = []
        all_contexts = []
        
        for retriever in self.sub_retrievers:
            if len(all_nodes) >= top_k:
                break
            
            try:
                nodes = retriever.retrieve(query)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"⚠️  Sub-retriever 失敗，嘗試下一個: {e}")
        
        for node in all_nodes[:top_k]:
            if hasattr(node, 'text'):
                all_contexts.append(node.text)
            elif hasattr(node, 'get_content'):
                all_contexts.append(node.get_content())
        
        trimmed = all_nodes[:top_k]
        return {
            "contexts": all_contexts,
            "nodes": trimmed,
            "retrieved_ids": self._extract_doc_ids(trimmed),
            "metadata": {"mode": "cascade"}
        }
    
    def _retrieve_single(self, query: str, top_k: int) -> Dict[str, Any]:
        """Single 模式：只使用第一個 retriever"""
        if not self.sub_retrievers:
            return {"contexts": [], "nodes": [], "retrieved_ids": [], "metadata": {}}
        
        try:
            nodes = self.sub_retrievers[0].retrieve(query)
            contexts = []
            
            for node in nodes[:top_k]:
                if hasattr(node, 'text'):
                    contexts.append(node.text)
                elif hasattr(node, 'get_content'):
                    contexts.append(node.get_content())
            
            trimmed = nodes[:top_k]
            return {
                "contexts": contexts,
                "nodes": trimmed,
                "retrieved_ids": self._extract_doc_ids(trimmed),
                "metadata": {"mode": "single"}
            }
        
        except Exception as e:
            print(f"❌ 檢索失敗: {e}")
            return {"contexts": [], "nodes": [], "retrieved_ids": [], "metadata": {"error": str(e)}}
