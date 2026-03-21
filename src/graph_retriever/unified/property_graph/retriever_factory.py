"""
PropertyGraph Retriever Factory

支援組合多個 LlamaIndex PropertyGraph sub_retrievers
"""

from typing import List, Dict, Any


class PropertyGraphRetrieverFactory:
    """
    PropertyGraph 專用的 Retriever Factory
    
    支援組合多個 LlamaIndex sub_retrievers:
    - VectorContextRetriever: 向量相似度檢索
    - LLMSynonymRetriever: LLM 同義詞擴展檢索
    - TextToCypherRetriever: 自然語言轉 Cypher 查詢
    """
    
    @staticmethod
    def create_retrievers(
        pg_index: Any,
        settings: Any,
        retriever_config: Dict[str, Any]
    ) -> List[Any]:
        """
        根據配置建立多個 PropertyGraph sub_retrievers
        
        Args:
            pg_index: PropertyGraphIndex 實例
            settings: LlamaIndex Settings
            retriever_config: Retriever 配置字典
                {
                    "vector": {"enabled": true, "similarity_top_k": 5},
                    "synonym": {"enabled": true, "include_text": true},
                    "text2cypher": {"enabled": false}
                }
        
        Returns:
            Sub-retriever 實例列表
        """
        sub_retrievers = []
        
        if not retriever_config:
            print("⚠️  未提供 retriever_config，將使用預設 retrievers")
            retriever_config = {
                "vector": {"enabled": True},
                "synonym": {"enabled": True}
            }
        
        # 1. VectorContextRetriever
        if retriever_config.get("vector", {}).get("enabled", False):
            try:
                from llama_index.core.indices.property_graph import VectorContextRetriever
                vector_cfg = retriever_config["vector"]
                
                embed_model = getattr(settings, 'embed_model', None)
                if not embed_model:
                    print("⚠️  Vector retriever 需要 embed_model，跳過")
                else:
                    similarity_top_k = vector_cfg.get("similarity_top_k", 5)
                    
                    retriever = VectorContextRetriever(
                        pg_index.property_graph_store,
                        vector_store=pg_index.vector_store if hasattr(pg_index, 'vector_store') else None,
                        embed_model=embed_model,
                        similarity_top_k=similarity_top_k
                    )
                    sub_retrievers.append(retriever)
                    print(f"✅ 已啟用 VectorContextRetriever (top_k={similarity_top_k})")
            
            except ImportError as e:
                print(f"⚠️  VectorContextRetriever 匯入失敗: {e}")
        
        # 2. LLMSynonymRetriever
        if retriever_config.get("synonym", {}).get("enabled", False):
            try:
                from llama_index.core.indices.property_graph import LLMSynonymRetriever
                synonym_cfg = retriever_config["synonym"]
                
                llm = getattr(settings, 'llm', None)
                if not llm:
                    print("⚠️  Synonym retriever 需要 LLM，跳過")
                else:
                    include_text = synonym_cfg.get("include_text", True)
                    
                    retriever = LLMSynonymRetriever(
                        pg_index.property_graph_store,
                        llm=llm,
                        include_text=include_text
                    )
                    sub_retrievers.append(retriever)
                    print(f"✅ 已啟用 LLMSynonymRetriever (include_text={include_text})")
            
            except ImportError as e:
                print(f"⚠️  LLMSynonymRetriever 匯入失敗: {e}")
        
        # 3. TextToCypherRetriever
        if retriever_config.get("text2cypher", {}).get("enabled", False):
            try:
                from llama_index.core.indices.property_graph import TextToCypherRetriever
                cypher_cfg = retriever_config["text2cypher"]
                
                graph_store = pg_index.property_graph_store
                if not getattr(graph_store, 'supports_structured_queries', False):
                    from src.utils.cypher_capable_store import CypherCapablePropertyGraphStore
                    graph_store = CypherCapablePropertyGraphStore.from_simple_store(graph_store)
                    print("🔄 已包裝 graph store 以支援 Cypher 查詢")
                
                llm = getattr(settings, 'llm', None)
                if not llm:
                    print("⚠️  Text2Cypher retriever 需要 LLM，跳過")
                else:
                    retriever = TextToCypherRetriever(
                        graph_store,
                        llm=llm
                    )
                    sub_retrievers.append(retriever)
                    print(f"✅ 已啟用 TextToCypherRetriever")
            
            except ImportError as e:
                print(f"⚠️  TextToCypherRetriever 匯入失敗: {e}")
            except Exception as e:
                print(f"⚠️  TextToCypherRetriever 建立失敗: {e}")
        
        if not sub_retrievers:
            print("⚠️  未建立任何 sub_retriever，檢索可能失敗")
        else:
            print(f"📊 總共建立了 {len(sub_retrievers)} 個 sub_retrievers")
        
        return sub_retrievers
