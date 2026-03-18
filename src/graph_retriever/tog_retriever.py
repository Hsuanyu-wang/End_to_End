"""
Think-on-Graph (ToG) 風格檢索器

實作文本與圖譜的緊密耦合、迭代式檢索
基於 ToG-2.0 論文的設計理念
"""

from typing import List, Dict, Any, Set
import time


class ToGRetriever:
    """ToG 迭代式檢索器"""
    
    def __init__(
        self,
        vector_index,
        graph_index,
        llm=None,
        max_iterations: int = 3,
        convergence_threshold: float = 0.9,
        use_llm_sufficiency_check: bool = True
    ):
        """
        初始化 ToG 檢索器
        
        Args:
            vector_index: 向量索引
            graph_index: 圖譜索引
            llm: LLM 實例（用於收斂判斷）
            max_iterations: 最大迭代次數
            convergence_threshold: 收斂閾值
            use_llm_sufficiency_check: 是否使用 LLM 判斷是否充分
        """
        self.vector_index = vector_index
        self.graph_index = graph_index
        self.llm = llm
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_llm_sufficiency_check = use_llm_sufficiency_check
        
        print(f"🔧 初始化 ToG 檢索器 (max_iter={max_iterations})")
    
    def iterative_retrieve(
        self,
        query: str,
        initial_entities: List[Any] = None,
        max_iterations: int = None
    ) -> Dict[str, Any]:
        """
        執行迭代式檢索
        
        Args:
            query: 查詢字串
            initial_entities: 初始實體（可選）
            max_iterations: 最大迭代次數（覆蓋初始化設定）
            
        Returns:
            檢索結果字典
        """
        max_iter = max_iterations or self.max_iterations
        
        contexts = []
        current_entities = initial_entities or []
        all_entities = set()
        all_relations = []
        
        print(f"🔍 開始 ToG 迭代檢索: {query}")
        
        for iteration in range(max_iter):
            print(f"   迭代 {iteration + 1}/{max_iter}")
            
            # 1. Vector retrieval（提供實體上下文）
            if iteration == 0:
                doc_contexts = self._vector_retrieve(query, top_k=5)
                contexts.extend(doc_contexts)
                current_entities = self._extract_entities(doc_contexts)
                print(f"      向量檢索取得 {len(current_entities)} 個實體")
            
            # 2. Graph retrieval（基於當前實體擴展）
            if current_entities:
                graph_contexts = self._graph_retrieve(current_entities, depth=2)
                contexts.extend(graph_contexts)
                
                # 記錄新實體和關係
                new_entities = self._extract_entities(graph_contexts)
                new_relations = self._extract_relations(graph_contexts)
                
                for e in new_entities:
                    all_entities.add(self._entity_to_str(e))
                all_relations.extend(new_relations)
                
                print(f"      圖譜檢索取得 {len(new_entities)} 個實體, {len(new_relations)} 個關係")
            
            # 3. 判斷是否需要繼續迭代
            if self._is_sufficient(query, contexts):
                print(f"   ✅ 資訊充分，提前結束 (迭代 {iteration + 1})")
                break
                
            # 4. 從圖譜上下文中抽取新實體作為下一輪的起點
            current_entities = self._extract_entities(graph_contexts) if 'graph_contexts' in locals() else []
        
        result = {
            "entities": list(all_entities),
            "relations": all_relations,
            "contexts": contexts,
            "iterations": iteration + 1
        }
        
        print(f"✅ ToG 檢索完成: {len(all_entities)} 個實體, {len(all_relations)} 個關係")
        
        return result
    
    def _vector_retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """向量檢索"""
        if self.vector_index is None:
            return []
        
        try:
            # 假設 vector_index 有 retrieve 方法
            results = self.vector_index.retrieve(query, top_k=top_k)
            return results if isinstance(results, list) else []
        except Exception as e:
            print(f"⚠️  向量檢索失敗: {e}")
            return []
    
    def _graph_retrieve(self, entities: List[Any], depth: int = 2) -> List[Any]:
        """圖譜檢索（基於實體擴展）"""
        if self.graph_index is None:
            return []
        
        try:
            # 假設 graph_index 有 retrieve_neighbors 方法
            results = self.graph_index.retrieve_neighbors(entities, depth=depth)
            return results if isinstance(results, list) else []
        except Exception as e:
            print(f"⚠️  圖譜檢索失敗: {e}")
            return []
    
    def _is_sufficient(self, query: str, contexts: List[Any]) -> bool:
        """
        判斷是否有足夠的資訊回答查詢
        
        Args:
            query: 查詢字串
            contexts: 已檢索的上下文
            
        Returns:
            是否充分
        """
        if not self.use_llm_sufficiency_check or self.llm is None:
            # Fallback: 簡單判斷（基於上下文數量）
            return len(contexts) >= 10
        
        try:
            # 使用 LLM 判斷
            context_text = "\n".join([self._context_to_str(c) for c in contexts[:5]])
            
            prompt = f"""
            問題: {query}
            
            目前收集的上下文:
            {context_text}
            
            請判斷這些上下文是否足以回答問題。
            只回答 "是" 或 "否"。
            """
            
            response = self.llm.complete(prompt)
            return "是" in response.text if hasattr(response, 'text') else "是" in str(response)
            
        except Exception as e:
            print(f"⚠️  LLM 判斷失敗: {e}")
            return len(contexts) >= 10
    
    def _extract_entities(self, contexts: List[Any]) -> List[Any]:
        """從上下文中抽取實體"""
        entities = []
        
        for context in contexts:
            if hasattr(context, 'entities'):
                entities.extend(context.entities)
            elif hasattr(context, 'metadata') and 'entities' in context.metadata:
                entities.extend(context.metadata['entities'])
        
        return entities
    
    def _extract_relations(self, contexts: List[Any]) -> List[Any]:
        """從上下文中抽取關係"""
        relations = []
        
        for context in contexts:
            if hasattr(context, 'relations'):
                relations.extend(context.relations)
            elif hasattr(context, 'metadata') and 'relations' in context.metadata:
                relations.extend(context.metadata['relations'])
        
        return relations
    
    def _entity_to_str(self, entity: Any) -> str:
        """將實體轉換為字串"""
        if isinstance(entity, str):
            return entity
        elif hasattr(entity, 'name'):
            return entity.name
        elif hasattr(entity, 'text'):
            return entity.text
        else:
            return str(entity)
    
    def _context_to_str(self, context: Any) -> str:
        """將上下文轉換為字串"""
        if isinstance(context, str):
            return context
        elif hasattr(context, 'text'):
            return context.text
        elif hasattr(context, 'content'):
            return context.content
        else:
            return str(context)
