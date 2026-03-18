"""
實體消歧模組

使用 embedding 相似度和 LLM 判斷來合併重複的實體，
解決同一實體不同提及方式導致的圖譜碎片化問題
"""

from typing import List, Dict, Tuple, Any
import numpy as np


class EntityDisambiguator:
    """實體消歧器"""
    
    def __init__(
        self, 
        llm=None, 
        embed_model=None,
        similarity_threshold: float = 0.85,
        use_llm_verification: bool = True
    ):
        """
        初始化實體消歧器
        
        Args:
            llm: LLM 實例（用於驗證）
            embed_model: Embedding 模型實例
            similarity_threshold: 相似度閾值
            use_llm_verification: 是否使用 LLM 進行驗證
        """
        self.llm = llm
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.use_llm_verification = use_llm_verification
        
        print(f"🔧 初始化實體消歧器 (threshold={similarity_threshold})")
        
    def merge_entities(self, entities: List[Any]) -> List[Any]:
        """
        合併重複的實體
        
        Args:
            entities: 實體列表
            
        Returns:
            去重後的實體列表
        """
        if not entities:
            return entities
            
        print(f"📊 開始實體消歧: {len(entities)} 個實體")
        
        # 1. 計算 embedding 相似度
        entity_pairs = self._find_similar_pairs(entities)
        
        if not entity_pairs:
            print("✅ 沒有發現相似實體")
            return entities
            
        print(f"🔍 發現 {len(entity_pairs)} 對相似實體")
        
        # 2. 使用 LLM 驗證（可選）
        if self.use_llm_verification and self.llm:
            entity_pairs = self._verify_with_llm(entity_pairs)
            print(f"✅ LLM 驗證後剩餘 {len(entity_pairs)} 對")
        
        # 3. 合併實體
        merged_entities = self._perform_merge(entities, entity_pairs)
        
        print(f"✅ 消歧完成: {len(entities)} → {len(merged_entities)} 個實體")
        
        return merged_entities
    
    def _find_similar_pairs(self, entities: List[Any]) -> List[Tuple[int, int, float]]:
        """
        尋找相似的實體對
        
        Args:
            entities: 實體列表
            
        Returns:
            [(idx1, idx2, similarity), ...] 相似實體對列表
        """
        if self.embed_model is None:
            # Fallback: 使用簡單的字串匹配
            return self._simple_string_matching(entities)
        
        try:
            # 取得實體文本
            entity_texts = [self._get_entity_text(e) for e in entities]
            
            # 計算 embeddings
            embeddings = self.embed_model.get_text_embedding_batch(entity_texts)
            embeddings = np.array(embeddings)
            
            # 計算相似度矩陣
            similarities = np.dot(embeddings, embeddings.T)
            
            # 尋找高相似度對
            pairs = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    if similarities[i, j] >= self.similarity_threshold:
                        pairs.append((i, j, float(similarities[i, j])))
            
            return pairs
            
        except Exception as e:
            print(f"⚠️  Embedding 計算失敗: {e}，使用簡單匹配")
            return self._simple_string_matching(entities)
    
    def _simple_string_matching(self, entities: List[Any]) -> List[Tuple[int, int, float]]:
        """
        簡單的字串匹配（Fallback）
        
        Args:
            entities: 實體列表
            
        Returns:
            相似實體對列表
        """
        pairs = []
        entity_texts = [self._get_entity_text(e).lower() for e in entities]
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                text1, text2 = entity_texts[i], entity_texts[j]
                
                # 檢查是否為子字串
                if text1 in text2 or text2 in text1:
                    score = 0.9
                    pairs.append((i, j, score))
                # 檢查編輯距離
                elif self._levenshtein_similarity(text1, text2) >= self.similarity_threshold:
                    score = self._levenshtein_similarity(text1, text2)
                    pairs.append((i, j, score))
        
        return pairs
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """計算 Levenshtein 相似度"""
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
            
        # 簡化版：計算公共字符比例
        common = len(set(s1) & set(s2))
        total = len(set(s1) | set(s2))
        return common / total if total > 0 else 0.0
    
    def _verify_with_llm(self, pairs: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        使用 LLM 驗證實體對是否真的相同
        
        Args:
            pairs: 候選實體對
            
        Returns:
            驗證後的實體對
        """
        # TODO: 實作 LLM 驗證邏輯
        # 暫時返回原始 pairs
        return pairs
    
    def _perform_merge(
        self, 
        entities: List[Any], 
        pairs: List[Tuple[int, int, float]]
    ) -> List[Any]:
        """
        執行實體合併
        
        Args:
            entities: 實體列表
            pairs: 要合併的實體對
            
        Returns:
            合併後的實體列表
        """
        # 使用 Union-Find 找出連通分量
        parent = list(range(len(entities)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 合併相似實體
        for i, j, _ in pairs:
            union(i, j)
        
        # 分組實體
        groups = {}
        for i in range(len(entities)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(entities[i])
        
        # 為每個組選擇代表實體並合併資訊
        merged = []
        for group in groups.values():
            merged_entity = self._merge_entity_group(group)
            merged.append(merged_entity)
        
        return merged
    
    def _merge_entity_group(self, group: List[Any]) -> Any:
        """
        合併一組實體
        
        Args:
            group: 實體組
            
        Returns:
            合併後的實體
        """
        # 選擇第一個作為代表
        representative = group[0]
        
        # TODO: 可以實作更複雜的合併邏輯
        # 例如：合併所有 descriptions、關係等
        
        return representative
    
    def _get_entity_text(self, entity: Any) -> str:
        """從實體物件中提取文本"""
        if hasattr(entity, 'name'):
            return entity.name
        elif hasattr(entity, 'text'):
            return entity.text
        elif hasattr(entity, 'label'):
            return entity.label
        elif isinstance(entity, str):
            return entity
        else:
            return str(entity)
    
    def resolve_coreference(self, text: str) -> Dict[str, str]:
        """
        共指消解（使用 LLM 識別代詞和實體指涉）
        
        Args:
            text: 輸入文本
            
        Returns:
            {mention: entity} 映射字典
        """
        # TODO: 實作共指消解邏輯
        # 暫時返回空字典
        return {}
