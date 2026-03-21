"""
關係抽取組件抽象基類

定義了關係抽取器的統一介面，所有關係抽取實作都應繼承此基類。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

# 從頂層 formats 模組導入
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from formats import Entity, EntityList, Relation, RelationList, Schema


class BaseRelationExtractor(ABC):
    """
    關係抽取器抽象基類
    
    所有關係抽取組件（LightRAG、模式匹配、深度學習等）都應實作此介面。
    
    關鍵設計原則：
    1. 輸入：文本 + 已抽取的實體 + 可選的 Schema 約束
    2. 輸出：標準 Relation 物件列表
    3. 關係應參照輸入的實體 ID
    
    Examples:
        >>> class LightRAGRelationExtractor(BaseRelationExtractor):
        ...     def extract(self, text, entities, schema=None):
        ...         # LightRAG 的關係抽取邏輯
        ...         return RelationList([...])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化關係抽取器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._initialize()
    
    def _initialize(self):
        """子類可覆寫此方法進行初始化"""
        pass
    
    @abstractmethod
    def extract(
        self,
        text: str,
        entities: EntityList,
        schema: Optional[Schema] = None,
        **kwargs
    ) -> RelationList:
        """
        從文本中抽取實體之間的關係
        
        這是核心抽象方法，所有子類必須實作。
        
        Args:
            text: 輸入文本
            entities: 已抽取的實體列表
            schema: 可選的 Schema 約束（用於指導關係抽取）
            **kwargs: 額外參數
        
        Returns:
            RelationList: 抽取到的關係列表
        
        Raises:
            ValueError: 當輸入無效時
            RuntimeError: 當抽取過程失敗時
        
        Notes:
            - 抽取的關係應使用 entities 中的 entity_id
            - 如果提供了 schema，應遵循 schema 中定義的關係類型
        """
        pass
    
    def extract_batch(
        self,
        texts: List[str],
        entities_list: List[EntityList],
        schema: Optional[Schema] = None,
        **kwargs
    ) -> List[RelationList]:
        """
        批次抽取關係
        
        預設實作為逐一處理，子類可覆寫以提供更高效的批次處理。
        
        Args:
            texts: 文本列表
            entities_list: 對應每個文本的實體列表
            schema: 可選的 Schema 約束
            **kwargs: 額外參數
        
        Returns:
            List[RelationList]: 每個文本對應的關係列表
        """
        if len(texts) != len(entities_list):
            raise ValueError("texts and entities_list must have the same length")
        
        results = []
        for text, entities in zip(texts, entities_list):
            relations = self.extract(text, entities, schema, **kwargs)
            results.append(relations)
        return results
    
    def get_name(self) -> str:
        """
        獲取抽取器名稱
        
        Returns:
            str: 抽取器名稱
        """
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """獲取當前配置"""
        return self.config.copy()
    
    def validate_input(self, text: str, entities: EntityList) -> bool:
        """
        驗證輸入
        
        Args:
            text: 輸入文本
            entities: 實體列表
        
        Returns:
            bool: 是否有效
        """
        if not text or not isinstance(text, str):
            return False
        if len(text.strip()) == 0:
            return False
        if not isinstance(entities, EntityList):
            return False
        if len(entities) == 0:
            # 沒有實體時無法抽取關係
            return False
        return True
    
    def filter_invalid_relations(
        self,
        relations: RelationList,
        entities: EntityList
    ) -> RelationList:
        """
        過濾無效的關係
        
        移除指向不存在實體的關係。
        
        Args:
            relations: 關係列表
            entities: 有效的實體列表
        
        Returns:
            RelationList: 過濾後的關係列表
        """
        entity_ids = set(e.entity_id for e in entities)
        valid_relations = []
        
        for relation in relations:
            if relation.head in entity_ids and relation.tail in entity_ids:
                valid_relations.append(relation)
        
        return RelationList(relations=valid_relations)
