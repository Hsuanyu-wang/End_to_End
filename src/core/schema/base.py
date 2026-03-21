"""
Schema 管理組件抽象基類

定義了 Schema Manager 的統一介面，支援不同的 Schema Learning 方法。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple

# 從頂層 formats 模組導入
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from formats import Entity, EntityList, Relation, RelationList, Schema


class BaseSchemaManager(ABC):
    """
    Schema 管理器抽象基類
    
    所有 Schema Learning 組件（Fixed、AutoSchema、Dynamic、Ontology）都應實作此介面。
    
    關鍵設計原則：
    1. 支援 Schema 學習：從實體和關係中學習 Schema
    2. 支援 Schema 對齊：將抽取的實體和關係對齊到 Schema
    3. 支援 Schema 演化：動態更新 Schema
    
    Examples:
        >>> class AutoSchemaManager(BaseSchemaManager):
        ...     def learn(self, entities, relations):
        ...         # 從資料中學習 Schema
        ...         return schema
        ...     def align(self, entities, relations, schema):
        ...         # 將資料對齊到 Schema
        ...         return aligned_entities, aligned_relations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Schema 管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.current_schema: Optional[Schema] = None
        self._initialize()
    
    def _initialize(self):
        """子類可覆寫此方法進行初始化"""
        pass
    
    @abstractmethod
    def learn(
        self,
        entities: EntityList,
        relations: RelationList,
        **kwargs
    ) -> Schema:
        """
        從實體和關係中學習 Schema
        
        這是核心抽象方法，所有子類必須實作。
        
        Args:
            entities: 實體列表
            relations: 關係列表
            **kwargs: 額外參數
        
        Returns:
            Schema: 學習到的 Schema
        
        Notes:
            - Fixed Schema Manager: 返回預定義的 Schema
            - AutoSchema Manager: 從資料中無監督學習
            - Dynamic Schema Manager: 基於當前 schema 演化
            - Ontology Schema Manager: 從本體推導
        """
        pass
    
    @abstractmethod
    def align(
        self,
        entities: EntityList,
        relations: RelationList,
        schema: Schema,
        **kwargs
    ) -> Tuple[EntityList, RelationList]:
        """
        將實體和關係對齊到 Schema
        
        這是核心抽象方法，所有子類必須實作。
        
        Args:
            entities: 原始實體列表
            relations: 原始關係列表
            schema: 目標 Schema
            **kwargs: 額外參數
        
        Returns:
            Tuple[EntityList, RelationList]: 對齊後的實體和關係
        
        Notes:
            - 可能包括：類型標準化、實體消歧、關係驗證等
            - 不符合 schema 的資料可以被過濾或調整
        """
        pass
    
    def get_schema(self) -> Optional[Schema]:
        """
        獲取當前 Schema
        
        Returns:
            Optional[Schema]: 當前 Schema，如果尚未學習則返回 None
        """
        return self.current_schema
    
    def set_schema(self, schema: Schema):
        """
        設定 Schema
        
        Args:
            schema: 要設定的 Schema
        """
        self.current_schema = schema
    
    def update_schema(
        self,
        new_entities: EntityList,
        new_relations: RelationList,
        **kwargs
    ) -> Schema:
        """
        更新 Schema（用於動態演化）
        
        預設實作：重新學習。子類可覆寫以提供增量更新。
        
        Args:
            new_entities: 新的實體
            new_relations: 新的關係
            **kwargs: 額外參數
        
        Returns:
            Schema: 更新後的 Schema
        """
        if self.current_schema is None:
            # 如果沒有現有 schema，直接學習
            self.current_schema = self.learn(new_entities, new_relations, **kwargs)
        else:
            # 重新學習（子類可覆寫以提供更智能的更新）
            new_schema = self.learn(new_entities, new_relations, **kwargs)
            self.current_schema = self.current_schema.merge(new_schema)
        
        return self.current_schema
    
    def validate(
        self,
        entities: EntityList,
        relations: RelationList,
        schema: Optional[Schema] = None
    ) -> Tuple[List[str], List[str]]:
        """
        驗證實體和關係是否符合 Schema
        
        Args:
            entities: 要驗證的實體
            relations: 要驗證的關係
            schema: Schema，如果為 None 則使用 current_schema
        
        Returns:
            Tuple[List[str], List[str]]: (實體錯誤列表, 關係錯誤列表)
        """
        if schema is None:
            schema = self.current_schema
        
        if schema is None:
            return [], []
        
        entity_errors = []
        relation_errors = []
        
        # 驗證實體
        for entity in entities:
            if not schema.validate_entity(entity.type):
                entity_errors.append(f"Invalid entity type: {entity.type} (entity_id: {entity.entity_id})")
        
        # 驗證關係
        entity_type_map = {e.entity_id: e.type for e in entities}
        for relation in relations:
            head_type = entity_type_map.get(relation.head, "")
            tail_type = entity_type_map.get(relation.tail, "")
            if not schema.validate_relation(relation.relation, head_type, tail_type):
                relation_errors.append(
                    f"Invalid relation: {relation.relation} "
                    f"({head_type} -> {tail_type}, relation_id: {relation.relation_id})"
                )
        
        return entity_errors, relation_errors
    
    def get_name(self) -> str:
        """
        獲取管理器名稱
        
        Returns:
            str: 管理器名稱
        """
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """獲取當前配置"""
        return self.config.copy()
