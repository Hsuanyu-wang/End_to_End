"""
實體標準格式定義

定義了統一的實體表示格式，支援從不同來源（LightRAG、AutoSchema、Ontology）的實體。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class Entity:
    """
    標準實體格式
    
    所有組件產出的實體都應轉換為此格式，確保互操作性。
    
    Attributes:
        entity_id: 實體唯一識別碼
        name: 實體名稱（顯示名稱）
        type: 實體類型（如 "Person", "Organization", "Event"）
        properties: 實體屬性字典（可包含任意額外資訊）
        aliases: 實體別名列表
        confidence: 信心分數 (0.0-1.0)
        source: 來源標識（如 "lightrag", "autoschema", "ontology"）
        metadata: 元資料（如時間戳、來源文件等）
    
    Examples:
        >>> entity = Entity(
        ...     entity_id="e1",
        ...     name="張三",
        ...     type="Person",
        ...     properties={"age": 30, "title": "工程師"},
        ...     aliases=["Zhang San", "小張"],
        ...     confidence=0.95,
        ...     source="lightrag"
        ... )
    """
    entity_id: str
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """驗證實體資料的有效性"""
        if not self.entity_id:
            raise ValueError("entity_id cannot be empty")
        if not self.name:
            raise ValueError("name cannot be empty")
        if not self.type:
            raise ValueError("type cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """從字典創建實體"""
        return cls(
            entity_id=data["entity_id"],
            name=data["name"],
            type=data["type"],
            properties=data.get("properties", {}),
            aliases=data.get("aliases", []),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )
    
    def merge_with(self, other: "Entity") -> "Entity":
        """
        合併兩個實體
        
        當檢測到重複實體時使用，保留信心度較高的資訊。
        """
        if self.entity_id != other.entity_id:
            raise ValueError(f"Cannot merge entities with different IDs: {self.entity_id} vs {other.entity_id}")
        
        # 使用信心度較高的名稱
        name = self.name if self.confidence >= other.confidence else other.name
        
        # 使用信心度較高的類型
        entity_type = self.type if self.confidence >= other.confidence else other.type
        
        # 合併屬性
        merged_properties = {**self.properties, **other.properties}
        
        # 合併別名
        merged_aliases = list(set(self.aliases + other.aliases))
        
        # 使用較高的信心度
        confidence = max(self.confidence, other.confidence)
        
        # 合併來源
        sources = [s for s in [self.source, other.source] if s]
        source = "+".join(sorted(set(sources)))
        
        # 合併元資料
        merged_metadata = {**self.metadata, **other.metadata}
        
        return Entity(
            entity_id=self.entity_id,
            name=name,
            type=entity_type,
            properties=merged_properties,
            aliases=merged_aliases,
            confidence=confidence,
            source=source,
            metadata=merged_metadata,
        )


@dataclass
class EntityList:
    """
    實體列表容器
    
    提供實體列表的便利操作方法。
    """
    entities: List[Entity] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.entities)
    
    def __iter__(self):
        return iter(self.entities)
    
    def __getitem__(self, index):
        return self.entities[index]
    
    def add(self, entity: Entity):
        """新增實體"""
        self.entities.append(entity)
    
    def get_by_id(self, entity_id: str) -> Optional[Entity]:
        """根據 ID 獲取實體"""
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        return None
    
    def get_by_type(self, entity_type: str) -> List[Entity]:
        """根據類型獲取實體列表"""
        return [e for e in self.entities if e.type == entity_type]
    
    def get_by_source(self, source: str) -> List[Entity]:
        """根據來源獲取實體列表"""
        return [e for e in self.entities if source in e.source]
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """轉換為字典列表"""
        return [e.to_dict() for e in self.entities]
    
    @classmethod
    def from_dict_list(cls, data_list: List[Dict[str, Any]]) -> "EntityList":
        """從字典列表創建"""
        entities = [Entity.from_dict(d) for d in data_list]
        return cls(entities=entities)
    
    def deduplicate(self, merge: bool = True) -> "EntityList":
        """
        去除重複實體
        
        Args:
            merge: 是否合併重複實體（True）或僅保留第一個（False）
        """
        seen = {}
        result = []
        
        for entity in self.entities:
            if entity.entity_id in seen:
                if merge:
                    # 合併實體
                    existing = seen[entity.entity_id]
                    merged = existing.merge_with(entity)
                    # 更新結果列表中的實體
                    for i, e in enumerate(result):
                        if e.entity_id == entity.entity_id:
                            result[i] = merged
                            break
                    seen[entity.entity_id] = merged
            else:
                seen[entity.entity_id] = entity
                result.append(entity)
        
        return EntityList(entities=result)
