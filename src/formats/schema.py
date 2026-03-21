"""
Schema 標準格式定義

定義了統一的 Schema 表示格式，支援不同 Schema Learning 方法的輸出。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum


class SchemaType(Enum):
    """Schema 類型"""
    FIXED = "fixed"  # 固定 schema（如 LightRAG 預定義）
    LEARNED = "learned"  # 學習到的 schema（如 AutoSchema）
    DYNAMIC = "dynamic"  # 動態演化 schema
    ONTOLOGY = "ontology"  # 本體驅動 schema


@dataclass
class EntityType:
    """
    實體類型定義
    
    Attributes:
        name: 類型名稱
        description: 類型描述
        properties: 該類型實體的屬性定義
        examples: 示例實體
        parent_types: 父類型（支援層級結構）
        constraints: 約束條件
    """
    name: str
    description: str = ""
    properties: Dict[str, str] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    parent_types: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "examples": self.examples,
            "parent_types": self.parent_types,
            "constraints": self.constraints,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityType":
        """從字典創建"""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            examples=data.get("examples", []),
            parent_types=data.get("parent_types", []),
            constraints=data.get("constraints", {}),
        )


@dataclass
class RelationType:
    """
    關係類型定義
    
    Attributes:
        name: 關係名稱
        description: 關係描述
        head_types: 允許的頭實體類型
        tail_types: 允許的尾實體類型
        properties: 該關係的屬性定義
        is_symmetric: 是否對稱
        inverse_relation: 反向關係名稱
    """
    name: str
    description: str = ""
    head_types: List[str] = field(default_factory=list)
    tail_types: List[str] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)
    is_symmetric: bool = False
    inverse_relation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "head_types": self.head_types,
            "tail_types": self.tail_types,
            "properties": self.properties,
            "is_symmetric": self.is_symmetric,
            "inverse_relation": self.inverse_relation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationType":
        """從字典創建"""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            head_types=data.get("head_types", []),
            tail_types=data.get("tail_types", []),
            properties=data.get("properties", {}),
            is_symmetric=data.get("is_symmetric", False),
            inverse_relation=data.get("inverse_relation"),
        )


@dataclass
class Schema:
    """
    標準 Schema 格式
    
    統一的 Schema 表示，支援不同 Schema Learning 方法的輸出。
    
    Attributes:
        entity_types: 實體類型列表
        relation_types: 關係類型列表
        schema_type: Schema 類型
        version: Schema 版本號
        metadata: Schema 元資料
    
    Examples:
        >>> entity_types = [
        ...     EntityType(name="Person", description="人物"),
        ...     EntityType(name="Organization", description="組織")
        ... ]
        >>> relation_types = [
        ...     RelationType(name="works_at", head_types=["Person"], tail_types=["Organization"])
        ... ]
        >>> schema = Schema(entity_types=entity_types, relation_types=relation_types)
    """
    entity_types: List[EntityType] = field(default_factory=list)
    relation_types: List[RelationType] = field(default_factory=list)
    schema_type: SchemaType = SchemaType.FIXED
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_entity_type(self, name: str) -> Optional[EntityType]:
        """根據名稱獲取實體類型"""
        for et in self.entity_types:
            if et.name == name:
                return et
        return None
    
    def get_relation_type(self, name: str) -> Optional[RelationType]:
        """根據名稱獲取關係類型"""
        for rt in self.relation_types:
            if rt.name == name:
                return rt
        return None
    
    def has_entity_type(self, name: str) -> bool:
        """檢查是否包含指定實體類型"""
        return any(et.name == name for et in self.entity_types)
    
    def has_relation_type(self, name: str) -> bool:
        """檢查是否包含指定關係類型"""
        return any(rt.name == name for rt in self.relation_types)
    
    def add_entity_type(self, entity_type: EntityType):
        """新增實體類型"""
        if not self.has_entity_type(entity_type.name):
            self.entity_types.append(entity_type)
    
    def add_relation_type(self, relation_type: RelationType):
        """新增關係類型"""
        if not self.has_relation_type(relation_type.name):
            self.relation_types.append(relation_type)
    
    def validate_entity(self, entity_type: str) -> bool:
        """驗證實體類型是否符合 schema"""
        if self.schema_type == SchemaType.DYNAMIC:
            # 動態 schema 接受任何類型
            return True
        return self.has_entity_type(entity_type)
    
    def validate_relation(self, relation_type: str, head_type: str, tail_type: str) -> bool:
        """驗證關係是否符合 schema"""
        if self.schema_type == SchemaType.DYNAMIC:
            # 動態 schema 接受任何關係
            return True
        
        rt = self.get_relation_type(relation_type)
        if rt is None:
            return False
        
        # 檢查頭尾類型是否允許
        if rt.head_types and head_type not in rt.head_types:
            return False
        if rt.tail_types and tail_type not in rt.tail_types:
            return False
        
        return True
    
    def get_entity_type_names(self) -> List[str]:
        """獲取所有實體類型名稱"""
        return [et.name for et in self.entity_types]
    
    def get_relation_type_names(self) -> List[str]:
        """獲取所有關係類型名稱"""
        return [rt.name for rt in self.relation_types]
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "entity_types": [et.to_dict() for et in self.entity_types],
            "relation_types": [rt.to_dict() for rt in self.relation_types],
            "schema_type": self.schema_type.value,
            "version": self.version,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """從字典創建 Schema"""
        entity_types = [EntityType.from_dict(et) for et in data.get("entity_types", [])]
        relation_types = [RelationType.from_dict(rt) for rt in data.get("relation_types", [])]
        schema_type = SchemaType(data.get("schema_type", "fixed"))
        
        return cls(
            entity_types=entity_types,
            relation_types=relation_types,
            schema_type=schema_type,
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )
    
    def merge(self, other: "Schema") -> "Schema":
        """
        合併兩個 Schema
        
        去重並合併實體類型和關係類型。
        """
        # 合併實體類型（去重）
        entity_type_names = set(et.name for et in self.entity_types)
        merged_entity_types = list(self.entity_types)
        for et in other.entity_types:
            if et.name not in entity_type_names:
                merged_entity_types.append(et)
                entity_type_names.add(et.name)
        
        # 合併關係類型（去重）
        relation_type_names = set(rt.name for rt in self.relation_types)
        merged_relation_types = list(self.relation_types)
        for rt in other.relation_types:
            if rt.name not in relation_type_names:
                merged_relation_types.append(rt)
                relation_type_names.add(rt.name)
        
        # 確定合併後的 schema 類型
        if self.schema_type == SchemaType.DYNAMIC or other.schema_type == SchemaType.DYNAMIC:
            merged_schema_type = SchemaType.DYNAMIC
        elif self.schema_type == SchemaType.LEARNED or other.schema_type == SchemaType.LEARNED:
            merged_schema_type = SchemaType.LEARNED
        else:
            merged_schema_type = self.schema_type
        
        return Schema(
            entity_types=merged_entity_types,
            relation_types=merged_relation_types,
            schema_type=merged_schema_type,
            version=f"{self.version}+{other.version}",
            metadata={**self.metadata, **other.metadata, "is_merged": True},
        )
    
    def to_lightrag_format(self) -> List[str]:
        """
        轉換為 LightRAG 格式的實體類型列表
        
        LightRAG 使用簡單的字串列表作為 entity_types。
        """
        return self.get_entity_type_names()
    
    @classmethod
    def from_lightrag_types(cls, entity_types: List[str]) -> "Schema":
        """
        從 LightRAG 的實體類型列表創建 Schema
        
        Args:
            entity_types: LightRAG 的實體類型字串列表
        """
        entity_type_objects = [EntityType(name=et) for et in entity_types]
        return cls(
            entity_types=entity_type_objects,
            schema_type=SchemaType.FIXED,
            metadata={"source": "lightrag"},
        )
