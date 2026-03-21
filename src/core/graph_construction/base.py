"""
圖譜建構組件抽象基類

定義了圖譜建構器的統一介面，所有圖譜建構實作都應繼承此基類。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from llama_index.core.schema import Document

# 從頂層 formats 模組導入
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from formats import Entity, EntityList, Relation, RelationList, Schema, Graph, GraphData


class BaseGraphConstructor(ABC):
    """
    圖譜建構器抽象基類
    
    所有圖譜建構組件（LightRAG、NetworkX、Neo4j、PropertyGraph）都應實作此介面。
    
    關鍵設計原則：
    1. 輸入：實體 + 關係 + Schema
    2. 輸出：標準 GraphData 格式
    3. 支援不同的圖譜儲存格式
    
    Examples:
        >>> class LightRAGGraphConstructor(BaseGraphConstructor):
        ...     def build(self, entities, relations, schema=None):
        ...         # 建構 LightRAG 格式的圖譜
        ...         return GraphData(...)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化圖譜建構器
        
        Args:
            config: 配置字典（如儲存路徑、資料庫連線等）
        """
        self.config = config or {}
        self._initialize()
    
    def _initialize(self):
        """子類可覆寫此方法進行初始化"""
        pass
    
    @abstractmethod
    def build(
        self,
        entities: EntityList,
        relations: RelationList,
        schema: Optional[Schema] = None,
        documents: Optional[List[Document]] = None,
        **kwargs
    ) -> GraphData:
        """
        建構圖譜
        
        這是核心抽象方法，所有子類必須實作。
        
        Args:
            entities: 實體列表
            relations: 關係列表
            schema: 可選的 Schema（用於驗證和約束）
            documents: 可選的原始文件（某些方法可能需要）
            **kwargs: 額外參數
        
        Returns:
            GraphData: 建構的圖譜資料
        
        Notes:
            - 應驗證實體和關係的一致性
            - 可以包含圖譜優化（如去重、連通性檢查等）
            - 輸出的 GraphData 應包含標準 Graph 格式
        """
        pass
    
    def validate_graph(
        self,
        entities: EntityList,
        relations: RelationList
    ) -> List[str]:
        """
        驗證圖譜的一致性
        
        Args:
            entities: 實體列表
            relations: 關係列表
        
        Returns:
            List[str]: 錯誤訊息列表，空列表表示驗證通過
        """
        errors = []
        entity_ids = set(e.entity_id for e in entities)
        
        # 檢查關係是否指向存在的實體
        for relation in relations:
            if relation.head not in entity_ids:
                errors.append(f"Relation {relation.relation_id} head {relation.head} not found in entities")
            if relation.tail not in entity_ids:
                errors.append(f"Relation {relation.relation_id} tail {relation.tail} not found in entities")
        
        # 檢查是否有孤立節點（可選，某些場景允許）
        connected_entities = set()
        for relation in relations:
            connected_entities.add(relation.head)
            connected_entities.add(relation.tail)
        
        isolated = entity_ids - connected_entities
        if isolated and self.config.get("warn_isolated_nodes", False):
            errors.append(f"Found {len(isolated)} isolated nodes: {list(isolated)[:5]}...")
        
        return errors
    
    def optimize_graph(
        self,
        graph: Graph
    ) -> Graph:
        """
        優化圖譜
        
        預設實作：去重。子類可覆寫以提供更多優化。
        
        Args:
            graph: 原始圖譜
        
        Returns:
            Graph: 優化後的圖譜
        """
        # 去重實體和關係
        optimized_entities = graph.entities.deduplicate(merge=True)
        optimized_relations = graph.relations.deduplicate()
        
        return Graph(
            entities=optimized_entities,
            relations=optimized_relations,
            metadata={**graph.metadata, "optimized": True}
        )
    
    def get_name(self) -> str:
        """
        獲取建構器名稱
        
        Returns:
            str: 建構器名稱
        """
        return self.__class__.__name__
    
    def get_config(self) -> Dict[str, Any]:
        """獲取當前配置"""
        return self.config.copy()
    
    def save(self, graph_data: GraphData, path: str) -> bool:
        """
        儲存圖譜
        
        預設實作：儲存為 JSON。子類可覆寫以支援其他格式。
        
        Args:
            graph_data: 要儲存的圖譜資料
            path: 儲存路徑
        
        Returns:
            bool: 是否成功
        """
        import json
        from pathlib import Path
        
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data.to_dict(), f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save graph: {e}")
            return False
    
    def load(self, path: str) -> Optional[GraphData]:
        """
        載入圖譜
        
        預設實作：從 JSON 載入。子類可覆寫以支援其他格式。
        
        Args:
            path: 載入路徑
        
        Returns:
            Optional[GraphData]: 載入的圖譜資料，失敗則返回 None
        """
        import json
        from pathlib import Path
        
        try:
            with open(Path(path), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return GraphData.from_dict(data)
        except Exception as e:
            print(f"Failed to load graph: {e}")
            return None
