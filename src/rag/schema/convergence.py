"""
Schema 收斂機制

評估 Schema 品質並判斷是否需要繼續演化，
避免 Schema 無限膨脹
"""

from typing import Dict, List, Any, Set
import numpy as np


class SchemaQualityMetrics:
    """Schema 品質評估指標"""
    
    def __init__(
        self,
        max_entity_types: int = 50,
        convergence_window: int = 3,
        min_improvement: float = 0.05
    ):
        """
        初始化 Schema 品質評估器
        
        Args:
            max_entity_types: 實體類型數量上限
            convergence_window: 收斂判斷視窗大小
            min_improvement: 最小改進幅度
        """
        self.max_entity_types = max_entity_types
        self.convergence_window = convergence_window
        self.min_improvement = min_improvement
        
        # 記錄歷史指標
        self.history = {
            "coverage": [],
            "specificity": [],
            "entity_count": []
        }
        
        print(f"🔧 初始化 Schema 收斂機制")
        print(f"   - 最大實體類型數: {max_entity_types}")
        print(f"   - 收斂視窗: {convergence_window}")
        print(f"   - 最小改進幅度: {min_improvement}")
    
    def calculate_coverage(
        self, 
        schema: Dict[str, Any], 
        entities_in_corpus: List[str]
    ) -> float:
        """
        計算 Schema 覆蓋率
        
        Schema 實體類型對文本實體的覆蓋率
        
        Args:
            schema: Schema 字典
            entities_in_corpus: 文本中的實體列表
            
        Returns:
            覆蓋率 [0, 1]
        """
        if not entities_in_corpus:
            return 0.0
            
        schema_types = set(schema.get("entities", []))
        corpus_entity_types = set(entities_in_corpus)
        
        if not corpus_entity_types:
            return 0.0
        
        # 計算有多少文本實體類型被 Schema 覆蓋
        covered = corpus_entity_types & schema_types
        coverage = len(covered) / len(corpus_entity_types)
        
        return coverage
    
    def calculate_specificity(self, schema: Dict[str, Any]) -> float:
        """
        計算 Schema 具體性
        
        實體類型的具體性（避免過於抽象）
        使用實體類型的平均詞數作為具體性指標
        
        Args:
            schema: Schema 字典
            
        Returns:
            具體性分數 [0, 1]
        """
        entities = schema.get("entities", [])
        
        if not entities:
            return 0.0
        
        # 計算平均詞數（具體的實體類型通常有更多詞）
        word_counts = [len(str(e).split()) for e in entities]
        avg_words = np.mean(word_counts)
        
        # 歸一化到 [0, 1]
        # 假設 1 詞為最不具體（0），3+ 詞為非常具體（1）
        specificity = min(avg_words / 3.0, 1.0)
        
        return specificity
    
    def calculate_coherence(self, schema: Dict[str, Any]) -> float:
        """
        計算 Schema 一致性
        
        評估實體類型和關係之間的邏輯一致性
        
        Args:
            schema: Schema 字典
            
        Returns:
            一致性分數 [0, 1]
        """
        entities = set(schema.get("entities", []))
        relations = schema.get("relations", [])
        validation_schema = schema.get("validation_schema", {})
        
        if not entities:
            return 0.0
        
        # 檢查關係中的實體是否在實體列表中
        valid_relations = 0
        for relation in relations:
            # 假設關係格式為 {source, relation, target}
            if isinstance(relation, dict):
                source = relation.get("source_type")
                target = relation.get("target_type")
                if source in entities and target in entities:
                    valid_relations += 1
        
        coherence = valid_relations / len(relations) if relations else 1.0
        
        return coherence
    
    def should_continue_evolution(
        self, 
        current_schema: Dict[str, Any],
        entities_in_corpus: List[str] = None
    ) -> Tuple[bool, str]:
        """
        判斷是否應該繼續演化
        
        Args:
            current_schema: 當前 Schema
            entities_in_corpus: 文本中的實體（可選）
            
        Returns:
            (should_continue, reason)
        """
        # 1. 檢查實體數量上限
        entity_count = len(current_schema.get("entities", []))
        if entity_count >= self.max_entity_types:
            return False, f"達到實體類型上限 ({entity_count}/{self.max_entity_types})"
        
        # 2. 計算當前指標
        current_coverage = 0.0
        if entities_in_corpus:
            current_coverage = self.calculate_coverage(current_schema, entities_in_corpus)
        
        current_specificity = self.calculate_specificity(current_schema)
        
        # 記錄歷史
        self.history["coverage"].append(current_coverage)
        self.history["specificity"].append(current_specificity)
        self.history["entity_count"].append(entity_count)
        
        # 3. 檢查收斂條件
        if len(self.history["coverage"]) >= self.convergence_window:
            recent_coverage = self.history["coverage"][-self.convergence_window:]
            recent_entity_count = self.history["entity_count"][-self.convergence_window:]
            
            # 條件1: Coverage 增長小於閾值
            coverage_improvement = max(recent_coverage) - min(recent_coverage)
            if coverage_improvement < self.min_improvement:
                return False, f"Coverage 增長過小 ({coverage_improvement:.3f} < {self.min_improvement})"
            
            # 條件2: 新增實體類型過少
            entity_growth = max(recent_entity_count) - min(recent_entity_count)
            if entity_growth < 2:
                return False, f"新增實體類型過少 ({entity_growth} < 2)"
        
        # 4. 繼續演化
        return True, f"繼續演化 (實體數: {entity_count}, Coverage: {current_coverage:.3f})"
    
    def get_summary(self) -> Dict[str, Any]:
        """
        取得 Schema 演化摘要
        
        Returns:
            摘要字典
        """
        return {
            "total_iterations": len(self.history["entity_count"]),
            "final_entity_count": self.history["entity_count"][-1] if self.history["entity_count"] else 0,
            "final_coverage": self.history["coverage"][-1] if self.history["coverage"] else 0.0,
            "final_specificity": self.history["specificity"][-1] if self.history["specificity"] else 0.0,
            "history": self.history
        }
    
    def reset(self):
        """重設歷史記錄"""
        self.history = {
            "coverage": [],
            "specificity": [],
            "entity_count": []
        }


def should_stop_schema_evolution(
    current_schema: Dict[str, Any],
    previous_schema: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Tuple[bool, str]:
    """
    簡化版本的收斂判斷函數
    
    Args:
        current_schema: 當前 Schema
        previous_schema: 前一個 Schema
        config: 配置參數
        
    Returns:
        (should_stop, reason)
    """
    if config is None:
        config = {
            "max_entity_types": 50,
            "min_new_entities": 2
        }
    
    current_entities = set(current_schema.get("entities", []))
    previous_entities = set(previous_schema.get("entities", []))
    
    # 檢查是否達到上限
    if len(current_entities) >= config.get("max_entity_types", 50):
        return True, "達到實體類型上限"
    
    # 檢查新增實體數量
    new_entities = current_entities - previous_entities
    if len(new_entities) < config.get("min_new_entities", 2):
        return True, f"新增實體類型過少 ({len(new_entities)} < 2)"
    
    return False, "繼續演化"
