"""
可插拔的檢索 Pipeline 架構

支援動態組合多個檢索增強元件，方便進行 Ablation Study
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import itertools


@dataclass
class RetrievalContext:
    """檢索上下文，在 pipeline 中傳遞"""
    query: str
    retrieved_entities: List[Any] = field(default_factory=list)
    retrieved_relations: List[Any] = field(default_factory=list)
    retrieved_chunks: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RetrievalComponent:
    """檢索元件基類"""
    
    def __init__(self, name: str, enabled: bool = True):
        """
        初始化檢索元件
        
        Args:
            name: 元件名稱
            enabled: 是否啟用
        """
        self.name = name
        self.enabled = enabled
        
    def process(self, context: RetrievalContext) -> RetrievalContext:
        """
        處理檢索上下文
        
        Args:
            context: 檢索上下文
            
        Returns:
            更新後的檢索上下文
        """
        if not self.enabled:
            return context
        return self._process_impl(context)
        
    def _process_impl(self, context: RetrievalContext) -> RetrievalContext:
        """子類實作的處理邏輯"""
        raise NotImplementedError(f"Component {self.name} must implement _process_impl")


class BaseRetriever(RetrievalComponent):
    """基礎檢索器（LightRAG 原始檢索）"""
    
    def __init__(self, lightrag_engine, mode: str = "hybrid", **kwargs):
        """
        初始化基礎檢索器
        
        Args:
            lightrag_engine: LightRAG 引擎實例
            mode: 檢索模式 (local/global/hybrid/mix)
            **kwargs: 其他檢索參數
        """
        super().__init__(name="base_retriever", enabled=True)  # 基礎檢索必須啟用
        self.lightrag_engine = lightrag_engine
        self.mode = mode
        self.kwargs = kwargs
        
    def _process_impl(self, context: RetrievalContext) -> RetrievalContext:
        """執行 LightRAG 原始檢索"""
        # 執行 LightRAG 檢索
        result = self.lightrag_engine.retrieve(
            context.query, 
            mode=self.mode,
            **self.kwargs
        )
        
        context.retrieved_entities = result.get("entities", [])
        context.retrieved_relations = result.get("relations", [])
        context.retrieved_chunks = result.get("chunks", [])
        
        # 記錄 metadata
        context.metadata["base_retrieval"] = {
            "mode": self.mode,
            "num_entities": len(context.retrieved_entities),
            "num_relations": len(context.retrieved_relations),
            "num_chunks": len(context.retrieved_chunks)
        }
        
        return context


class EntityDisambiguationComponent(RetrievalComponent):
    """實體消歧元件"""
    
    def __init__(self, disambiguator, enabled: bool = True):
        """
        初始化實體消歧元件
        
        Args:
            disambiguator: 實體消歧器實例
            enabled: 是否啟用
        """
        super().__init__(name="entity_disambiguation", enabled=enabled)
        self.disambiguator = disambiguator
        
    def _process_impl(self, context: RetrievalContext) -> RetrievalContext:
        """執行實體消歧"""
        if context.retrieved_entities:
            original_count = len(context.retrieved_entities)
            context.retrieved_entities = self.disambiguator.merge_entities(
                context.retrieved_entities
            )
            merged_count = len(context.retrieved_entities)
            
            context.metadata["entity_disambiguation"] = {
                "original_count": original_count,
                "merged_count": merged_count,
                "reduction_rate": (original_count - merged_count) / original_count if original_count > 0 else 0
            }
        
        return context


class RerankerComponent(RetrievalComponent):
    """Re-ranking 元件"""
    
    def __init__(self, reranker, top_k: int = 10, enabled: bool = True):
        """
        初始化 Re-ranking 元件
        
        Args:
            reranker: Re-ranker 實例
            top_k: 保留的 top-k 結果
            enabled: 是否啟用
        """
        super().__init__(name="reranker", enabled=enabled)
        self.reranker = reranker
        self.top_k = top_k
        
    def _process_impl(self, context: RetrievalContext) -> RetrievalContext:
        """執行 Re-ranking"""
        # Re-rank entities
        if context.retrieved_entities:
            entity_texts = [self._get_text(e) for e in context.retrieved_entities]
            ranked = self.reranker.rerank(context.query, entity_texts, self.top_k)
            context.retrieved_entities = [context.retrieved_entities[i] for i, _ in ranked]
            
        # Re-rank relations
        if context.retrieved_relations:
            relation_texts = [self._get_text(r) for r in context.retrieved_relations]
            ranked = self.reranker.rerank(context.query, relation_texts, self.top_k)
            context.retrieved_relations = [context.retrieved_relations[i] for i, _ in ranked]
            
        # Re-rank chunks
        if context.retrieved_chunks:
            chunk_texts = [self._get_text(c) for c in context.retrieved_chunks]
            ranked = self.reranker.rerank(context.query, chunk_texts, self.top_k)
            context.retrieved_chunks = [context.retrieved_chunks[i] for i, _ in ranked]
            
        context.metadata["reranking"] = {
            "top_k": self.top_k,
            "reranked_entities": len(context.retrieved_entities),
            "reranked_relations": len(context.retrieved_relations),
            "reranked_chunks": len(context.retrieved_chunks)
        }
        
        return context
    
    def _get_text(self, item) -> str:
        """從不同類型的物件中提取文本"""
        if hasattr(item, 'description'):
            return item.description
        elif hasattr(item, 'text'):
            return item.text
        elif hasattr(item, 'content'):
            return item.content
        else:
            return str(item)


class ToGIterativeRetriever(RetrievalComponent):
    """ToG 迭代式檢索器"""
    
    def __init__(self, tog_engine, max_iterations: int = 3, enabled: bool = True):
        """
        初始化 ToG 檢索器
        
        Args:
            tog_engine: ToG 引擎實例
            max_iterations: 最大迭代次數
            enabled: 是否啟用
        """
        super().__init__(name="tog_retriever", enabled=enabled)
        self.tog_engine = tog_engine
        self.max_iterations = max_iterations
        
    def _process_impl(self, context: RetrievalContext) -> RetrievalContext:
        """執行 ToG 迭代式檢索"""
        # 若已有基礎檢索結果，作為 ToG 的起點
        initial_entities = context.retrieved_entities or []
        
        tog_result = self.tog_engine.iterative_retrieve(
            query=context.query,
            initial_entities=initial_entities,
            max_iterations=self.max_iterations
        )
        
        # 合併結果
        context.retrieved_entities.extend(tog_result["entities"])
        context.retrieved_relations.extend(tog_result["relations"])
        
        context.metadata["tog_retrieval"] = {
            "iterations": tog_result.get("iterations", 0),
            "added_entities": len(tog_result["entities"]),
            "added_relations": len(tog_result["relations"])
        }
        
        return context


class RetrievalPipeline:
    """檢索 Pipeline，組合多個元件"""
    
    def __init__(self, components: List[RetrievalComponent]):
        """
        初始化檢索 Pipeline
        
        Args:
            components: 檢索元件列表
        """
        self.components = components
        
    def retrieve(self, query: str) -> RetrievalContext:
        """
        執行完整的檢索 pipeline
        
        Args:
            query: 查詢字串
            
        Returns:
            檢索上下文
        """
        context = RetrievalContext(query=query)
        
        for component in self.components:
            if component.enabled:
                print(f"🔧 執行元件: {component.name}")
                context = component.process(context)
                
        return context
        
    def get_ablation_configs(self) -> List[Dict[str, bool]]:
        """
        生成 Ablation Study 的配置組合
        
        Returns:
            配置列表，每個配置是一個 {component_name: enabled} 字典
        """
        # 取得所有可切換的元件（排除 base_retriever）
        toggleable = [c for c in self.components if c.name != "base_retriever"]
        
        configs = []
        # 生成所有可能的 on/off 組合
        for r in range(len(toggleable) + 1):
            for combo in itertools.combinations(toggleable, r):
                config = {c.name: (c in combo) for c in toggleable}
                configs.append(config)
                
        return configs
    
    def apply_config(self, config: Dict[str, bool]):
        """
        應用 Ablation Study 配置
        
        Args:
            config: 配置字典 {component_name: enabled}
        """
        for component in self.components:
            if component.name in config:
                component.enabled = config[component.name]
    
    def get_enabled_components(self) -> List[str]:
        """取得目前啟用的元件名稱列表"""
        return [c.name for c in self.components if c.enabled]
    
    def reset_components(self):
        """重設所有元件為預設啟用狀態（base_retriever 除外）"""
        for component in self.components:
            if component.name != "base_retriever":
                component.enabled = True
