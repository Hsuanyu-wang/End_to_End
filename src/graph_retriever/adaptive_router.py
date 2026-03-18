"""
Adaptive Query Router

自動分析查詢特性並選擇最佳檢索策略
"""

from typing import Dict, Any, Optional
from enum import Enum


class QueryComplexity(Enum):
    """查詢複雜度"""
    SINGLE_HOP = "single_hop"  # 單跳查詢
    MULTI_HOP = "multi_hop"    # 多跳查詢
    AGGREGATION = "aggregation"  # 聚合查詢


class RetrievalMode(Enum):
    """檢索模式"""
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"
    TOG = "tog"


class QueryRouter:
    """查詢路由器"""
    
    def __init__(
        self,
        llm=None,
        fallback_mode: str = "hybrid",
        use_llm_analysis: bool = True
    ):
        """
        初始化查詢路由器
        
        Args:
            llm: LLM 實例
            fallback_mode: 備用模式
            use_llm_analysis: 是否使用 LLM 分析
        """
        self.llm = llm
        self.fallback_mode = fallback_mode
        self.use_llm_analysis = use_llm_analysis
        
        print(f"🔧 初始化查詢路由器 (fallback={fallback_mode})")
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        路由查詢到最適合的檢索策略
        
        Args:
            query: 查詢字串
            
        Returns:
            路由結果 {"mode": str, "analysis": dict, "confidence": float}
        """
        # 分析查詢
        analysis = self.analyze_query(query)
        
        # 選擇模式
        mode = self.select_mode(analysis)
        
        return {
            "mode": mode,
            "analysis": analysis,
            "confidence": analysis.get("confidence", 0.5)
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析查詢特性
        
        Args:
            query: 查詢字串
            
        Returns:
            分析結果字典
        """
        if self.use_llm_analysis and self.llm:
            return self._llm_analysis(query)
        else:
            return self._rule_based_analysis(query)
    
    def _llm_analysis(self, query: str) -> Dict[str, Any]:
        """使用 LLM 分析查詢"""
        try:
            prompt = f"""
            分析以下查詢的特性:
            
            查詢: {query}
            
            請判斷:
            1. complexity: single_hop / multi_hop / aggregation
               - single_hop: 查詢單一實體或事實
               - multi_hop: 需要連結多個實體或關係
               - aggregation: 需要統計、總結或聚合資訊
            
            2. temporal: yes / no
               - 是否涉及時間相關的查詢（如「最近」、「2024年」等）
            
            3. entity_count: 估計涉及的實體數量 (1-10)
            
            只回答 JSON 格式:
            {{
                "complexity": "...",
                "temporal": "...",
                "entity_count": ...,
                "confidence": 0.0-1.0
            }}
            """
            
            response = self.llm.complete(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # 解析 JSON
            import json
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                return self._rule_based_analysis(query)
                
        except Exception as e:
            print(f"⚠️  LLM 分析失敗: {e}，使用規則式分析")
            return self._rule_based_analysis(query)
    
    def _rule_based_analysis(self, query: str) -> Dict[str, Any]:
        """基於規則的查詢分析（Fallback）"""
        query_lower = query.lower()
        
        # 判斷複雜度
        complexity = QueryComplexity.SINGLE_HOP
        
        # Multi-hop 關鍵詞
        multi_hop_keywords = [
            "如何", "怎麼", "為什麼", "關係", "影響", "導致",
            "how", "why", "relationship", "affect", "cause"
        ]
        if any(kw in query_lower for kw in multi_hop_keywords):
            complexity = QueryComplexity.MULTI_HOP
        
        # Aggregation 關鍵詞
        aggregation_keywords = [
            "多少", "總共", "統計", "列出", "所有",
            "how many", "total", "list all", "summarize"
        ]
        if any(kw in query_lower for kw in aggregation_keywords):
            complexity = QueryComplexity.AGGREGATION
        
        # 判斷是否有時間性
        temporal_keywords = [
            "最近", "今年", "去年", "2024", "2025", "2026",
            "recent", "this year", "last year"
        ]
        temporal = any(kw in query_lower for kw in temporal_keywords)
        
        # 估計實體數量（基於分詞）
        entity_count = min(len(query.split()), 10)
        
        return {
            "complexity": complexity.value,
            "temporal": "yes" if temporal else "no",
            "entity_count": entity_count,
            "confidence": 0.7  # 規則式分析的置信度較低
        }
    
    def select_mode(self, analysis: Dict[str, Any]) -> str:
        """
        根據分析結果選擇檢索模式
        
        Args:
            analysis: 查詢分析結果
            
        Returns:
            檢索模式字串
        """
        complexity = analysis.get("complexity", "single_hop")
        temporal = analysis.get("temporal", "no")
        entity_count = analysis.get("entity_count", 1)
        
        # 決策邏輯
        if complexity == "single_hop" and entity_count <= 2:
            # 簡單查詢 → local
            return RetrievalMode.LOCAL.value
            
        elif complexity == "multi_hop" or entity_count > 5:
            # 多跳查詢 → tog (若支援) 或 hybrid
            return RetrievalMode.TOG.value
            
        elif complexity == "aggregation":
            # 聚合查詢 → global
            return RetrievalMode.GLOBAL.value
            
        else:
            # 預設 → hybrid
            return RetrievalMode.HYBRID.value
    
    def explain_routing(self, query: str) -> str:
        """
        解釋路由決策
        
        Args:
            query: 查詢字串
            
        Returns:
            解釋文字
        """
        result = self.route(query)
        analysis = result["analysis"]
        mode = result["mode"]
        
        explanation = f"""
        查詢路由分析:
        - 查詢: {query}
        - 複雜度: {analysis.get('complexity')}
        - 時間性: {analysis.get('temporal')}
        - 預估實體數: {analysis.get('entity_count')}
        - 選擇模式: {mode}
        - 置信度: {result.get('confidence', 0.0):.2f}
        
        決策理由:
        """
        
        if mode == "local":
            explanation += "簡單的單實體查詢，使用 local 模式可以快速定位相關資訊。"
        elif mode == "global":
            explanation += "需要聚合或總結資訊，使用 global 模式獲取整體視角。"
        elif mode == "hybrid":
            explanation += "中等複雜度查詢，使用 hybrid 模式平衡精確度和覆蓋範圍。"
        elif mode == "tog":
            explanation += "多跳推理查詢，使用 ToG 迭代檢索深入探索關聯。"
        
        return explanation
