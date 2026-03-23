"""
DynamicLLMPathExtractor 插件

基於 LlamaIndex DynamicLLMPathExtractor 的動態實體類型檢測插件。
透過 PropertyGraphExtractorFactory 整合，提供 optional ontology 機制和動態三元組推斷。

DEPRECATED:
目前主流程未接線本插件；建議改用 schema factory 的 llamaindex_dynamic
與 property_graph dynamic extractor 作為統一路徑。
"""

from typing import List, Dict, Any, Optional
from src.plugins.base import BaseKGPlugin
from src.plugins.registry import register_plugin


@register_plugin("dynamic_path")
class DynamicPathPlugin(BaseKGPlugin):
    """
    DynamicLLMPathExtractor 插件
    
    核心功能：
    - 動態實體類型檢測（不限於預定義類型）
    - LLM 推斷實體和關係
    - 靈活 Schema（可擴展初始本體）
    
    參考：LlamaIndex 官方文檔
    """
    
    def __init__(self):
        super().__init__()
        self._extractor = None
        self._discovered_types: Dict[str, set] = {
            "entity_types": set(),
            "relation_types": set(),
        }
    
    def get_name(self) -> str:
        return "DynamicPath"
    
    def get_description(self) -> str:
        return "動態實體類型檢測與靈活 Schema 擴展插件"
    
    def _get_or_create_extractor(self, llm=None, **kwargs):
        """取得或建立 DynamicLLMPathExtractor 實例"""
        if self._extractor is not None:
            return self._extractor
        try:
            from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
        except ImportError:
            print("⚠️  [DynamicPath] 無法匯入 DynamicLLMPathExtractor，需安裝 llama-index-core")
            return None

        if llm is None:
            try:
                from llama_index.core import Settings as LlamaSettings
                llm = getattr(LlamaSettings, "llm", None)
            except ImportError:
                pass
        if llm is None:
            print("⚠️  [DynamicPath] 無可用 LLM，無法建立 extractor")
            return None

        allowed_entity_types = kwargs.get("allowed_entity_types", [])
        allowed_relation_types = kwargs.get("allowed_relation_types", [])
        max_triplets = kwargs.get("max_triplets_per_chunk", 20)
        num_workers = kwargs.get("num_workers", 4)

        ext_kwargs: Dict[str, Any] = {
            "llm": llm,
            "max_triplets_per_chunk": max_triplets,
            "num_workers": num_workers,
        }
        if allowed_entity_types:
            ext_kwargs["allowed_entity_types"] = allowed_entity_types
        if allowed_relation_types:
            ext_kwargs["allowed_relation_types"] = allowed_relation_types

        self._extractor = DynamicLLMPathExtractor(**ext_kwargs)
        print(f"✅ [DynamicPath] 建立 DynamicLLMPathExtractor "
              f"(max_triplets={max_triplets}, workers={num_workers})")
        return self._extractor

    def enhance_schema(
        self,
        text_corpus: List[str],
        base_schema: List[str],
        **kwargs
    ) -> List[str]:
        """
        Optional ontology 機制：以 base_schema 作為初始實體類型提示，
        同時允許 LLM 在抽取時動態擴展新類型。
        
        回傳的 schema 會合併已知類型和之前發現的動態類型。
        """
        print(f"🔍 [DynamicPath] 啟用動態 Schema 擴展...")
        
        enhanced = list(base_schema)
        if self._discovered_types["entity_types"]:
            new_types = self._discovered_types["entity_types"] - set(base_schema)
            if new_types:
                enhanced.extend(sorted(new_types))
                print(f"🔍 [DynamicPath] 合併 {len(new_types)} 個先前發現的動態類型")
        
        print(f"🔍 [DynamicPath] Schema 共 {len(enhanced)} 個實體類型（含動態擴展）")
        return enhanced
    
    def enhance_extraction(
        self,
        text: str,
        base_triples: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        使用 DynamicLLMPathExtractor 從文本動態推斷三元組，
        與 base_triples 合併後回傳。
        """
        llm = kwargs.get("llm")
        extractor = self._get_or_create_extractor(
            llm=llm,
            allowed_entity_types=kwargs.get("allowed_entity_types", []),
            allowed_relation_types=kwargs.get("allowed_relation_types", []),
            max_triplets_per_chunk=kwargs.get("max_triplets_per_chunk", 20),
            num_workers=kwargs.get("num_workers", 4),
        )
        if extractor is None:
            return base_triples

        try:
            from llama_index.core.schema import TextNode
            node = TextNode(text=text)
            result_nodes = extractor.extract(
                [node], show_progress=False
            )
            dynamic_triples = []
            for rn in result_nodes:
                metadata = getattr(rn, "metadata", {})
                kg_rels = metadata.get("kg_rel_texts", [])
                for rel_text in kg_rels:
                    if isinstance(rel_text, str) and " -> " in rel_text:
                        parts = rel_text.split(" -> ", 2)
                        if len(parts) == 3:
                            triple = {
                                "subject": parts[0].strip(),
                                "predicate": parts[1].strip(),
                                "object": parts[2].strip(),
                                "source": "dynamic_path",
                            }
                            dynamic_triples.append(triple)
                            self._discovered_types["entity_types"].add(
                                parts[0].strip().split(":")[-1] if ":" in parts[0] else ""
                            )
            
            if dynamic_triples:
                print(f"🔍 [DynamicPath] 動態推斷 {len(dynamic_triples)} 個三元組")
            
            return base_triples + dynamic_triples
            
        except Exception as e:
            print(f"⚠️  [DynamicPath] 動態抽取失敗: {e}")
            return base_triples
    
    def post_process_graph(
        self,
        graph_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """合併推斷的新類型到 graph_data 的 schema_info"""
        if self._discovered_types["entity_types"]:
            schema_info = graph_data.get("schema_info", {})
            existing = set(schema_info.get("entities", []))
            new_types = self._discovered_types["entity_types"] - existing - {""}
            if new_types:
                schema_info.setdefault("entities", []).extend(sorted(new_types))
                graph_data["schema_info"] = schema_info
                print(f"🔍 [DynamicPath] 後處理：新增 {len(new_types)} 個動態發現的實體類型")
        return graph_data
