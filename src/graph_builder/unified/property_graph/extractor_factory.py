"""
PropertyGraph Extractor Factory

支援組合多個 LlamaIndex PropertyGraph extractors
"""

from typing import List, Dict, Any


class PropertyGraphExtractorFactory:
    """
    PropertyGraph 專用的 Extractor Factory
    
    支援組合多個 LlamaIndex extractors:
    - ImplicitPathExtractor: 提取隱式結構關係
    - SchemaLLMPathExtractor: 基於 Schema 的關係抽取
    - SimpleLLMPathExtractor: 簡單 LLM 關係抽取
    - DynamicLLMPathExtractor: 動態 Schema 推斷抽取
    """
    
    @staticmethod
    def create_extractors(
        settings: Any,
        extractor_config: Dict[str, Any]
    ) -> List[Any]:
        """
        根據配置建立多個 PropertyGraph extractors
        
        Args:
            settings: LlamaIndex Settings（需要 builder_llm）
            extractor_config: Extractor 配置字典
                {
                    "implicit": {"enabled": true},
                    "schema": {
                        "enabled": true,
                        "entities": [...],
                        "relations": [...],
                        "strict": false
                    },
                    "simple": {"enabled": true, "max_paths_per_chunk": 10},
                    "dynamic": {"enabled": true, "max_triplets_per_chunk": 20}
                }
        
        Returns:
            Extractor 實例列表
        """
        extractors = []
        
        if not extractor_config:
            print("⚠️  未提供 extractor_config，將使用預設 extractors")
            extractor_config = {
                "implicit": {"enabled": True},
                "simple": {"enabled": True}
            }
        
        # 1. ImplicitPathExtractor
        if extractor_config.get("implicit", {}).get("enabled", False):
            try:
                from llama_index.core.indices.property_graph import ImplicitPathExtractor
                extractor = ImplicitPathExtractor()
                extractors.append(extractor)
                print("✅ 已啟用 ImplicitPathExtractor")
            except ImportError as e:
                print(f"⚠️  ImplicitPathExtractor 匯入失敗: {e}")
        
        # 2. SchemaLLMPathExtractor
        if extractor_config.get("schema", {}).get("enabled", False):
            try:
                from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
                schema_cfg = extractor_config["schema"]
                
                entities = schema_cfg.get("entities", [])
                relations = schema_cfg.get("relations", [])
                validation_schema = schema_cfg.get("validation_schema")
                strict = schema_cfg.get("strict", False)
                
                if not entities or not relations:
                    print("⚠️  Schema extractor 需要 entities 和 relations，跳過")
                else:
                    llm = getattr(settings, 'builder_llm', None) or getattr(settings, 'llm', None)
                    if not llm:
                        print("⚠️  Schema extractor 需要 LLM，跳過")
                    else:
                        extractor = SchemaLLMPathExtractor(
                            llm=llm,
                            possible_entities=entities,
                            possible_relations=relations,
                            kg_validation_schema=validation_schema,
                            strict=strict
                        )
                        extractors.append(extractor)
                        print(f"✅ 已啟用 SchemaLLMPathExtractor (entities={len(entities)}, relations={len(relations)}, strict={strict})")
            
            except ImportError as e:
                print(f"⚠️  SchemaLLMPathExtractor 匯入失敗: {e}")
        
        # 3. SimpleLLMPathExtractor
        if extractor_config.get("simple", {}).get("enabled", False):
            try:
                from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
                simple_cfg = extractor_config["simple"]
                
                llm = getattr(settings, 'builder_llm', None) or getattr(settings, 'llm', None)
                if not llm:
                    print("⚠️  Simple extractor 需要 LLM，跳過")
                else:
                    max_paths_per_chunk = simple_cfg.get("max_paths_per_chunk", 10)
                    num_workers = simple_cfg.get("num_workers", 4)
                    
                    extractor = SimpleLLMPathExtractor(
                        llm=llm,
                        max_paths_per_chunk=max_paths_per_chunk,
                        num_workers=num_workers
                    )
                    extractors.append(extractor)
                    print(f"✅ 已啟用 SimpleLLMPathExtractor (max_paths={max_paths_per_chunk}, workers={num_workers})")
            
            except ImportError as e:
                print(f"⚠️  SimpleLLMPathExtractor 匯入失敗: {e}")
        
        # 4. DynamicLLMPathExtractor
        if extractor_config.get("dynamic", {}).get("enabled", False):
            try:
                from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
                dynamic_cfg = extractor_config["dynamic"]
                
                llm = getattr(settings, 'builder_llm', None) or getattr(settings, 'llm', None)
                if not llm:
                    print("⚠️  Dynamic extractor 需要 LLM，跳過")
                else:
                    max_triplets_per_chunk = dynamic_cfg.get("max_triplets_per_chunk", 20)
                    num_workers = dynamic_cfg.get("num_workers", 4)
                    allowed_entity_types = dynamic_cfg.get("allowed_entity_types", [])
                    allowed_relation_types = dynamic_cfg.get("allowed_relation_types", [])
                    
                    # 建立 kwargs
                    kwargs = {
                        "llm": llm,
                        "max_triplets_per_chunk": max_triplets_per_chunk,
                        "num_workers": num_workers
                    }
                    
                    # 只有在有指定類型時才加入
                    if allowed_entity_types:
                        kwargs["allowed_entity_types"] = allowed_entity_types
                    if allowed_relation_types:
                        kwargs["allowed_relation_types"] = allowed_relation_types
                    
                    extractor = DynamicLLMPathExtractor(**kwargs)
                    extractors.append(extractor)
                    print(f"✅ 已啟用 DynamicLLMPathExtractor (max_triplets={max_triplets_per_chunk}, workers={num_workers})")
            
            except ImportError as e:
                print(f"⚠️  DynamicLLMPathExtractor 匯入失敗: {e}")
        
        if not extractors:
            print("⚠️  未建立任何 extractor，PropertyGraph 建圖可能失敗")
        else:
            print(f"📊 總共建立了 {len(extractors)} 個 extractors")
        
        return extractors
