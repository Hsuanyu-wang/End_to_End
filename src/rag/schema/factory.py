import json
from openai import OpenAI
from llama_index.core.schema import TextNode

# 1. 引用 extract_schema_only 內容
from src.rag.schema.evolution import (
    evolve_schema_with_pydantic,
    run_iterative_evolution,
    run_anchored_additive_evolution,
    bootstrap_domain_schema,
    consolidate_schema,
)
from src.rag.schema.schema_adaptive_consolidation import learn_schema_adaptively
from src.rag.schema.onto_learner import learn_entity_types as onto_learn_entity_types

# 2. 引用 AutoschemaKG 的相關套件
from atlas_rag.llm_generator import LLMGenerator

# 3. 引用 llama_index DynamicLLMPathExtractor
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

# 4. 引用 Schema Cache 管理器
from src.rag.schema.schema_cache import SchemaCacheManager, SchemaCacheKey


def _build_cache_config(method: str, settings=None) -> dict:
    """
    根據 method 組出會影響輸出的 config，避免不同參數共用同一份快取。
    """
    config = {}
    if settings and hasattr(settings, "builder_llm") and getattr(settings, "builder_llm") is not None:
        config["model"] = getattr(settings.builder_llm, "model", "default")
        config["base_url"] = getattr(settings.builder_llm, "base_url", None)

    # OntoLearner（entity-only）目前採用程式預設值；之後若外部化到 config.yml，可在此納入
    if method == "onto_learner_entities":
        config.update(
            {
                "max_entity_types": 30,
                "sample_docs": 5,
                "max_chars_per_doc": 900,
            }
        )
    return config


def _evolution_ratio_tag(evolution_ratio: float) -> str:
    """將 evolution_ratio 轉為整數百分比字串標籤，例如 0.5 → 'r50'。"""
    return f"r{int(round(evolution_ratio * 100))}"


def _resolve_data_context(
    data_type: str = None,
    data_mode: str = None,
    settings=None,
) -> tuple[str, str]:
    """
    統一解析 cache key 所需的 data_type/data_mode。

    優先序：
    1) 函式明確傳入
    2) settings 上的同名屬性
    3) 預設值
    """
    resolved_data_type = (
        data_type
        if data_type is not None
        else (getattr(settings, "data_type", None) if settings else None)
    )
    resolved_data_mode = (
        data_mode
        if data_mode is not None
        else (getattr(settings, "data_mode", None) if settings else None)
    )
    return resolved_data_type or "DI", resolved_data_mode or "natural_text"


def _build_schema_cache_key(
    *,
    cache_manager: SchemaCacheManager,
    method: str,
    text_corpus: list,
    settings=None,
    data_type: str = None,
    data_mode: str = None,
    evolution_ratio: float = 0.5,
    extra_config: dict | None = None,
) -> SchemaCacheKey:
    """統一建立 SchemaCacheKey，避免各 method 分支重複且不一致。"""
    resolved_data_type, resolved_data_mode = _resolve_data_context(
        data_type=data_type,
        data_mode=data_mode,
        settings=settings,
    )
    corpus_hash = cache_manager.compute_corpus_hash(text_corpus) if text_corpus else "default"
    config = _build_cache_config(method=method, settings=settings)
    if method == "iterative_evolution":
        config["evolution_ratio"] = _evolution_ratio_tag(evolution_ratio)
    if extra_config:
        config.update(extra_config)
    config_hash = cache_manager.compute_config_hash(config)
    return SchemaCacheKey(
        method=method,
        data_type=resolved_data_type,
        data_mode=resolved_data_mode,
        text_corpus_hash=corpus_hash,
        config_hash=config_hash,
    )


def _aae_cache_tag(
    evolution_ratio: float,
    min_frequency: int,
    use_cluster: bool,
    use_bootstrap: bool = True,
    use_consolidate: bool = True,
    adaptive_min_frequency: bool = True,
) -> str:
    """AAE 快取 tag，格式：r{ratio}_f{min_freq}[_clus][_boot][_cons][_afq]"""
    tag = f"{_evolution_ratio_tag(evolution_ratio)}_f{min_frequency}"
    if use_cluster:
        tag += "_clus"
    if use_bootstrap:
        tag += "_boot"
    if use_consolidate:
        tag += "_cons"
    if adaptive_min_frequency:
        tag += "_afq"
    return tag


def get_schema_by_method(
    method: str,
    text_corpus: list = None,
    settings=None,
    data_type: str = None,
    data_mode: str = None,
    return_full_schema: bool = True,
    use_cache: bool = True,
    force_rebuild: bool = False,
    evolution_ratio: float = 0.5,
    evolution_min_frequency: int = 2,
    evolution_use_cluster: bool = False,
    evolution_use_bootstrap: bool = True,
    evolution_use_consolidate: bool = True,
    evolution_adaptive_min_frequency: bool = True,
):
    """
    根據指定方法，回傳 schema 資訊
    
    Args:
        method: Schema 生成方法
        text_corpus: 文本語料
        settings: 設定物件
        data_type: 資料集類型（優先於 settings，用於 schema cache key）
        data_mode: 資料格式（優先於 settings，用於 schema cache key）
        return_full_schema: 若為 True，返回完整 schema 物件；若為 False，只返回 entity types 列表
        use_cache: 是否使用快取（預設 True）
        force_rebuild: 是否強制重建，忽略快取（預設 False）
        evolution_ratio: 指定使用語料的最大比例（iterative_evolution / anchored_additive_evolution）
        evolution_min_frequency: AAE 新 type 至少需在幾個批次中被建議（anchored_additive_evolution）
        evolution_use_cluster: AAE 是否使用 embedding 分群取代順序切片（anchored_additive_evolution）
        evolution_use_bootstrap: AAE 是否使用 embedding cluster bootstrap 初始 schema（anchored_additive_evolution）
        evolution_use_consolidate: AAE 是否在迭代後執行 schema consolidation（anchored_additive_evolution）
        evolution_adaptive_min_frequency: AAE 是否依批次數動態調整 min_frequency（anchored_additive_evolution）
    
    Returns:
        若 return_full_schema=True: {"method": str, "entities": list, "relations": list, "validation_schema": dict}
        若 return_full_schema=False: list（entity types）
    """
    
    # 初始化 cache manager
    cache_manager = SchemaCacheManager()
    
    # 若使用快取且非強制重建，嘗試載入
    if use_cache and not force_rebuild and method not in ["lightrag_default"]:
        _extra_cfg: dict | None = None
        if method == "anchored_additive_evolution":
            _extra_cfg = {
                "aae_tag": _aae_cache_tag(
                    evolution_ratio,
                    evolution_min_frequency,
                    evolution_use_cluster,
                    evolution_use_bootstrap,
                    evolution_use_consolidate,
                    evolution_adaptive_min_frequency,
                )
            }
        cache_key = _build_schema_cache_key(
            cache_manager=cache_manager,
            method=method,
            text_corpus=text_corpus,
            settings=settings,
            data_type=data_type,
            data_mode=data_mode,
            evolution_ratio=evolution_ratio,
            extra_config=_extra_cfg,
        )

        cached_schema = cache_manager.load(cache_key)
        if cached_schema:
            schema = cached_schema["schema"]
            if return_full_schema:
                return schema
            return schema.get("entities", [])
    
    # 1. 直接調用 extract_schema_only.py 的動態演化邏輯，不再只是讀取 JSON 檔案
    if method == "lightrag_default":
        # LightRAG 預設 Baseline
        entities = [
            "Person", "Creature", "Organization", "Location", "Event",
            "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject",
        ]
        if return_full_schema:
            return {
                "method": method,
                "entities": entities,
                "relations": [],
                "validation_schema": {}
            }
        return entities

    elif method == "adaptive_consolidation":
        if not text_corpus or not settings.builder_llm:
            raise ValueError("adaptive_consolidation 需要 text_corpus 與 llm 實例")

        # 以 LightRAG baseline 為起點，逐步收斂成較貼近語料的 schema。
        base_schema = {
            "entities": [
                "Person", "Creature", "Organization", "Location", "Event",
                "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject",
            ],
            "relations": [],
            "validation_schema": {},
        }

        learned_schema = learn_schema_adaptively(
            documents=text_corpus,
            base_schema=base_schema,
            llm=settings.builder_llm,
            batch_size=5,
            patience=3,
        )

        result_schema = {
            "method": method,
            "entities": learned_schema.get("entities", []),
            "relations": learned_schema.get("relations", []),
            "validation_schema": learned_schema.get("validation_schema", {}),
        }

        # 儲存到快取
        if use_cache and not force_rebuild:
            cache_key = _build_schema_cache_key(
                cache_manager=cache_manager,
                method=method,
                text_corpus=text_corpus,
                settings=settings,
                data_type=data_type,
                data_mode=data_mode,
            )
            cache_manager.save(cache_key, result_schema)

        if return_full_schema:
            return result_schema
        return result_schema.get("entities", [])
        
    elif method == "iterative_evolution":
        if not text_corpus or not settings.builder_llm:
            raise ValueError("動態演化萃取需要提供 text_corpus 與 llm 實例")

        print(f">> iterative_evolution: max_ratio={evolution_ratio} ({_evolution_ratio_tag(evolution_ratio)})")
        evolved_schema = run_iterative_evolution(
            documents=text_corpus,
            llm=settings.builder_llm,
            max_ratio=evolution_ratio,
            batch_ratio=0.1,
        )

        result_schema = {
            "method": method,
            "entities": evolved_schema.get("entities", []),
            "relations": evolved_schema.get("relations", []),
            "validation_schema": evolved_schema.get("validation_schema", {}),
        }

        # 儲存到快取
        if use_cache and not force_rebuild:
            cache_key = _build_schema_cache_key(
                cache_manager=cache_manager,
                method=method,
                text_corpus=text_corpus,
                settings=settings,
                data_type=data_type,
                data_mode=data_mode,
                evolution_ratio=evolution_ratio,
            )
            cache_manager.save(cache_key, result_schema)

        if return_full_schema:
            return result_schema
        return result_schema.get("entities", [])
        
    elif method == "anchored_additive_evolution":
        if not text_corpus or not settings.builder_llm:
            raise ValueError("anchored_additive_evolution 需要提供 text_corpus 與 llm 實例")

        aae_tag = _aae_cache_tag(
            evolution_ratio,
            evolution_min_frequency,
            evolution_use_cluster,
            evolution_use_bootstrap,
            evolution_use_consolidate,
            evolution_adaptive_min_frequency,
        )
        print(f">> anchored_additive_evolution: {aae_tag}")

        # embed_model 在 use_cluster 或 use_bootstrap 時都需要
        _need_embed = evolution_use_cluster or evolution_use_bootstrap
        embed_model = getattr(settings, "embed_model", None) if _need_embed else None
        if _need_embed and embed_model is None:
            print(">> [AAE] 警告：embed_model 不存在，cluster/bootstrap 功能將 fallback 至順序切片")

        evolved_schema = run_anchored_additive_evolution(
            documents=text_corpus,
            llm=settings.builder_llm,
            max_ratio=evolution_ratio,
            batch_ratio=0.1,
            min_frequency=evolution_min_frequency,
            use_cluster=evolution_use_cluster,
            embed_model=embed_model,
            use_bootstrap=evolution_use_bootstrap,
            use_consolidate=evolution_use_consolidate,
            adaptive_min_frequency=evolution_adaptive_min_frequency,
        )

        result_schema = {
            "method": method,
            "entities": evolved_schema.get("entities", []),
            "relations": evolved_schema.get("relations", []),
            "validation_schema": evolved_schema.get("validation_schema", {}),
        }

        # 儲存到快取
        if use_cache and not force_rebuild:
            cache_key = _build_schema_cache_key(
                cache_manager=cache_manager,
                method=method,
                text_corpus=text_corpus,
                settings=settings,
                data_type=data_type,
                data_mode=data_mode,
                evolution_ratio=evolution_ratio,
                extra_config={"aae_tag": aae_tag},
            )
            cache_manager.save(cache_key, result_schema)

        if return_full_schema:
            return result_schema
        return result_schema.get("entities", [])

    # 2. 參考 AutoschemaKG.py 與 autoschema_lightrag_package.py 實作
    elif method == "llm_dynamic":
        if not text_corpus:
            raise ValueError("需要提供 text_corpus")
        
        # 從 builder_llm 物件提取配置
        base_url = getattr(settings.builder_llm, 'base_url', 'http://192.168.63.174:11434')
        model_name = getattr(settings.builder_llm, 'model', 'qwen2.5:14b')
        
        # 參考 AutoschemaKG 連接本機 Ollama 服務 API
        client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key='ollama' 
        )
        
        sample_text = "\n".join([doc.text[:] if hasattr(doc, 'text') else str(doc)[:] for doc in text_corpus[:]])
        
        # 參考 autoschema_lightrag_package.py 的 Prompt 精神做約束
        custom_prompt = f"""
        You are a specialized enterprise customer service knowledge extractor.
        Please extract up to 10 most important entity types (e.g., Server_Version, Person, Organization) from the following text.
        Do not invent new entity types outside of the context.
        Output ONLY a comma-separated list of entity types.
        
        Text:
        {sample_text}
        """
        
        # 透過 OpenAI client 直接生成結果
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": custom_prompt}],
            max_tokens=2048
        )
        
        result_text = response.choices[0].message.content
        entities = [e.strip() for e in result_text.split(",") if e.strip()]
        
        result_schema = {
            "method": method,
            "entities": entities,
            "relations": [],
            "validation_schema": {}
        }
        
        # 儲存到快取
        if use_cache and not force_rebuild:
            cache_key = _build_schema_cache_key(
                cache_manager=cache_manager,
                method=method,
                text_corpus=text_corpus,
                settings=settings,
                data_type=data_type,
                data_mode=data_mode,
                extra_config={"model": model_name, "base_url": base_url},
            )
            cache_manager.save(cache_key, result_schema)
        
        if return_full_schema:
            return result_schema
        return entities

    elif method == "onto_learner_entities":
        if not text_corpus or not settings.builder_llm:
            raise ValueError("onto_learner_entities 需要 text_corpus 與 llm 實例")

        entities = onto_learn_entity_types(text_corpus, settings.builder_llm)

        result_schema = {
            "method": method,
            "entities": entities,
            "relations": [],
            "validation_schema": {},
        }

        # 儲存到快取
        if use_cache and not force_rebuild:
            cache_key = _build_schema_cache_key(
                cache_manager=cache_manager,
                method=method,
                text_corpus=text_corpus,
                settings=settings,
                data_type=data_type,
                data_mode=data_mode,
            )
            cache_manager.save(cache_key, result_schema)

        if return_full_schema:
            return result_schema
        return entities
        
    # 3. 使用 llamaindex 的 DynamicLLMPathExtractor 參考 graph_dynamic_schema_package.py
    elif method == "llamaindex_dynamic":
        if not text_corpus or not settings.builder_llm:
            raise ValueError("llamaindex_dynamic 需要 text_corpus 與 llm 實例")
            
        print(">> 初始化 DynamicLLMPathExtractor...")
        dynamic_extractor = DynamicLLMPathExtractor(
            llm=settings.builder_llm,
            max_triplets_per_chunk=20,
            num_workers=4,
        )
        
        # 將輸入文本轉換為 LlamaIndex 可處理的 TextNode 格式
        nodes = []
        for doc in text_corpus[:3]:  # 取少部分文本測試抽取以節省時間
            text = doc.text if hasattr(doc, 'text') else str(doc)
            nodes.append(TextNode(text=text))
            
        print(">> 開始動態抽取 Graph 節點與實體關聯...")
        extracted_nodes = dynamic_extractor(nodes)
        
        # 收集從動態生成的圖譜中取出的 Entity Types (Label)
        entity_types = set()
        for node in extracted_nodes:
            if hasattr(node, 'metadata') and 'kg_nodes' in node.metadata:
                for kg_node in node.metadata['kg_nodes']:
                    if hasattr(kg_node, 'label'):
                        entity_types.add(kg_node.label)
                        
        # 回傳集合；若未抽取成功則提供備用實體
        entities = list(entity_types) if entity_types else ["Person", "Organization", "Concept"]
        
        if return_full_schema:
            return {
                "method": method,
                "entities": entities,
                "relations": [],
                "validation_schema": {}
            }
        return entities
        
    else:
        raise ValueError(f"未知的 Schema 方法: {method}")