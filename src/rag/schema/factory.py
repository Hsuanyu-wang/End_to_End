import json
from openai import OpenAI
from llama_index.core.schema import TextNode

# 1. 引用 extract_schema_only 內容
from src.rag.schema.evolution import evolve_schema_with_pydantic

# 2. 引用 AutoschemaKG 的相關套件
from atlas_rag.llm_generator import LLMGenerator

# 3. 引用 llama_index DynamicLLMPathExtractor
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

# 4. 引用 Schema Cache 管理器
from src.rag.schema.schema_cache import SchemaCacheManager, SchemaCacheKey

def get_schema_by_method(
    method: str, 
    text_corpus: list = None, 
    settings=None, 
    return_full_schema: bool = True,
    use_cache: bool = True,
    force_rebuild: bool = False
):
    """
    根據指定方法，回傳 schema 資訊
    
    Args:
        method: Schema 生成方法
        text_corpus: 文本語料
        settings: 設定物件
        return_full_schema: 若為 True，返回完整 schema 物件；若為 False，只返回 entity types 列表
        use_cache: 是否使用快取（預設 True）
        force_rebuild: 是否強制重建，忽略快取（預設 False）
    
    Returns:
        若 return_full_schema=True: {"method": str, "entities": list, "relations": list, "validation_schema": dict}
        若 return_full_schema=False: list（entity types）
    """
    
    # 初始化 cache manager
    cache_manager = SchemaCacheManager()
    
    # 若使用快取且非強制重建，嘗試載入
    if use_cache and not force_rebuild and method not in ["lightrag_default"]:
        # 構建 cache key
        data_type = getattr(settings, 'data_type', 'DI') if settings else "DI"
        data_mode = getattr(settings, 'data_mode', 'natural_text') if settings else "natural_text"
        
        # 計算 corpus hash
        corpus_hash = cache_manager.compute_corpus_hash(text_corpus) if text_corpus else "default"
        
        # 計算 config hash
        config = {}
        if settings and hasattr(settings, 'builder_llm'):
            config['model'] = getattr(settings.builder_llm, 'model', 'default')
        config_hash = cache_manager.compute_config_hash(config)
        
        cache_key = SchemaCacheKey(
            method=method,
            data_type=data_type,
            data_mode=data_mode,
            text_corpus_hash=corpus_hash,
            config_hash=config_hash
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
        
    elif method == "iterative_evolution":
        if not text_corpus or not settings.builder_llm:
            raise ValueError("動態演化萃取需要提供 text_corpus 與 llm 實例")
            
        base_schema = {
            "entities": [],
            "relations": [],
            "validation_schema": {},
        }
        
        # 抽取部分文本批次避免超出 Token 限制
        batch_docs = text_corpus[:5] 
        evolved_schema = evolve_schema_with_pydantic(base_schema, batch_docs, settings.builder_llm)
        
        result_schema = {
            "method": method,
            "entities": evolved_schema.get("entities", []),
            "relations": evolved_schema.get("relations", []),
            "validation_schema": evolved_schema.get("validation_schema", {})
        }
        
        # 儲存到快取
        if use_cache and not force_rebuild:
            data_type = getattr(settings, 'data_type', 'DI') if settings else "DI"
            data_mode = getattr(settings, 'data_mode', 'natural_text') if settings else "natural_text"
            corpus_hash = cache_manager.compute_corpus_hash(text_corpus) if text_corpus else "default"
            config = {'model': getattr(settings.builder_llm, 'model', 'default')}
            config_hash = cache_manager.compute_config_hash(config)
            
            cache_key = SchemaCacheKey(
                method=method,
                data_type=data_type,
                data_mode=data_mode,
                text_corpus_hash=corpus_hash,
                config_hash=config_hash
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
            data_type = getattr(settings, 'data_type', 'DI') if settings else "DI"
            data_mode = getattr(settings, 'data_mode', 'natural_text') if settings else "natural_text"
            corpus_hash = cache_manager.compute_corpus_hash(text_corpus) if text_corpus else "default"
            config = {'model': model_name, 'base_url': base_url}
            config_hash = cache_manager.compute_config_hash(config)
            
            cache_key = SchemaCacheKey(
                method=method,
                data_type=data_type,
                data_mode=data_mode,
                text_corpus_hash=corpus_hash,
                config_hash=config_hash
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