import json
from openai import OpenAI
from llama_index.core.schema import TextNode

# 1. 引用 extract_schema_only 內容
from extract_schema_only import evolve_schema_with_pydantic

# 2. 引用 AutoschemaKG 的相關套件
from atlas_rag.llm_generator import LLMGenerator

# 3. 引用 llama_index DynamicLLMPathExtractor
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor

def get_schema_by_method(method: str, text_corpus: list = None, settings=None) -> list:
    """根據指定方法，回傳可用的 entity_types 列表"""
    
    # 1. 直接調用 extract_schema_only.py 的動態演化邏輯，不再只是讀取 JSON 檔案
    if method == "lightrag_default":
        # LightRAG 預設 Baseline
        return [
            "Person", "Creature", "Organization", "Location", "Event",
            "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject",
        ]
        
    elif method == "iterative_evolution":
        if not text_corpus or not settings.builder_llm:
            raise ValueError("動態演化萃取需要提供 text_corpus 與 llm 實例")
            
        # 提供給 LLM 演化的初始 base schema
        # base_schema = {
        #     "entities": ["Record", "Engineer", "Customer", "Action", "Issue"],
        #     "relations": ["BELONGS_TO", "HANDLED", "TAKEN_ACTION", "HAS_ISSUE"],
        #     "validation_schema": {
        #         "Record": ["BELONGS_TO", "TAKEN_ACTION", "HAS_ISSUE"],
        #         "Engineer": ["HANDLED"],
        #         "Customer": ["BELONGS_TO"]
        #     }
        # }
        base_schema = {
            "entities": [],
            "relations": [],
            "validation_schema": {},
        }
        
        # 抽取部分文本批次避免超出 Token 限制
        batch_docs = text_corpus[:5] 
        evolved_schema = evolve_schema_with_pydantic(base_schema, batch_docs, settings.builder_llm)
        
        return evolved_schema.get("entities", [])
        
    # 2. 參考 AutoschemaKG.py 與 autoschema_lightrag_package.py 實作
    elif method == "llm_dynamic":
        if not text_corpus:
            raise ValueError("需要提供 text_corpus")
        
        # 參考 AutoschemaKG 連接本機 Ollama 服務 API
        client = OpenAI(
            base_url='http://192.168.63.174:11434/v1',
            api_key='ollama' 
        )
        model_name = settings.builder_llm
        
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
        return [e.strip() for e in result_text.split(",") if e.strip()]
        
    # 3. 使用 llamaindex 的 DynamicLLMPathExtractor 參考 graph_dynamic_schema_package.py
    elif method == "llamaindex_dynamic":
        if not text_corpus or not llm:
            raise ValueError("llamaindex_dynamic 需要 text_corpus 與 llm 實例")
            
        print(">> 初始化 DynamicLLMPathExtractor...")
        dynamic_extractor = DynamicLLMPathExtractor(
            llm=llm,
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
        return list(entity_types) if entity_types else ["Person", "Organization", "Concept"]
        
    else:
        raise ValueError(f"未知的 Schema 方法: {method}")