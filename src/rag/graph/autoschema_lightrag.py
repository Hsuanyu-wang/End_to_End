import os
from lightrag import LightRAG
from lightrag.llm import openai_complete_if_cache

class AutoSchemaLightRAGPackage:
    def __init__(self, working_dir, schema_path, llm_model="gpt-4o-mini"):
        """
        [引用: AutoSchemaKG 概念]
        強制 LightRAG 在抽取實體時遵循指定的 JSON Schema。
        """
        self.working_dir = working_dir
        self.schema_path = schema_path
        
        # 讀取預先透過抽樣學習到的企業客服 Schema
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.domain_schema = f.read()

        # 客製化 LightRAG 的 PROMPT，注入 Schema 約束
        custom_entity_extract_prompt = f"""
        You are a specialized enterprise customer service knowledge extractor.
        You MUST extract entities and relationships strictly following this schema:
        {self.domain_schema}
        
        Do not invent new entity types outside of this schema.
        """
        
        # 初始化 LightRAG 並覆寫預設 Prompt
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=openai_complete_if_cache,
            llm_model_name=llm_model,
            entity_extract_prompt=custom_entity_extract_prompt # 覆寫 Prompt
        )

    def insert(self, texts: list[str]):
        self.rag.insert(texts)

    def query(self, query_text: str, mode="hybrid"):
        # mode 可以是 'local', 'global', 或 'hybrid'
        return self.rag.query(query_text, param={"mode": mode})