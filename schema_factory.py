import json

def get_schema_by_method(method: str, text_corpus: list = None, llm=None) -> list:
    """根據指定方法，回傳 LightRAG 可用的 entity_types 列表"""
    
    if method == "lightrag_default":
        # LightRAG 預設 Baseline
        return [
            "Person", "Creature", "Organization", "Location", "Event",
            "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject",
        ]
        
    elif method == "iterative_evolution":
        # 讀取由 extract_schema_only.py 產生並收斂的 JSON 檔案
        #
        try:
            with open("/home/End_to_End_RAG/custom_domain_schema.json", "r") as f:
                schema_data = json.load(f)
            return schema_data.get("entities", [])
        except FileNotFoundError:
            print("找不到預先抽取的 Schema，請先執行 extract_schema_only.py")
            return ["Record", "Engineer", "Customer"]
            
    elif method == "llm_dynamic":
        # 概念實作：即時呼叫 LLM 進行一次性摘要萃取
        if not text_corpus or not llm:
            raise ValueError("LLM 動態萃取需要文本與 LLM 實例")
        sample_text = "\n".join([doc.text[:500] for doc in text_corpus[:5]])
        prompt = f"請從以下文本中歸納出 10 個最重要的實體類別(如: Server_Version)，僅以逗號分隔輸出：\n{sample_text}"
        response = llm.complete(prompt).text
        return [e.strip() for e in response.split(",")]
        
    else:
        raise ValueError(f"未知的 Schema 方法: {method}")