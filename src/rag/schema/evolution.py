import json
import math
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Dict

# 載入自定義設定與資料 (沿用您原本的架構)
from llama_index.core import PromptTemplate
from src.config.settings import get_settings
from src.data.processors import data_processing

Settings = get_settings()

# 1. 定義期望的 Schema 資料結構 (保留您原本的設計)
class DynamicKGSchema(BaseModel):
    entities: List[str] = Field(description="領域專屬的實體類型列表 (Entity Types)")
    relations: List[str] = Field(description="實體之間的關係類型列表 (Relation Types)")
    validation_schema: Dict[str, List[str]] = Field(description="實體與關係的合法連接規則，格式如 {'實體A': ['關係1', '關係2']}")

# 2. 核心：結合 LLM 進行 Schema 的動態演化
def evolve_schema_with_pydantic(current_schema: dict, batch_docs: list, llm) -> dict:
    # 取適當長度的文本，避免超出 Token 限制 (可依據您的 LLM Context Window 調整)
    sample_text = "\n\n".join([doc.text[:] if hasattr(doc, 'text') else str(doc)[:] for doc in batch_docs[:]]) 
    
    # 強化 Prompt：明確告知 LLM 需要「融合」新舊 Schema，而不是每次都重新發明
    prompt_str = """
    你是一個專業的領域本體論（Ontology）與知識圖譜架構師。
    任務：從給定的領域文本中，萃取出專屬的「實體類型」、「關係類型」與「連接規則」。

    【現有 Schema 狀態】
    {current_schema}

    【任務要求】
    1. 閱讀下方文本，找出重要且具代表性的業務實體與關聯。
    2. 若文本中出現現有 Schema 未涵蓋的新概念，請將其「擴充」至 Schema 中。
    3. 若文本內容未提供新資訊，請「保留」現有 Schema，切勿隨意刪除舊有結構。
    4. 提取的類別必須具備「抽象性」（例如提取 "Server" 而不是特定的 "Server_001"）。

    【文本內容】
    {text}
    """
    
    prompt_template = PromptTemplate(template=prompt_str)
    
    try:
        # 呼叫 LlamaIndex 的 structured_predict，強迫 LLM 吐出符合 Pydantic 格式的 JSON
        response_obj = llm.structured_predict(
            DynamicKGSchema,
            prompt=prompt_template,
            current_schema=json.dumps(current_schema, ensure_ascii=False),
            text=sample_text
        )
        return response_obj.model_dump()
    except Exception as e:
        print(f">> Schema 更新失敗，沿用上一輪狀態: {e}")
        return current_schema

# 3. 主流程：文件批次處理與 Schema 收斂
def main():
    print(">> 正在載入文件...")
    documents = data_processing(mode="natural_text", data_type="DI")
    total_docs = len(documents)
    print(f">> 總文件數: {total_docs}")

    # 給定一個初始的 Base Schema (幫助 LLM 抓準您想要的顆粒度與方向)
    dynamic_schema = {
        "entities": ["Record", "Engineer", "Customer", "Action", "Issue"],
        "relations": ["BELONGS_TO", "HANDLED", "TAKEN_ACTION", "HAS_ISSUE"],
        "validation_schema": {
            "Record": ["BELONGS_TO", "TAKEN_ACTION", "HAS_ISSUE"],
            "Engineer": ["HANDLED"],
            "Customer": ["BELONGS_TO"]
        }
    }

    # 設定批次大小，將文件切塊餵給 LLM
    BATCH_SIZE = max(1, int(total_docs * 0.1))
    
    print(f">> 開始進行 Schema 動態演化 (每批次 {BATCH_SIZE} 篇)...")
    pbar = tqdm(range(0, total_docs, BATCH_SIZE), desc="Extracting Schema")

    for i in pbar:
        batch = documents[i : i + BATCH_SIZE]
        round_num = (i // BATCH_SIZE) + 1
        
        # 記錄更新前的實體，方便觀察變化
        old_entities = set(dynamic_schema.get("entities", []))
        
        # 進行演化
        dynamic_schema = evolve_schema_with_pydantic(dynamic_schema, batch, Settings.llm)
        
        new_entities = set(dynamic_schema.get("entities", []))
        added_entities = new_entities - old_entities
        
        pbar.set_postfix({"Round": round_num, "Entity_Count": len(new_entities)})
        
        if added_entities:
            print(f"\n[Round {round_num}] 發現新實體: {added_entities}")

    print("\n>> 領域本體論 (Schema) 萃取完成！")
    
    # 4. 將收斂完成的 Schema 存出
    output_path = "/home/End_to_End_RAG/custom_domain_schema.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dynamic_schema, f, indent=4, ensure_ascii=False)
        
    print(f">> 最終 Schema 已儲存至: {output_path}")
    print("\n最終 Schema 預覽:")
    print(json.dumps(dynamic_schema, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()