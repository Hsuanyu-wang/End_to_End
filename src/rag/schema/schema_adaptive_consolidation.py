import json
from copy import deepcopy
from pydantic import BaseModel, Field
from typing import List, Dict
from llama_index.core import PromptTemplate

# 1. 定義 Pydantic 輸出格式
class DynamicKGSchema(BaseModel):
    entities: List[str] = Field(description="實體類型列表")
    relations: List[str] = Field(description="關係類型列表")
    validation_schema: Dict[str, List[str]] = Field(description="實體與關係的合法連接規則")

class SchemaMergeResult(BaseModel):
    merged_entities: List[str] = Field(description="合併同義詞後的最終實體列表")
    merged_relations: List[str] = Field(description="合併同義詞後的最終關係列表")
    updated_validation: Dict[str, List[str]] = Field(description="修正後的驗證規則")

# 2. 階段一：提取初步提議 (維持您原有的功能，但改為提議性質)
def propose_schema_additions(current_schema: dict, sample_text: str, llm) -> dict:
    prompt_str = f"""
    你是一個知識圖譜分析師。
    現有的 Schema 為: {json.dumps(current_schema, ensure_ascii=False)}
    請閱讀以下文本，判斷是否有「絕對必要」的新實體類型或關係類型需要加入。
    若無，請直接輸出原有 Schema。
    文本: \"\"\"{sample_text}\"\"\"
    """
    prompt_template = PromptTemplate(template=prompt_str)
    response = llm.structured_predict(DynamicKGSchema, prompt=prompt_template)
    return response.model_dump()

# 3. 階段二：新增的「合併與對齊」代理 (解決圖譜漂移的核心)
def refine_and_merge_schema(proposed_schema: dict, llm) -> dict:
    prompt_str = f"""
    你是一個知識圖譜架構總監。
    下屬提交了一份新提案 Schema: {json.dumps(proposed_schema, ensure_ascii=False)}
    
    你的任務是進行「本體對齊與修剪」：
    1. 合併同義實體 (例如: Client, Customer, Buyer 應該統一為 Customer)
    2. 合併同義關係 (例如: HAS_ISSUE, FACES_PROBLEM 應該統一為 HAS_ISSUE)
    3. 刪除過於空泛或具體的實體 (例如: "Thing", "System_Error_Code_404")
    4. 確保 validation_schema 邏輯連貫。
    
    請輸出最終精煉後的 Schema。
    """
    prompt_template = PromptTemplate(template=prompt_str)
    response = llm.structured_predict(SchemaMergeResult, prompt=prompt_template)
    
    # 轉回標準格式
    result = response.model_dump()
    return {
        "entities": result["merged_entities"],
        "relations": result["merged_relations"],
        "validation_schema": result["updated_validation"]
    }

# 4. 主流程：自適應收斂迴圈 (取代原來的 for loop)
def learn_schema_adaptively(documents, base_schema, llm, batch_size=5, patience=3):
    current_schema = deepcopy(base_schema)
    stable_rounds = 0
    
    # 建議：這裡可替換為 KMeans 多樣性抽樣，而非依序
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        # 限制字數以防破表
        sample_text = "\n".join([doc.text[:800] for doc in batch_docs]) 
        
        print(f"\n--- [Round {i//batch_size + 1}] 分析新文本 ---")
        
        # 步驟 A: 提議新 Schema
        proposed_schema = propose_schema_additions(current_schema, sample_text, llm)
        
        # 步驟 B: 合併與精煉 Schema (防禦發散)
        refined_schema = refine_and_merge_schema(proposed_schema, llm)
        
        # 步驟 C: 檢查是否收斂 (Early Stopping)
        old_entities = set(current_schema["entities"])
        new_entities = set(refined_schema["entities"])
        
        if old_entities == new_entities:
            stable_rounds += 1
            print(f">> Schema 無變動，收斂計數: {stable_rounds}/{patience}")
        else:
            stable_rounds = 0
            print(f">> Schema 發生更新！新增: {new_entities - old_entities}")
            print(f">> 被整併/移除: {old_entities - new_entities}")
            
        current_schema = refined_schema
        
        # 若連續 N 輪都沒有新實體，宣告收斂，提早結束學習
        if stable_rounds >= patience:
            print(f"\n✅ Schema 已連續 {patience} 輪未變更，達到全局收斂！提早終止本體學習。")
            break

    return current_schema

# --- 執行階段 ---
# 1. 執行 ASC 學習 Schema
# final_schema = learn_schema_adaptively(documents, initial_schema, Settings.llm, batch_size=10, patience=3)

# 2. 將收斂後的 Schema 傳給 SchemaLLMPathExtractor 進行全局抽取
# llm_extractor = SchemaLLMPathExtractor(
#     llm=Settings.llm,
#     possible_entities=final_schema["entities"],
#     ...
# )