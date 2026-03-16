import nest_asyncio
nest_asyncio.apply()

import os
import yaml
import json
import argparse
import math
import random
from tqdm import tqdm  # 導入 tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from llama_index.core import Document, PropertyGraphIndex, PromptTemplate
from llama_index.core.schema import TransformComponent
from llama_index.core.graph_stores import EntityNode, Relation
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.llms import ChatMessage

# 載入自定義設定與資料
from model_settings import get_settings
from data_processing import data_processing

load_dotenv()
Settings = get_settings()
config = yaml.safe_load(open("/home/End_to_End_RAG/config.yml"))

# --- 參數解析 (整合至一處) ---
parser = argparse.ArgumentParser(description="Knowledge Graph Construction Script")
parser.add_argument("--init_neo4j", action="store_true", help="清空 Neo4j 資料庫中的舊資料")
args = parser.parse_args()

class DynamicKGSchema(BaseModel):
    entities: List[str] = Field(description="實體類型（大寫駝峰，如：MaintenanceRecord, Hardware）")
    relations: List[str] = Field(description="關係類型（全大寫底線，如：HAS_ISSUE, PERFORMED_BY）")
    validation_schema: Dict[str, List[str]] = Field(description="定義各實體類型可發出的關係。Key 為來源實體類型，Value 為可使用的關係列表")

class SafeNeo4jStore(Neo4jPropertyGraphStore):
    def upsert_nodes(self, nodes):
        for node in nodes:
            if hasattr(node, "metadata") and isinstance(node.metadata, dict):
                node.metadata.pop("kg_nodes", None)
                node.metadata.pop("kg_relations", None)
            if hasattr(node, "properties") and isinstance(node.properties, dict):
                node.properties.pop("kg_nodes", None)
                node.properties.pop("kg_relations", None)
        super().upsert_nodes(nodes)

    def upsert_relations(self, relations):
        for rel in relations:
            if hasattr(rel, "properties") and isinstance(rel.properties, dict):
                rel.properties.pop("kg_nodes", None)
                rel.properties.pop("kg_relations", None)
            if hasattr(rel, "metadata") and isinstance(rel.metadata, dict):
                rel.metadata.pop("kg_nodes", None)
                rel.metadata.pop("kg_relations", None)
        super().upsert_relations(relations)

    def delete_all(self):
        """清空 Neo4j 資料庫中的所有節點與關係"""
        print(">> 正在清空 Neo4j 舊資料...")
        cypher_query = "MATCH (n) DETACH DELETE n"
        # 方案 A: 使用內部的 client 執行 (最直接)
        # Neo4jPropertyGraphStore 實例化時會建立 self.client
        with self.client.session(database=self._database) as session:
            session.run(cypher_query)
        print(">> 清理完成。")

# --- Metadata 抽取器 (保留) ---
class CustomMetadataExtractor(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            if "kg_nodes" not in node.metadata:
                node.metadata["kg_nodes"] = []
            if "kg_relations" not in node.metadata:
                node.metadata["kg_relations"] = []
            
            record_id = node.metadata.get("NO", "Unknown")
            customer = node.metadata.get("Customer")
            engineer = node.metadata.get("Engineers")
            m_category = node.metadata.get("維護類別")
            
            record_node = EntityNode(
                name=f"Record_{record_id}", 
                label="Record", 
                properties={
                    "id": record_id,
                    "service_start": node.metadata.get("Service Start"),
                    "service_end": node.metadata.get("Service End")
                }
            )
            node.metadata["kg_nodes"].append(record_node)

            if customer:
                customer_node = EntityNode(name=customer, label="Customer")
                rel = Relation(source_id=record_node.id, target_id=customer_node.id, label="BELONGS_TO")
                node.metadata["kg_nodes"].append(customer_node)
                node.metadata["kg_relations"].append(rel)
            
            if engineer:
                engineer_node = EntityNode(name=engineer, label="Engineer")
                rel = Relation(source_id=engineer_node.id, target_id=record_node.id, label="HANDLED")
                node.metadata["kg_nodes"].append(engineer_node)
                node.metadata["kg_relations"].append(rel)

        return nodes

def evolve_schema_with_pydantic(current_schema: dict, batch_docs: list, llm) -> dict:
    sample_docs = random.sample(batch_docs, min(len(batch_docs), 5))
    sample_text = "\n---\n".join([doc.text[:1000] for doc in sample_docs]) # 限制長度防止 Token 爆炸
    prompt_str = """
    你是一位資深的知識圖譜架構師。
    任務：從提供的維護紀錄文本中，提取抽象的『實體類型(Entity Types)』與『關係類型(Relation Types)』。
    
    注意：
    1. 不要提取具體的實體（例如：不要提取 '伺服器A'，而要提取 'Server'）。
    2. 關係應該是抽象的連接（例如：'RESOLVES', 'LOCATED_IN'）。
    3. validation_schema 必須定義哪些實體類型可以透過哪種關係連接。
       格式範例：{"Engineer": ["HANDLED"], "Record": ["BELONGS_TO"]}
    
    現有 Schema: {current_schema}
    文本內容: 
    \"\"\"
    {sample_text}
    \"\"\"
    """
    prompt_template = PromptTemplate(template=prompt_str)
    full_prompt = prompt_template.format(
        current_schema=json.dumps(current_schema, ensure_ascii=False),
        sample_text=sample_text
    )
    try:
        response_obj = llm.structured_predict(
            DynamicKGSchema,
            prompt=PromptTemplate(template=full_prompt)
        )
        return response_obj.model_dump()
    except Exception as e:
        print(f">> Schema 更新失敗: {e}")
        return current_schema

def save_schema(schema, filepath="final_schema.json"):
    """將最終 Schema 儲存為 JSON 檔案"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=4, ensure_ascii=False)
        print(f">> 最終 Schema 已成功儲存至: {filepath}")
    except Exception as e:
        print(f">> 儲存 Schema 失敗: {e}")
        
graph_store = SafeNeo4jStore(
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    url=os.getenv("NEO4J_URI"),
    database=config["graph"]["neo4j_database"]
)

# 初始 Schema
current_schema = {
    "entities": ["MaintenanceCategory", "Customer", "Record", "Engineer", "Action", "Issue", "ActionCategory", "ActionSubcategory", "System", "Vulnerability", "Software", "Server"],
    "relations": ["TAKEN_ACTION", "HAS_ISSUE", "HAS_ACTION_CATEGORY", "RESOLVES", "TRIGGERED_BY", "CAUSED_BY", "INVOLVES_SERVER",
                    "INVOLVES_SOFTWARE", "INVOLVES_SYSTEM", "INVOLVES_VULNERABILITY", "RELATED_TO"],
    "validation_schema": {
        "Engineer": ["HANDLED"],
        "Customer": ["BELONGS_TO"],
        "Record": ["BELONGS_TO", "HAS_MAINTENANCE_CATEGORY", "TAKEN_ACTION", "HAS_ISSUE"],
        "MaintenanceCategory": ["HAS_MAINTENANCE_CATEGORY"],
        "Action": ["TAKEN_ACTION", "HAS_ACTION_CATEGORY", "RESOLVES", "TRIGGERED_BY", "INVOLVES_SERVER", "INVOLVES_SOFTWARE", "INVOLVES_VULNERABILITY"],
        "Issue": ["HAS_ISSUE", "CAUSED_BY", "INVOLVES_SERVER", "INVOLVES_SOFTWARE", "INVOLVES_SYSTEM", "INVOLVES_VULNERABILITY"],
        "ActionCategory": ["HAS_ACTION_CATEGORY"],
        "ActionSubcategory": ["HAS_ACTION_SUBCATEGORY"],
        "System": ["INVOLVES_SYSTEM"],
        "Vulnerability": ["INVOLVES_VULNERABILITY"],
        "Software": ["INVOLVES_SOFTWARE"],
        "Server": ["INVOLVES_SERVER"],
    }
}

# --- 第一階段：Schema 定義 (Schema Sampling Phase) ---
def schema_definition_phase(documents, iterations=5, sample_ratio=0.1):
    print(f">> 階段 1: 開始 Schema 定義階段 (抽樣比例: {sample_ratio*100}%)")
    
    consecutive_stable_rounds = 0
    
    # 抽樣部分資料來演化 Schema
    sample_size = max(1, int(len(documents) * sample_ratio))
    random.shuffle(documents, random.seed(42))
    # print (f"documents[:{sample_size}]: ", documents[:sample_size])
    sample_docs = documents[:sample_size]
    
    # 這裡可以跑 2-3 輪演化來收斂
    for i in range(iterations):
        print(f"   [演化輪次 {i+1}/{iterations}] 正在分析文本特徵...")
        
        # 紀錄舊的以便比較
        old_entities = set(current_schema["entities"])
        old_relations = set(current_schema["relations"])
        
        # 更新 Schema
        current_schema = evolve_schema_with_pydantic(current_schema, sample_docs, Settings.llm)
        
        # 計算差異
        new_entities = set(current_schema["entities"])
        new_relations = set(current_schema["relations"])
        
        # 顯示變動
        added_e = new_entities - old_entities
        removed_e = old_entities - new_entities
        added_r = new_relations - old_relations
        removed_r = old_relations - new_relations
        
        if not added_e and not removed_e and not added_r and not removed_r:
            consecutive_stable_rounds += 1
            if consecutive_stable_rounds >= 2:
                print("   (Schema 已收斂，提前結束)")
                break
        else:
            consecutive_stable_rounds = 0
        
        if added_e or removed_e or added_r or removed_r:
            print(f"   + 新增實體: {added_e if added_e else '無'}")
            print(f"   - 刪減實體: {removed_e if removed_e else '無'}")
            print(f"   + 新增關係: {added_r if added_r else '無'}")
            print(f"   - 刪減關係: {removed_r if removed_r else '無'}")
        else:
            print("   (Schema 無變動，已趨於穩定)")
            print ("final_schema: ", current_schema)
            
        if i == iterations - 1:
            print ("final_schema: ", current_schema)
    print(">> 階段 1 完成，最終 Schema 已鎖定。\n")
    return current_schema

# --- 第二階段：全量抽取 (Bulk Extraction Phase) ---
def bulk_extraction_phase(documents, final_schema, graph_store):
    print(f">> 階段 2: 開始全量資料抽取 (總文件數: {len(documents)})")
    
    # 統一使用階段 1 產出的 Extractor
    llm_extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=final_schema["entities"],
        possible_relations=final_schema["relations"],
        kg_validation_schema=final_schema["validation_schema"],
        strict=False # 改為 True 可強制符合 Schema，False 則具備彈性
    )
    
    # 直接利用 LlamaIndex 內建的 show_progress 處理全量數據
    PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        kg_extractors=[CustomMetadataExtractor(), llm_extractor],
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        show_progress=True 
    )
    print(">> 階段 2 完成，知識圖譜已成功寫入 Neo4j。")

# --- 執行流程 ---
if __name__ == "__main__":
    # 初始化資料
    documents = data_processing(mode="unstructured_text")
    
    if args.init_neo4j:
        graph_store.delete_all()

    # 1. 定義 Schema
    final_locked_schema = schema_definition_phase(documents, iterations=5, sample_ratio=0.8)
    
    save_schema(final_locked_schema, filepath="/home/End_to_End_RAG/kg_config_schema.json")

    # 2. 全量抽取
    bulk_extraction_phase(documents, final_locked_schema, graph_store)
    # bulk_extraction_phase(documents, current_schema, graph_store)