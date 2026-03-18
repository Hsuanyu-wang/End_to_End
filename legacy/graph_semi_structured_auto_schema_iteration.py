import nest_asyncio
nest_asyncio.apply()

import os
import yaml
import json
import argparse
import math
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
    entities: List[str] = Field(description="實體類型列表")
    relations: List[str] = Field(description="關係類型列表")
    validation_schema: Dict[str, List[str]] = Field(description="實體與關係的合法連接規則")

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
    sample_text = "\n\n".join([doc.text[:] for doc in batch_docs]) # 取前 500 字避免 token 過長
    prompt_str = f"你是一個 KG Schema 架構師。現有 Schema: {current_schema}\n請根據以下文本判斷是否需新增實體或關係:\n\"\"\"{sample_text}\"\"\""
    prompt_template = PromptTemplate(template=prompt_str)
    try:
        response_obj = llm.structured_predict(
            DynamicKGSchema,
            prompt=prompt_template,
            schema=json.dumps(current_schema, ensure_ascii=False),
            text=sample_text
        )
        return response_obj.model_dump()
    except Exception as e:
        print(f">> Schema 更新失敗: {e}")
        return current_schema

# --- 初始化 ---
# dynamic_schema = {
#     "entities": [""],
#     "relations": [""],
# }
dynamic_schema = {
    "entities": ["Record", "Engineer", "Customer", "Action", "Issue"],
    "relations": ["BELONGS_TO", "HANDLED", "TAKEN_ACTION", "HAS_ISSUE"],
    "validation_schema": {
        "Record": ["BELONGS_TO", "TAKEN_ACTION", "HAS_ISSUE"],
        "Engineer": ["HANDLED"],
        "Customer": ["BELONGS_TO"]
    }
}

graph_store = SafeNeo4jStore(
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    url=os.getenv("NEO4J_URI"),
    database=config["graph"]["neo4j_database"]
)

if args.init_neo4j:
    graph_store.delete_all()

documents = data_processing(mode="unstructured_text")
total_docs = len(documents)
print(f"Total documents: {total_docs}")

# --- 批次處理與 tqdm 進度條 ---
BATCH_SIZE = max(1, int(total_docs * 0.1))
MAX_EVOLVE_ROUNDS = 5 # 限制進階次數

total_rounds = math.ceil(total_docs / BATCH_SIZE)

EVOLVE_RATIO = 0.5
MAX_EVOLVE_ROUNDS = max(1, int(total_rounds * EVOLVE_RATIO))

print(f"預計總共分 {total_rounds} 輪處理 (每輪 {BATCH_SIZE} 篇文件)。")
print(f"前 {MAX_EVOLVE_ROUNDS} 輪將進行 Schema 動態演化，之後固定。")
print("開始批次處理與 Schema 動態演化...")

# 使用 tqdm 包裹 range 產生總進度條
pbar = tqdm(range(0, len(documents), BATCH_SIZE), desc="Processing Batches")

for i in pbar:
    batch = documents[i : i + BATCH_SIZE]
    round_num = (i // BATCH_SIZE) + 1
    pbar.set_postfix({"Round": round_num, "Schema_Size": len(dynamic_schema["entities"])})
    
    # 步驟 A：Schema 進化
    if round_num <= MAX_EVOLVE_ROUNDS:
        old_entities = set(dynamic_schema["entities"])
        dynamic_schema = evolve_schema_with_pydantic(dynamic_schema, batch, Settings.llm)
        new_entities = set(dynamic_schema["entities"])
        
        # 輸出本輪 Schema 狀態
        print(f"\n--- Round {round_num} Schema 報告 ---")
        print(f"新增實體: {new_entities - old_entities}")
        print(f"目前所有實體: {dynamic_schema['entities']}")
        print(f"目前所有關係: {dynamic_schema['relations']}")
        print("-" * 30)
    else:
        print(">> 已超過演化輪數上限，使用固定 Schema 進行抽取。")
        
    # 步驟 B：建立 Extractor
    llm_extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=dynamic_schema["entities"],
        possible_relations=dynamic_schema["relations"],
        kg_validation_schema=dynamic_schema["validation_schema"],
        strict=False
    )
    
    # 步驟 C：寫入圖譜
    PropertyGraphIndex.from_documents(
        batch,
        property_graph_store=graph_store,
        kg_extractors=[CustomMetadataExtractor(), llm_extractor],
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        show_progress=False # 關閉內部進度條以免與 tqdm 衝突
    )

print("\n>> 處理完成！最終 Schema 已收斂。")

# --- 查詢測試 ---
# graph_index = PropertyGraphIndex.from_documents(
#     [], property_graph_store=graph_store, embed_model=Settings.embed_model
# )
# query_engine = graph_store.as_query_engine()

graph_index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    embed_model=Settings.embed_model,
    llm=Settings.llm
)
query_engine = graph_index.as_query_engine()
response = query_engine.query("C客戶現在的splunk版本是什麼")
print(f"Response: {response}")