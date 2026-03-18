import nest_asyncio
nest_asyncio.apply()

import os
import yaml
from dotenv import load_dotenv
from typing import Literal
# LlamaIndex 核心套件
from llama_index.core import StorageContext, PropertyGraphIndex
from llama_index.core.schema import TransformComponent
from llama_index.core.graph_stores import EntityNode, Relation
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# 載入自定義設定與資料
from model_settings import get_settings
from data_processing import data_processing

# 1. 載入環境變數與設定
load_dotenv() # 自動讀取 .env 檔案
Settings = get_settings()
config = yaml.safe_load(open("/home/End_to_End_RAG/config.yml"))
STORAGE_DIR = config["graph"]["storage_dir"]

documents = data_processing(mode="unstructured_text") # [natural_text, key_value_text, unstructured_text]

# ==========================================
# 2. 定義自定義抽取器 (處理結構化 Metadata)
# ==========================================
class CustomMetadataExtractor(TransformComponent):
    """
    這個抽取器完全不使用 LLM，直接將 Document 中的 Metadata 
    轉換成圖譜的實體(EntityNode)與關聯(Relation)。
    """
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            # LlamaIndex 預設用這些 key 來暫存抽出的圖譜資料
            if "kg_nodes" not in node.metadata:
                node.metadata["kg_nodes"] = []
            if "kg_relations" not in node.metadata:
                node.metadata["kg_relations"] = []
            
            # 從你的資料中取得固定 metadata (請依據你的實際 key 名稱調整)
            record_id = node.metadata.get("NO", "Unknown")
            customer = node.metadata.get("Customer")
            engineer = node.metadata.get("Engineers")
            m_category = node.metadata.get("維護類別")
            start_time = node.metadata.get("Service Start")
            end_time = node.metadata.get("Service End")
            
            # (A) 建立核心 RECORD 節點 (時間作為屬性存入)
            record_node = EntityNode(
                name=f"Record_{record_id}", 
                label="Record", 
                properties={
                    "id": record_id,
                    "service_start": start_time,
                    "service_end": end_time
                }
            )
            node.metadata["kg_nodes"].append(record_node)

            # (B) 建立 CUSTOMER 節點與關係
            if customer:
                customer_node = EntityNode(name=customer, label="Customer")
                rel_customer = Relation(
                    source_id=record_node.id, 
                    target_id=customer_node.id, 
                    label="BELONGS_TO"
                )
                node.metadata["kg_nodes"].append(customer_node)
                node.metadata["kg_relations"].append(rel_customer)
            
            # (C) 建立 ENGINEER 節點與關係
            if engineer:
                engineer_node = EntityNode(name=engineer, label="Engineer")
                rel_engineer = Relation(
                    source_id=engineer_node.id, 
                    target_id=record_node.id, 
                    label="HANDLED"
                )
                node.metadata["kg_nodes"].append(engineer_node)
                node.metadata["kg_relations"].append(rel_engineer)
                
            # (D) 建立 維護類別 節點與關係
            if m_category:
                cat_node = EntityNode(name=m_category, label="MaintenanceCategory")
                rel_cat = Relation(
                    source_id=record_node.id, 
                    target_id=cat_node.id, 
                    label="HAS_MAINTENANCE_CATEGORY"
                )
                node.metadata["kg_nodes"].append(cat_node)
                node.metadata["kg_relations"].append(rel_cat)

            # (E) 限制傳遞給 LLM 的文本 (可選)
            # 確保 node.text 只有 Description 和 Action，避免 LLM 重複抽取 Metadata
            # node.text = f"Description: {node.metadata.get('Description', '')}\nAction: {node.metadata.get('Action', '')}"

        return nodes

# ==========================================
# 3. 定義 LLM 抽取器 (處理非結構化文本)
# ==========================================
# 嚴格定義 Schema，限制 LLM 只能抽取這些節點與關係

AllowedEntities = Literal[
    "Action", "Issue", "ActionCategory", "ActionSubcategory", 
    "System", "Vulnerability", "Software", "Server"
]

AllowedRelations = Literal[
    "TAKEN_ACTION", "HAS_ISSUE", "HAS_ACTION_CATEGORY", 
    "RESOLVES", "TRIGGERED_BY", "CAUSED_BY", "INVOLVES_SERVER", 
    "INVOLVES_SOFTWARE", "INVOLVES_SYSTEM", "INVOLVES_VULNERABILITY"
    "RELATED_TO"
]

# 2. 建議加入 kg_validation_schema (驗證規則)
# 這會告訴 LLM 「哪些節點」可以連接「哪些關係」，大幅降低 LLM 亂抽的機率
validation_schema = {
    "Engineer": ["HANDLED"],
    "Record": ["BELONGS_TO", "HAS_MAINTENANCE_CATEGORY", "TAKEN_ACTION", "HAS_ISSUE"],
    "Action": ["HAS_ACTION_CATEGORY", "RESOLVES", "TRIGGERED_BY", "INVOLVES_SERVER", "INVOLVES_SOFTWARE", "INVOLVES_VULNERABILITY"],
    "Issue": ["CAUSED_BY", "INVOLVES_SERVER", "INVOLVES_SOFTWARE", "INVOLVES_SYSTEM", "INVOLVES_VULNERABILITY"],
    "Customer": ["BELONGS_TO"],
    "MaintenanceCategory": ["HAS_MAINTENANCE_CATEGORY"],
    "System": ["INVOLVES_SYSTEM"],
    "Vulnerability": ["INVOLVES_VULNERABILITY"],
    "Software": ["INVOLVES_SOFTWARE"],
    "Server": ["INVOLVES_SERVER"],
}

llm_extractor = SchemaLLMPathExtractor(
    llm=Settings.llm,
    possible_entities=AllowedEntities,
    possible_relations=AllowedRelations,
    kg_validation_schema=validation_schema, # 加入驗證規則
    max_triplets_per_chunk=10,
    strict=True # 強制 LLM 遵守 Schema
)

# ==========================================
# 4. 初始化 Neo4j Graph Store
# ==========================================
graph_store = Neo4jPropertyGraphStore(
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    url=os.getenv("NEO4J_URI"),
    database=config["graph"]["neo4j_database"]
)

# ==========================================
# 5. 建立或讀取 Property Graph Index
# ==========================================
print("開始連接 Neo4j 並處理圖譜索引...")

try:
    # 這裡的邏輯是：如果 Neo4j 裡面有資料，LlamaIndex 會自動接上
    # 如果你要強制重新建構，可以檢查 documents 是否已存在
    graph_index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        kg_extractors=[
            CustomMetadataExtractor(), # 第一步：規則抽取 Metadata
            llm_extractor              # 第二步：LLM 抽取文本
        ],
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        show_progress=True,
        embed_batch_size=2
    )
    print("索引建立並同步至 Neo4j 完成。")
except Exception as e:
    print(f"建立索引時發生錯誤: {e}")

# ==========================================
# 6. 查詢測試
# ==========================================
query_engine = graph_index.as_query_engine(include_text=True)
print("query_engine Ready.")

response = query_engine.query("列出所有處理過 B 客戶 KVstore 異常問題的工程師與其相關 Action")
print("Response: ", response)