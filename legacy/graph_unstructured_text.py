import nest_asyncio
nest_asyncio.apply()

import os
from llama_index.core import StorageContext, load_index_from_storage, PropertyGraphIndex
import yaml
from model_settings import get_settings
Settings = get_settings()

from data_processing import data_processing
documents = data_processing(mode="unstructured_text") # [natural_text, key_value_text, unstructured_text]

# STORAGE_DIR = "/home/End_to_End_RAG/storage_graph"
config = yaml.safe_load(open("/home/End_to_End_RAG/config.yml"))
STORAGE_DIR = config["graph"]["storage_dir"]

if os.path.exists(STORAGE_DIR):
    print("正在從本地讀取索引...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    graph_index = load_index_from_storage(storage_context)
else:
    print("正在建立新的屬性圖索引 (此過程可能較久)...")
    try:
        graph_index = PropertyGraphIndex.from_documents(
            documents,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            show_progress=True,
            embed_batch_size=2, # 稍微調低觀察穩定度
# allowed_nodes=[
        #     "Record", "Customer", "Engineer", "MaintenanceCategory", "Action", "Issue", "ActionCategory", "ActionSubcategory", "System", "Vulnerability", "Software", "Server",
        # ],
        # allowed_relationships=[
        #     "HANDLED", "BELONGS_TO", "HAS_MAINTENANCE_CATEGORY", "TAKEN_ACTION", "HAS_ISSUE", "REPORTS_LICENSE", "HAS_ACTION_CATEGORY", "RESOLVES", "TRIGGERED_BY", "CAUSED_BY", "INVOLVES_SERVER", "INVOLVES_SOFTWARE", "INVOLVES_CVE", "INVOLVES_RULE", "DERIVED_FROM"
        # ],
        # node_properties=[
        #     "id", "start_time", "end_time", "description", "name", "status", "hostname", "ip", "current_version", "latest_version", "cve_id", "detail", "version", "status"
        # ]
        )
        graph_index.storage_context.persist(persist_dir=STORAGE_DIR)
        print("索引建立並儲存完成。")
    except Exception as e:
        print(f"建立索引時發生錯誤: {e}")


# 查詢
query_engine = graph_index.as_query_engine(include_text=True)
print("query_engine: ", query_engine)
response = query_engine.query("C客戶現在的splunk版本是什麼")
print("response: ", response)