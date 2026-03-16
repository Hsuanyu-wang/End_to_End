################################################################################################
# graph_unstructured_text.py (封裝後)
################################################################################################
import os
import yaml
import nest_asyncio
from typing import Dict, Any
from llama_index.core import StorageContext, load_index_from_storage, PropertyGraphIndex, KnowledgeGraphIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from typing import Optional

# 模組匯入
from model_settings import get_settings
from data_processing import data_processing
from graph_retriever import CSRGraphQueryEngine

nest_asyncio.apply()

INDEX_METHODS: Dict[str, Any] = {
    "propertyindex": PropertyGraphIndex,
    "ontology_learning": KnowledgeGraphIndex,  # 未來擴充用
}

def get_graph_query_engine(
    Settings,
    data_mode: str = "unstructured_text",
    data_type: str = "DI",
    fast_build: bool = False,
    graph_method: str = "propertyindex",
    *,
    top_k: int = 2,
    use_vector_docs: bool = False,
    doc_entity_mode: str = "metadata_only",
    use_schema_hint: bool = False,
) -> Optional[BaseQueryEngine]:
    """
    建立並回傳屬性圖 (Property Graph) 的 BaseQueryEngine
    """
    # Settings = get_settings()
    documents = data_processing(mode=data_mode, data_type=data_type)
    if not documents:
        raise ValueError(f"Failed to load documents. Check if data_mode '{data_mode}' and data_type '{data_type}' are correct.")

    # CSR methods：走 NetworkX graph + expand，不走 llama_index 的 persist_dir
    if graph_method in ("csr_khop", "csr_bridge"):
        if fast_build:
            print(f"⚡ [{graph_method}] 啟用微型模式：僅抽取前 2 筆文本建圖（cache 仍會依方法名隔離）...")
        else:
            print(f"啟用完整模式，建構 CSR {graph_method} 圖譜（cache 依方法名隔離）...")
        return CSRGraphQueryEngine(
            Settings,
            method_name=graph_method,
            data_mode=data_mode,
            data_type=data_type,
            documents=documents,
            fast_build=fast_build,
            top_k=top_k,
            use_vector_docs=use_vector_docs,
            doc_entity_mode=doc_entity_mode,
            use_schema_hint=use_schema_hint,
        )

    if fast_build:
        print("⚡ [PropertyGraph] 啟用微型建圖模式：僅抽取前 2 筆文本進行快速建圖...")
        documents = documents[:2]
    else:
        print("啟用完整建圖模式，進行完整文本建圖...")

    # 讀取設定檔
    try:
        config = yaml.safe_load(open("/home/End_to_End_RAG/config.yml"))
        STORAGE_DIR = config["graph"]["storage_dir"]
    except Exception as e:
        print(f"讀取 config.yml 失敗，使用預設路徑: {e}")
        STORAGE_DIR = "./storage_graph" # 提供預設路徑防呆
    STORAGE_DIR += '_' + data_type + '_' + graph_method
    if fast_build:
        STORAGE_DIR = STORAGE_DIR + "_fast_test"
        print(f"🛡️ 已將微型圖譜儲存路徑重導向至: {STORAGE_DIR}")

    if os.path.exists(STORAGE_DIR):
        print("正在從本地讀取圖索引...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        graph_index = load_index_from_storage(storage_context)
    else:
        print("正在建立新的屬性圖索引 (此過程可能較久)...")
        if graph_method == "propertyindex":
            # llm_extractor = SchemaLLMPathExtractor(
            #     llm=Settings.llm,
            #     possible_entities=final_schema["entities"],
            #     possible_relations=final_schema["relations"],
            #     kg_validation_schema=Settings.kg_validation_schema,
            #     strict=False # 改為 True 可強制符合 Schema，False 則具備彈性
            # )
            graph_index = PropertyGraphIndex.from_documents(
            documents,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            show_progress=True,
            embed_batch_size=2,
            # kg_extractors=[
            #     CustomMetadataExtractor(),
            #     llm_extractor,
            # ],
            )
        graph_index.storage_context.persist(persist_dir=STORAGE_DIR)
        
    # 將 Graph Index 轉換為 Query Engine 回傳
    return graph_index.as_query_engine()

# 保留單獨執行此檔案的功能
if __name__ == "__main__":
    Settings = get_settings()
    engine = get_graph_query_engine(Settings, fast_build=True)
    print("成功初始化 Property Graph Query Engine!")
    # response = engine.query("測試問題")
    # print(response)