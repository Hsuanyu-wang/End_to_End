################################################################################################
# vector.py (封裝後)
################################################################################################
import argparse
import os
import copy
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# 模組匯入
from src.config.settings import get_settings
from src.data.processors import data_processing
from src.storage import get_storage_path


# 註解：generation_max_tokens 功能已隱藏
# def _llm_with_generation_cap(llm, max_tokens: int):
#     """嘗試複製 LLM 並注入輸出 token 上限。"""
#     if not max_tokens or max_tokens <= 0:
#         return llm
#     try:
#         capped = copy.deepcopy(llm)
#         if hasattr(capped, "max_tokens"):
#             capped.max_tokens = max_tokens
#         if hasattr(capped, "num_predict"):
#             capped.num_predict = max_tokens
#         if hasattr(capped, "additional_kwargs") and isinstance(capped.additional_kwargs, dict):
#             capped.additional_kwargs["max_tokens"] = max_tokens
#             capped.additional_kwargs["num_predict"] = max_tokens
#         return capped
#     except Exception:
#         return llm

def get_vector_query_engine(Settings, vector_method: str = "vector", top_k: int = 2, data_mode: str = "key_value_text", data_type: str = "DI", fast_build: bool = False, retrieval_max_tokens: int = 2048) -> RetrieverQueryEngine:
    """
    vector_method: 可選 "hybrid", "vector", "bm25"
    top_k: 檢索結果數量
    data_mode: 資料模式，可選 "key_value_text", "natural_text", "unstructured_text"
    data_type: 資料類型，可選 "DI", "GEN"
    retrieval_max_tokens: 檢索內容最大 token 數（限制傳給 LLM 的 context 長度）
    """
    
    # 資料處理
    documents = data_processing(mode=data_mode, data_type=data_type)
    if not documents:
        raise ValueError(f"Failed to load documents. Check if data_mode '{data_mode}' and data_type '{data_type}' are correct.")

    # 使用 StorageManager 取得路徑
    persist_dir = get_storage_path(
        storage_type="vector_index",
        data_type=data_type,
        method=vector_method,
        top_k=top_k,
        fast_test=fast_build
    )
    print(f"📂 Vector Index 儲存路徑: {persist_dir}")

    # 1. 建立或載入 Vector Index
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("正在從本地載入 Vector Index...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("正在建立新的 Vector Index...")
        if fast_build:
            print("⚡ [Vector] 啟用微型建圖模式：僅抽取前 2 筆文本進行快速建圖...")
            documents = documents[:2]
        
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"✅ Vector Index 已儲存至: {persist_dir}")

    # 2. 建立 BM25 Retriever
    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    # 3. 建立 Vector Retriever
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    # 4. 根據參數選擇 Retriever
    if vector_method == "hybrid":
        # 假設您使用的是 QueryFusionRetriever 進行混合檢索
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False # 如果在某些環境下報錯，可以關閉非同步
        )
    elif vector_method == "vector":
        retriever = vector_retriever
    elif vector_method == "bm25":
        retriever = bm25_retriever
    else:
        raise ValueError(f"Invalid vector method: {vector_method}")

    # 5. 端到端生成：將 Retriever 包裝成 QueryEngine
    # 註解：retrieval_max_tokens 的限制將在 VectorRAGWrapper 中實作（截斷過長的 retrieved_contexts）
    response_synthesizer = get_response_synthesizer(response_mode="compact", llm=Settings.llm)
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    
    return query_engine

# 保留單獨執行此檔案的功能
if __name__ == "__main__":
    Settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_method", type=str, default="vector", choices=["hybrid", "vector", "bm25"])
    parser.add_argument("--top_k", type=int, default=2)
    args = parser.parse_args()
    
    # 測試執行
    engine = get_vector_query_engine(Settings, vector_method=args.vector_method, fast_build=True)
    print(f"成功初始化 {args.vector_method} Query Engine!")
    # response = engine.query("測試問題")
    # print(response)