################################################################################################
# vector.py (封裝後)
################################################################################################
import argparse
from llama_index.core import Document, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# 模組匯入
from model_settings import get_settings
from data_processing import data_processing

def get_vector_query_engine(Settings, vector_method: str = "vector", top_k: int = 2, data_mode: str = "key_value_text", data_type: str = "DI", fast_build: bool = False) -> RetrieverQueryEngine:
    """
    vector_method: 可選 "hybrid", "vector", "bm25"
    top_k: 檢索結果數量
    data_mode: 資料模式，可選 "key_value_text", "natural_text", "unstructured_text"
    data_type: 資料類型，可選 "DI", "GEN"
    """

    # Settings = get_settings()
    
    # 資料處理
    documents = data_processing(mode=data_mode, data_type=data_type)
    if not documents:
            raise ValueError(f"Failed to load documents. Check if data_mode '{data_mode}' and data_type '{data_type}' are correct.")

    # 1. 建立 Vector Index
    if fast_build:
        print("⚡ [Vector] 啟用微型建圖模式：僅抽取前 2 筆文本進行快速建圖...")
        documents = documents[:2]
    else:
        print("啟用完整建圖模式，進行完整文本建圖...")
        
    index = VectorStoreIndex.from_documents(documents)

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
    response_synthesizer = get_response_synthesizer(response_mode="compact")
    
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