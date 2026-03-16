################################################################################################
# END-TO-END VECTOR RAG
################################################################################################
from llama_index.core import Document, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# 模型設定
from model_settings import get_settings
Settings = get_settings()

# 資料處理
from data_processing import data_processing
documents = data_processing(mode="key_value_text") # [natural_text, key_value_text, unstructured_text]

# 參數設定
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--vector_method", type=str, default="hybrid", choices=["hybrid", "vector", "bm25"])
args = parser.parse_args()
vector_method = args.vector_method

################################################################################################
# END-TO-END VECTOR RETRIEVAL
################################################################################################

# 2. 建立 Vector Index (現在會使用 Ollama 進行 Embedding)
index = VectorStoreIndex.from_documents(documents)

# 3. 建立 BM25 Retriever
# 注意：BM25 需要先取得 nodes，我們從 index 的儲存層直接拿取
nodes = index.docstore.docs.values()
bm25_retriever = BM25Retriever.from_defaults(nodes=list(nodes), similarity_top_k=2)
### 測試stemmer="zh"是否有效
# from pystemmer import Stemmer
# stemmer = Stemmer(language="zh")
# bm25_retriever = BM25Retriever.from_defaults(nodes=list(nodes), similarity_top_k=2, stemmer=stemmer)
### 使用jieba分詞 + 屬性
# from jieba import posseg
# bm25_retriever = BM25Retriever.from_defaults(nodes=list(nodes), similarity_top_k=2, tokenizer=posseg.cut)
### 使用jieba分詞
# import jieba
# def chinese_tokenizer(text: str):
#     return jieba.lcut(text, cut_all = False)

# bm25_retriever = BM25Retriever.from_defaults(
#     nodes=nodes,
#     tokenizer=chinese_tokenizer # 確保 BM25 看得懂中文詞
# )

# 4. 建立 Vector Retriever
vector_retriever = index.as_retriever(similarity_top_k=2)

# 5. 實現 Hybrid Search (使用 RRF 混合)
hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False # 如果在某些環境下報錯，可以關閉非同步
)

# 執行檢索
query = "C客戶現在的splunk版本是什麼" # C客戶現在的splunk版本是什麼, C客戶關於 GitLab 漏洞修補的記錄
if vector_method == "hybrid":
    retriever = hybrid_retriever
elif vector_method == "vector":
    retriever = vector_retriever
elif vector_method == "bm25":
    retriever = bm25_retriever
else:
    raise ValueError(f"Invalid vector method: {vector_method}")

# results = retriever.retrieve(query)

# for i, res in enumerate(results):
#     print(f"--- Result {i+1} ---")
#     print(f"Score: {res.get_score()}")
#     print(f"Text: {res.node.get_content()[:200]}...")

################################################################################################
# END-TO-END GENERATION
################################################################################################

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# 接續您原有的 hybrid_retriever 代碼
# 6. 端到端生成：將 Retriever 包裝成 QueryEngine
# get_response_synthesizer 可以自定義生成策略，例如 "compact", "tree_summarize" 或 "refine"
response_synthesizer = get_response_synthesizer(response_mode="compact") # "compact", "tree_summarize", "refine", "no_text", "context_only", "simple_summarize", "generation", "accumulate", "compact_accumulate"

query_engine = RetrieverQueryEngine(
    retriever=retriever, # vector_retriever, bm25_retriever, hybrid_retriever
    response_synthesizer=response_synthesizer,
)

# 執行查詢
response = query_engine.query(query)

# 查看來源 (Citations)
print("\n--- Sources ---")
for node in response.source_nodes:
    print(f"NO: {node.metadata.get('service_no')}, Score: {node.score:.4f}")
    
print(f"--- 最終生成答案 ---")
print(response.response)