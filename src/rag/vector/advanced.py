import os
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever, AutoMergingRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from src.data.processors import data_processing
    
def get_self_query_engine(mySettings, data_mode="natural_text", data_type="DI", top_k=2, fast_build=False, retrieval_max_tokens=2048):
    """
    實作 1: Self-Querying RAG (Metadata Filtering)
    
    Args:
        retrieval_max_tokens: 檢索內容最大 token 數（實際限制在 wrapper 層處理）
    """
    documents = data_processing(mode=data_mode, data_type=data_type)
    if fast_build:
        documents = documents[:2]

    index = VectorStoreIndex.from_documents(documents)
    
    # 這裡必須根據你的文本 Metadata 實際欄位進行定義
    # LlamaIndex 會根據這些描述讓 LLM 自動推論並產生 Query Filter
    vector_store_info = VectorStoreInfo(
        content_info="企業維護紀錄與客服問答集",
        metadata_info=[
            MetadataInfo(name="NO", type="str", description="文件的唯一識別碼"),
            MetadataInfo(name="Customer", type="str", description="客戶名稱"),
            MetadataInfo(name="Engineers", type="str", description="工程師名稱"),
            MetadataInfo(name="Service Start", type="str", description="服務開始時間"),
            MetadataInfo(name="Service End", type="str", description="服務結束時間"),
            MetadataInfo(name="Description", type="str", description="描述"),
            MetadataInfo(name="Action", type="str", description="動作"),            
        ]
    )
    
    retriever = VectorIndexAutoRetriever(
        index,
        vector_store_info=vector_store_info,
        llm=mySettings.llm,
        similarity_top_k=top_k,
        verbose=True
    )
    
    return RetrieverQueryEngine.from_args(retriever=retriever, llm=mySettings.llm)

def get_parent_child_query_engine(mySettings, data_mode="natural_text", data_type="DI", top_k=2, fast_build=False, retrieval_max_tokens=2048):
    """
    實作 2: Multi-Vector / Parent-Child RAG (Auto Merging)
    
    Args:
        retrieval_max_tokens: 檢索內容最大 token 數（實際限制在 wrapper 層處理）
    """
    documents = data_processing(mode=data_mode, data_type=data_type)
    if fast_build:
        documents = documents[:2]

    # 1. 建立階層式切塊解析器 (Parent -> Child -> Grandchild)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 1024, 512]
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    # 2. 將所有 Nodes (包含 Parent) 存入 Document Store
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    # 3. 僅對最底層的 Leaf Nodes 建立向量索引
    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

    # 4. 設定 Auto Merging Retriever
    base_retriever = index.as_retriever(similarity_top_k=top_k * 3)
    retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context, 
        verbose=True
    )

    return RetrieverQueryEngine.from_args(retriever=retriever, llm=mySettings.llm)