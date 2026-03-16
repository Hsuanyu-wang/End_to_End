import os
import json
import numpy as np
import nest_asyncio
import asyncio
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# from lightrag.llm import gpt_4o_mini_complete, openai_embedding

# 若您有自定義的 LLM (如 Ollama, vLLM)，可參考 LightRAG 官方文件替換此處的 llm_model_func
# 這裡預設使用 openai_complete 作為範例
from model_settings import get_settings # 保留您的設定模組以便後續自定義擴充
from data_processing import data_processing

nest_asyncio.apply()

# 嘗試匯入 LightRAG 內建的 LlamaIndex 適配器 (建議更新至最新版 LightRAG 以支援)
try:
    from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
    HAS_LLAMA_INDEX_IMPL = True
except ImportError:
    HAS_LLAMA_INDEX_IMPL = False

def get_lightrag_engine(Settings, data_type: str = "DI", sup: str = ""):
    # 【修正 1】精準抓取 Embedding 維度，強制以模型實際產出的向量長度為準
    # 忽略 Settings.embed_model.embedding_dim 的潛在錯誤數值
    print("正在測量 Embedding 實際維度...")
    dummy_emb = Settings.embed_model.get_text_embedding("test_dimension")
    EMBEDDING_DIM = len(dummy_emb)
    print(f"檢測到 Embedding 維度為: {EMBEDDING_DIM}")

    # if HAS_LLAMA_INDEX_IMPL:
    #     # 寫法 A：使用 LightRAG 官方的 LlamaIndex 適配器 (自帶緩存機制)
    #     async def custom_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    #         # 【修正 2】LightRAG 內部會傳入快取用的 hashing_kv 參數，LlamaIndex Adapter 若不支援會報錯，需過濾掉
    #         kwargs.pop("hashing_kv", None)
    #         kwargs.pop("keyword_extraction", None)
    #         return await llama_index_complete_if_cache(
    #             Settings.llm, 
    #             prompt, 
    #             system_prompt=system_prompt, 
    #             history_messages=history_messages, 
    #             **kwargs
    #         )
            
    #     custom_embed_func = EmbeddingFunc(
    #         embedding_dim=EMBEDDING_DIM,
    #         max_token_size=8192,
    #         func=lambda texts: llama_index_embed(texts, embed_model=Settings.embed_model)
    #     )
    # else:
    # 寫法 B：手動包裝非同步函式
    async def custom_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        # 呼叫 LlamaIndex 的非同步生成方法
        response = await Settings.builder_llm.acomplete(full_prompt)
        return response.text

    # async def perform_ontology_learning(Settings):
    #     # 抽取部分樣本
    #     texts = data_processing(mode="natural_text", data_type=data_type)
    #     sample_text = "\n".join(texts[:5]) 
    #     prompt = f"""
    #     分析以下文本，並列出 5-10 個核心實體類別（如：伺服器、軟體版本、客戶名稱）
    #     以及它們之間可能的關係。請輸出為 JSON 格式。
    #     文本內容：{sample_text}
    #     """
    #     response = await Settings.llm.acomplete(prompt)
    #     # 解析 response 並更新 settings.lightrag_entity_types
    #     # 這裡可以根據 LLM 回傳的結果動態調整
    #     return response.text

    async def manual_embed_func(texts: list[str]) -> np.ndarray:
        # 確保傳入的是 list
        if isinstance(texts, str):
            texts = [texts]
            
        # 呼叫 LlamaIndex 的非同步批次向量化方法
        embeddings = await Settings.embed_model.aget_text_embedding_batch(texts)
        
        # 轉換為 numpy array，LightRAG 預期維度為 (N, embedding_dim)
        emb_array = np.array(embeddings)
        return emb_array

    custom_embed_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=manual_embed_func
    )

    # 將「包裝後的函式」傳入 LightRAG，避開深度複製物件的報錯
    rag = LightRAG(
        working_dir=os.path.join(Settings.lightrag_storage_path_DIR, data_type + ("_" + sup if sup else "")),
        llm_model_func=custom_llm_func,
        embedding_func=custom_embed_func,
        addon_params={
            # "language": Settings.lightrag_language,
            "entity_types": Settings.lightrag_entity_types
            # DEFAULT_ENTITY_TYPES = [
                # "Person",
                # "Creature",
                # "Organization",
                # "Location",
                # "Event",
                # "Concept",
                # "Method",
                # "Content",
                # "Data",
                # "Artifact",
                # "NaturalObject",
            # ]
        }
    )
    
    loop = asyncio.get_event_loop()
    if hasattr(rag, "initialize_storages"):
        loop.run_until_complete(rag.initialize_storages())
    
    return rag

def build_lightrag_index(Settings, mode: str = "natural_text", data_type: str = "DI", sup: str = "", fast_build: bool = False) -> None:
    """
    執行資料插入與圖譜建立
    """
    rag = get_lightrag_engine(Settings, data_type=data_type, sup=sup)
    print("正在處理文本並匯入 LightRAG (此過程包含 LLM 實體與關係抽取，可能較久)...")
    # 修正 3：因為 data_processing 回傳的可能是 LlamaIndex 的 Document 物件，
    # 而 LightRAG 的 insert 需要的是字串列表 (list[str])
    docs = data_processing(mode=mode, data_type=data_type)
    if not docs:
        raise ValueError(f"Failed to load documents. Check if data_mode '{mode}' and data_type '{data_type}' are correct.")
    if docs is not None:
        if fast_build:
            print("⚡ [LightRAG] 啟用微型建圖模式：僅抽取前 2 筆文本進行快速建圖...")
            docs = docs[:2]
        else:
            print("啟用完整建圖模式，進行完整文本建圖...")
        texts = [doc.text for doc in docs] if hasattr(docs[0], "text") else docs
        rag.insert(texts)
    else:
        print("Error: No documents to insert")
        return None
    
    print(f"成功匯入 {len(texts)} 筆維護紀錄！")

if __name__ == "__main__":
    Settings = get_settings()
    # 指定您的原始檔案路徑
    # raw_file_path = Settings.raw_file_path
    
    # 建立索引 (只需執行一次)
    build_lightrag_index(Settings, mode="natural_text", data_type="DI", sup="", fast_build=True)

    # 測試查詢
    rag = get_lightrag_engine(Settings)
    
    query_text = "C客戶現在的splunk版本是什麼？" # 來自 QA_43.csv 的 Q1
    
    # 支援四種檢索模式: naive (純向量), local (局部圖譜), global (全局圖譜), hybrid (混合)
    print("=== Hybrid Query ===")
    response = rag.query(query_text, param=QueryParam(mode="hybrid", enable_rerank=False))
    print(response)