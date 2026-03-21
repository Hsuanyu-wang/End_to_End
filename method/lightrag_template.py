import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# 1. 設定儲存目錄
WORKING_DIR = "/home/End_to_End_RAG/lightrag_backbone_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

OLLAMA_URL = "http://192.168.63.174:11434"

async def main():
    # 2. 初始化 LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        
        # --- LLM 設定 ---
        llm_model_func=ollama_model_complete,
        llm_model_name='qwen2.5:7b',
        llm_model_kwargs={"host": OLLAMA_URL, "options": {"num_ctx": 32768}}, 
        
        # --- Embedding 設定 ---
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed.func(
                texts, 
                embed_model="nomic-embed-text",
                host=OLLAMA_URL,
            )
        ),
        
        # from lightrag.utils import wrap_embedding_func_with_attrs

        # @wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
        # async def custom_embedding_func(texts: list[str]) -> np.ndarray:
        #     return await ollama_embed.func(texts, embed_model="nomic-embed-text", host=OLLAMA_URL)
    
    )

    # 【重要】3. 顯式初始化儲存組件，解決你的錯誤訊息
    await rag.initialize_storages()

    # 4. 準備測試文本
    sample_text = """
    LightRAG 是一種輕量級、快速的檢索增強生成系統。
    它結合了知識圖譜 (Knowledge Graph) 的優勢，能夠進行雙層檢索，
    從而大幅提升對複雜關聯性問題的理解與回答能力。
    Ollama 是一個可以讓你在本地輕鬆運行大型語言模型的開源工具。
    """
    
    # 5. 插入文本 (非同步)
    print("正在處理文本並建立索引，這可能需要一些時間...")
    # 這裡直接傳入字串，或從檔案讀取
    await rag.ainsert(sample_text)

    # 6. 進行查詢測試
    query_text = "LightRAG 是什麼？它與傳統工具有什麼不同？"
    print(f"\n問題: {query_text}\n")

    # 使用 hybrid 模式 (向量 + 圖譜)
    response = await rag.aquery(
        query_text, 
        param=QueryParam(mode="hybrid")
    )

    print("回答:")
    print(response)

if __name__ == "__main__":
    # 執行非同步主程式
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n使用者停止執行。")