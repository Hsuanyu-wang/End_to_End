import os
import json
import asyncio
import pandas as pd
from tqdm import tqdm
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# === 嘗試引入 RAGAS 評估套件 ===
try:
    from datasets import Dataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy, 
        Faithfulness, 
        ContextPrecision, 
        ContextRecall
    )
    from ragas.run_config import RunConfig
    # from langchain_community.chat_models import ChatOllama
    # from langchain_community.embeddings import OllamaEmbeddings
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    print("未安裝 ragas 或相關套件。若要進行評估，請先執行: pip install ragas datasets langchain-community openpyxl")


# 1. 設定儲存目錄與 Ollama URL
WORKING_DIR = "/home/End_to_End_RAG/method/lightrag_backbone_storage"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

OLLAMA_URL = "http://192.168.63.184:11434"

# [新增控制選項] 定義你要一次跑完的 6 種 query mode
# 請依照 LightRAG 實際支援與你需要的模式進行修改
QUERY_MODES = ["hybrid", "mix"]  # "naive", "local", "global", 

async def main():
    # 2. 初始化 LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name='qwen2.5:14b',
        llm_model_kwargs={"host": OLLAMA_URL, "options": {"num_ctx": 32768}}, 
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed.func(
                texts, 
                embed_model="nomic-embed-text",
                host=OLLAMA_URL,
            )
        )
    )

    # 3. 顯式初始化儲存組件
    await rag.initialize_storages()

    # 4. 準備測試文本 (Docs)
    print("正在讀取文件資料...")
    docs_path = "/home/End_to_End_RAG/Data/CSR_20260122_v4_92_clean.jsonl"
    
    # # [修改點 1] 改用 list 來存放每筆文件，而不是全部拼成一個超大字串
    # # 這樣有助於 LightRAG 進行更細粒度的文件 Hash 比對與斷點續傳
    # docs_list = []
    # with open(docs_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         clean_line = line.strip()
    #         if clean_line:
    #             # 【修改點】直接將原始字串加入 list，避免 json.dumps 造成格式微調與 Hash 改變
    #             docs_list.append(clean_line)
    
    # 5. 插入文本
    has_cache = any(os.path.isfile(os.path.join(WORKING_DIR, f)) for f in os.listdir(WORKING_DIR))

    if not has_cache:
        print("未偵測到快取檔案，正在處理文本並建立/補全索引，這可能需要一些時間...")
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    await rag.ainsert(clean_line)
    else:
        print("偵測到快取檔案，跳過插入文本，直接進入檢索 (Retrieval)...")

    # 6. 讀取測試問題集 (Query & Ground Truth)
    print("正在讀取測試問題...")
    query_path = "/home/End_to_End_RAG/Data/QA_43_clean.jsonl"
    qa_pairs = []
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))

    # [修改點] 在外層加入 Mode 的迴圈
    for current_mode in QUERY_MODES:
        print(f"\n" + "="*60)
        print(f"🚀 正在執行查詢模式 (Query Mode): 【 {current_mode.upper()} 】")
        print("="*60)

        # 每次切換模式時，重新初始化評估資料收集字典
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        # 開始針對當前模式進行批次查詢
        for i, qa in enumerate(tqdm(qa_pairs, desc=f"QA 進度 ({current_mode})", unit="題", ncols=100)):
            question = qa.get("Q", "")
            ground_truth = qa.get("GT", "")
            
            if not question:
                continue
                
            tqdm.write(f"\n問題{i}: {question}")
            
            # [修改點] 將 param 中的 mode 替換為變數 current_mode
            retrieved_context = await rag.aquery(
                question, 
                param=QueryParam(mode=current_mode, only_need_context=True, enable_rerank=False)
            )
            
            response = await rag.aquery(
                question, 
                param=QueryParam(mode=current_mode, enable_rerank=False)
            )
            
            tqdm.write(f"回答: {response[:150]}...") 

            # 收集進入評估清單
            eval_data["question"].append(question)
            eval_data["answer"].append(response)
            eval_data["ground_truth"].append(ground_truth)
            
            if isinstance(retrieved_context, str):
                eval_data["contexts"].append([retrieved_context])
            elif isinstance(retrieved_context, list):
                eval_data["contexts"].append([str(c) for c in retrieved_context])
            else:
                eval_data["contexts"].append([str(retrieved_context)])

        # 7. 針對當前模式使用 Ragas 進行評估
        if HAS_RAGAS:
            print(f"\n⏳ 開始進行 RAGAS 評估 (模式: {current_mode})...")
            
            # eval_llm = ChatOllama(
            #     model="qwen2.5:14b", 
            #     base_url=OLLAMA_URL, 
            #     timeout=300,
            #     # format="json",      # 強制 Ollama 輸出 JSON 格式
            #     temperature=0.0     # 將隨機性降到最低，確保格式穩定
            # )
            # eval_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
            
            langchain_llm = ChatOllama(
                model="qwen2.5:14b", 
                base_url=OLLAMA_URL, 
                num_ctx=16384,
                timeout=300,
                temperature=0.0,
                # format="json"
            )
            langchain_embeddings = OllamaEmbeddings(
                model="nomic-embed-text", 
                base_url=OLLAMA_URL
            )
            ragas_llm = LangchainLLMWrapper(langchain_llm)
            ragas_emb = LangchainEmbeddingsWrapper(langchain_embeddings)

            dataset = Dataset.from_dict(eval_data)
            
            metrics = [
                AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),    
                Faithfulness(llm=ragas_llm),       
                ContextPrecision(llm=ragas_llm),   
                ContextRecall(llm=ragas_llm)   
            ]
            
            custom_run_config = RunConfig(
                timeout=600,       # 設定 Ragas 的超時時間 (秒)
                max_workers=1,     # 限制並行數量為 1 (最穩定的設定，但速度較慢)
                max_retries=5      # 若超時允許重試的次數
            )
            
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_emb,
                run_config=custom_run_config
            )
            
            print(f"\n📊 評估結果 ({current_mode}):")
            print(results)
            
            # [修改點] 動態生成檔名，避免不同模式的結果互相覆蓋
            df = results.to_pandas()
            output_csv_path = f"/home/End_to_End_RAG/method/result/lightrag_evaluation_{current_mode}.csv"
            
            df.to_csv(output_csv_path, index=False)
            print(f"✅ 模式 {current_mode} 的評估結果已儲存至 {output_csv_path}\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n使用者停止執行。")