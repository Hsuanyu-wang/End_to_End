import os
import time
from typing import Dict, Any
# 官方 Microsoft GraphRAG API (需確保已初始化並建立好 GraphRAG 專案)
from graphrag.query.cli import run_global_search, run_local_search

class MSGraphRAGWrapper:
    """
    實作 3: 針對 Microsoft End-to-End GraphRAG 的評估 Wrapper
    """
    def __init__(self, name: str, root_dir: str, mode="local"):
        self.name = name
        self.root_dir = root_dir # GraphRAG 的初始化目錄 (包含 output 與 config)
        self.mode = mode # "local" 或 "global"

    async def aquery_and_log(self, user_query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # 由於微軟 GraphRAG 主要是 CLI 或內部 Async API，此處示範呼叫邏輯
            # 實務上你可能需要根據你存放 graphrag output 的實體路徑進行調整
            if self.mode == "global":
                # Global Search: 針對社群摘要 (Community Summaries) 進行 Map-Reduce
                response_text, context_data = await run_global_search(
                    config_dir=self.root_dir,
                    data_dir=os.path.join(self.root_dir, "output"),
                    query=user_query,
                )
            else:
                # Local Search: 針對鄰近節點與關聯進行檢索
                response_text, context_data = await run_local_search(
                    config_dir=self.root_dir,
                    data_dir=os.path.join(self.root_dir, "output"),
                    query=user_query,
                )
            
            # 處理上下文 (擷取微軟格式的 context)
            retrieved_contexts = [str(context_data)] if context_data else []
            generated_answer = str(response_text)
            
        except Exception as e:
            print(f"❌ MS GraphRAG 查詢發生錯誤: {e}")
            generated_answer = f"發生錯誤: {e}"
            retrieved_contexts = []

        end_time = time.time()
        execution_time_sec = round(end_time - start_time, 4)

        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": [], # MS GraphRAG 通常回傳 Entity/Community ID，可於此處對接
            "source_nodes": [],
            "execution_time_sec": execution_time_sec
        }