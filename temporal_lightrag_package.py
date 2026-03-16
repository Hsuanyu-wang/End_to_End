from lightrag import LightRAG
from lightrag.llm import openai_complete_if_cache
import json

class TemporalLightRAGPackage:
    def __init__(self, working_dir, llm_model="gpt-4o-mini"):
        """
        [引用: Graphiti 時序概念]
        增強 LightRAG 的節點屬性，將 Time/Date 作為檢索排序的重要權重。
        """
        self.working_dir = working_dir
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=openai_complete_if_cache,
            llm_model_name=llm_model
        )

    def _preprocess_with_time(self, text: str, timestamp: str) -> str:
        # 將時間元資料硬編碼進文本開頭，確保 LightRAG 的 LLM 能捕捉到時序實體
        return f"[RECORD_TIME: {timestamp}] {text}"

    def insert(self, records: list[dict]):
        """
        預期輸入格式: [{"text": "客戶反映螢幕閃爍", "timestamp": "2023-10-01"}]
        """
        processed_texts = [
            self._preprocess_with_time(rec["text"], rec["timestamp"]) 
            for rec in records
        ]
        self.rag.insert(processed_texts)

    def query(self, query_text: str, mode="hybrid", time_weighting=True):
        # 1. 執行標準 LightRAG 檢索
        raw_response = self.rag.query(query_text, param={"mode": mode})
        
        if not time_weighting:
            return raw_response
            
        ################################################################################################################################
        # 2. ToG / Graphiti 啟發式後處理: 
        # (這裡簡化為利用 LLM 針對檢索結果進行時序對齊與過濾)
        temporal_prompt = f"""
        Given the following response generated from customer service knowledge graph:
        {raw_response}
        
        And the user's query: "{query_text}"
        
        Please re-organize the answer strictly in chronological order. If there are conflicting resolutions, prioritize the most recent one.
        """
        ################################################################################################################################
        # 呼叫您的 LLM function 進行二次過濾
        final_response = openai_complete_if_cache(temporal_prompt, model="gpt-4o-mini")
        return final_response