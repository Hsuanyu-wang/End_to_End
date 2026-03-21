from lightrag import LightRAG
from lightrag.lightrag import QueryParam
import json
import asyncio


class TemporalLightRAGPackage:
    def __init__(self, working_dir, settings):
        """
        [引用: Graphiti 時序概念]
        增強 LightRAG 的節點屬性，將 Time/Date 作為檢索排序的重要權重。
        """
        self.settings = settings
        
        async def custom_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = await settings.builder_llm.acomplete(full_prompt)
            return response.text
        
        self.working_dir = working_dir
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=custom_llm_func,
            llm_model_name=getattr(self.settings, 'llm_model', 'unknown')
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
        raw_response = self.rag.query(query_text, param=QueryParam(mode=mode))
        
        if not time_weighting:
            return raw_response
        
        # 利用 LLM 針對檢索結果進行時序對齊與過濾
        temporal_prompt = f"""
        Given the following response generated from customer service knowledge graph:
        {raw_response}
        
        And the user's query: "{query_text}"
        
        Please re-organize the answer strictly in chronological order. If there are conflicting resolutions, prioritize the most recent one.
        """
        llm = getattr(self.settings, 'llm', None) or getattr(self.settings, 'builder_llm', None)
        if llm is None:
            return raw_response
        
        try:
            final_response = llm.complete(temporal_prompt)
            return final_response.text if hasattr(final_response, 'text') else str(final_response)
        except Exception as e:
            print(f"⚠️  時序後處理失敗，回傳原始結果: {e}")
            return raw_response