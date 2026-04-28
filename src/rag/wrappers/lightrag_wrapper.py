"""
LightRAG Wrapper

封裝 LightRAG 框架的 RAG Pipeline。

DEPRECATED: 此模組中的 LightRAGWrapper、LightRAGWrapper_Original、LightRAGStrategyWrapper
仍可正常使用，但新程式碼建議透過 run_evaluation.py 的統一入口
（--graph_type lightrag --graph_retrieval native|ppr|pcst|tog）自動建立，
無需直接匯入這些 Wrapper。未來版本可能移除。
"""

import inspect
from typing import Dict, Any
from .base_wrapper import BaseRAGWrapper
from src.config.settings import get_settings, ModelSettings
from src.utils.token_counter import get_token_counter


class LightRAGWrapper(BaseRAGWrapper):
    """
    LightRAG 封裝器

    預設 use_context=True：以 aquery_data / query_data 搭配 only_need_context=True
    取得結構化檢索結果，再以專案自訂 prompt + llm.acomplete 生成答案（六種 mode 皆如此）。

    use_context=False（見 LightRAGWrapper_Original）：單次官方 aquery / query，
    only_need_context=False，不另匯出與自訂管線相同的 retrieved_contexts。
    
    支援 LightRAG 的多種檢索模式：
    - local: 關注特定實體與細節
    - global: 關注整體趨勢與總結
    - hybrid: 混合模式
    - mix: 知識圖譜 + 向量檢索
    - naive: 僅向量檢索
    - bypass: 直接查詢 LLM
    
    Attributes:
        name: Wrapper 名稱
        rag: LightRAG 實例
        mode: 檢索模式
        use_context: True 為結構化檢索＋自訂生成；False 為官方端到端單次查詢
        _initialized: 是否已初始化（用於新版 LightRAG）
    """
    
    def __init__(
        self,
        name: str,
        rag_instance,
        mode: str = "hybrid",
        use_context: bool = True,
        model_type: str = "small",
        schema_info: Dict[str, Any] = None
    ):
        """
        初始化 LightRAG Wrapper
        
        Args:
            name: Wrapper 名稱
            rag_instance: LightRAG 實例
            mode: 檢索模式
            use_context: True 為 only_need_context=True 取 context 後自訂生成；False 為官方單次 aquery
            model_type: 模型類型（用於取得 Settings）
            schema_info: Schema 資訊字典
        """
        super().__init__(name, schema_info=schema_info)
        self.rag = rag_instance
        self.mode = mode
        self.use_context = use_context
        self.model_type = model_type
        self._initialized = False
        self.token_budget: Dict[str, int] = {}

    def apply_token_budget(self, budget: Dict[str, int]) -> None:
        """套用 token budget 參數（由評估流程動態注入）。"""
        self.token_budget = budget or {}
        print(f"🎯 [{self.name}] 已套用 token budget: {self.token_budget}")

    def _native_answer_query_param(self):
        """
        官方端到端答案路徑用的 QueryParam（單次 retrieve+generate，不先取純 context）。
        """
        from lightrag.lightrag import QueryParam
        return QueryParam(mode=self.mode, only_need_context=False)
    
    def query(self, question: str) -> str:
        """
        同步查詢；僅支援 use_context=False（與 _execute_original_mode 一致）。
        use_context=True 時請改用非同步評估流程（_execute_query / aquery）。
        
        Args:
            question: 使用者問題
        
        Returns:
            LightRAG 官方生成之字串答案
        """
        if self.use_context:
            raise NotImplementedError(
                "use_context=True 時請使用非同步 _execute_query()；"
                "同步 query() 僅支援官方端到端模式（use_context=False）。"
            )
        param = self._native_answer_query_param()
        return self.rag.query(question, param=param)
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        執行 LightRAG 查詢
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果字典
        """
        my_settings = get_settings(model_type=self.model_type)
        
        # 解決 LightRAG 新版本的初始化問題
        if not self._initialized and hasattr(self.rag, "initialize_storages"):
            await self.rag.initialize_storages()
            self._initialized = True
        
        # 不使用自訂 context（原生模式）
        if not self.use_context:
            response = await self._execute_original_mode(query)
            return {
                "generated_answer": str(response) if response else "找不到答案",
                "retrieved_contexts": [],
                "retrieved_ids": [],
                "source_nodes": [],
            }
        
        # 使用自訂 context 模式（新版）
        response = await self._execute_context_mode(query, my_settings)
        return response
    
    async def _execute_original_mode(self, query: str) -> str:
        """
        官方端到端：單次 aquery / query，only_need_context=False（與 QueryParam 預設一致但明確寫出）。
        
        Args:
            query: 使用者查詢
        
        Returns:
            LightRAG 生成之答案字串
        """
        param = self._native_answer_query_param()
        if hasattr(self.rag, "aquery"):
            response = await self.rag.aquery(query, param=param)
        elif inspect.iscoroutinefunction(self.rag.query):
            response = await self.rag.query(query, param=param)
        else:
            response = self.rag.query(query, param=param)
        
        # 相容處理：如果回傳的是 LlamaIndex 物件才取 .text
        if hasattr(response, "text"):
            response = response.text
        elif not isinstance(response, str):
            response = str(response)
        
        return response
    
    async def _execute_context_mode(
        self, query: str, settings: ModelSettings
    ) -> Dict[str, Any]:
        """
        執行自訂 context 模式（使用 LightRAG context + 自訂 prompt）
        
        Args:
            query: 使用者查詢
            settings: ModelSettings（含 llm 等）
        
        Returns:
            查詢結果字典（包含詳細token統計）
        """
        from lightrag.lightrag import QueryParam
        from lightrag.prompt import PROMPTS
        from src.rag.graph.lightrag_id_mapper import ChunkIDMapper
        
        # 取得 LightRAG context（若可用，優先帶入 chunk_top_k）
        chunk_top_k = int(self.token_budget.get("chunk_top_k", 0)) if self.token_budget else 0
        param_kwargs = {"mode": self.mode, "only_need_context": True}
        if chunk_top_k > 0:
            param_kwargs["top_k"] = chunk_top_k

        if hasattr(self.rag, "aquery"):
            raw_data = await self.rag.aquery_data(
                query,
                param=QueryParam(**param_kwargs)
            )
        elif inspect.iscoroutinefunction(self.rag.query):
            raw_data = self.rag.query_data(
                query,
                param=QueryParam(**param_kwargs)
            )
        else:
            raw_data = self.rag.query_data(
                query,
                param=QueryParam(**param_kwargs)
            )
        
        # 提取 context 和 chunk IDs
        retrieved_contexts = []
        retrieved_ids = []
        
        # 用於詳細token統計
        entity_contexts = []
        relation_contexts = []
        chunk_contexts = []
        
        # 初始化 ChunkIDMapper
        storage_dir = self.rag.working_dir
        mapper = ChunkIDMapper(storage_dir)
        
        # 解析 raw_data 取得各組件
        if raw_data and isinstance(raw_data, dict):
            data_content = raw_data.get("data", {})
            
            # 提取 entities（與 LightRAGRetriever._parse_query_data 對齊：description 優先）
            if "entities" in data_content and isinstance(data_content["entities"], list):
                for entity in data_content["entities"]:
                    if isinstance(entity, dict):
                        entity_text = entity.get("description", "") or entity.get("entity_name", "")
                        if entity_text:
                            entity_contexts.append(str(entity_text))
                    elif isinstance(entity, str):
                        entity_contexts.append(entity)
            
            # 提取 relationships（與 LightRAGRetriever._parse_query_data 對齊：description 優先，fallback src->tgt）
            if "relationships" in data_content and isinstance(data_content["relationships"], list):
                for rel in data_content["relationships"]:
                    if isinstance(rel, dict):
                        desc = rel.get("description", "") or ""
                        src = rel.get("src_id", "")
                        tgt = rel.get("tgt_id", "")
                        rel_text = desc if desc else f"{src} -> {tgt}"
                        if rel_text:
                            relation_contexts.append(str(rel_text))
                    elif isinstance(rel, str):
                        relation_contexts.append(rel)
            
            # 提取 chunks
            if "chunks" in data_content and isinstance(data_content["chunks"], list):
                for chunk in data_content["chunks"]:
                    if isinstance(chunk, dict):
                        # 提取 content
                        content = chunk.get("content", "")
                        if content:
                            chunk_contexts.append(content)
                            retrieved_contexts.append(content)
                        
                        # 提取 chunk_id 或 reference_id
                        chunk_id = chunk.get("chunk_id") or chunk.get("reference_id") or chunk.get("file_path")
                        if chunk_id:
                            # 使用 mapper 轉換為原始 NO
                            original_no = mapper.get_original_no(chunk_id)
                            retrieved_ids.append(original_no if original_no else chunk_id)
            
            # 如果沒有找到 chunks，使用整個 context
            if not retrieved_contexts:
                raw_context = str(raw_data)
                retrieved_contexts = [raw_context] if raw_context else []
                chunk_contexts = retrieved_contexts
        else:
            raw_context = str(raw_data)
            retrieved_contexts = [raw_context] if raw_context else []
            chunk_contexts = retrieved_contexts
        
        # 先依 chunk_top_k 截斷（確保方法間可比較）
        if chunk_top_k > 0:
            retrieved_contexts = retrieved_contexts[:chunk_top_k]
            chunk_contexts = chunk_contexts[:chunk_top_k]
            if retrieved_ids:
                retrieved_ids = retrieved_ids[:chunk_top_k]

        # 應用 retrieval_max_tokens 限制（透過基底類別的方法）
        if self.retrieval_max_tokens > 0:
            retrieved_contexts = self._truncate_contexts_by_tokens(retrieved_contexts)
            chunk_contexts = retrieved_contexts
            if retrieved_ids:
                retrieved_ids = retrieved_ids[:len(retrieved_contexts)]
        # 或使用 token budget 機制（若有設定）
        elif self.token_budget:
            # 再依 max_total_tokens 做 context budget 截斷
            max_total_tokens = int(self.token_budget.get("max_total_tokens", 0))
            if max_total_tokens > 0 and retrieved_contexts:
                kept_contexts = []
                running_tokens = 0
                for ctx in retrieved_contexts:
                    ctx_tokens = get_token_counter().count_tokens(ctx)
                    if kept_contexts and (running_tokens + ctx_tokens) > max_total_tokens:
                        break
                    kept_contexts.append(ctx)
                    running_tokens += ctx_tokens
                retrieved_contexts = kept_contexts
                chunk_contexts = kept_contexts
                if retrieved_ids:
                    retrieved_ids = retrieved_ids[:len(kept_contexts)]

        context_str = "\n".join(retrieved_contexts) if retrieved_contexts else ""
        
        # 計算詳細token統計
        token_counter = get_token_counter()
        entity_tokens = token_counter.count_tokens_batch(entity_contexts) if entity_contexts else 0
        relation_tokens = token_counter.count_tokens_batch(relation_contexts) if relation_contexts else 0
        chunk_tokens = token_counter.count_tokens_batch(chunk_contexts) if chunk_contexts else 0
        total_tokens = token_counter.count_tokens(context_str)
        
        #############################################
        # 舊版手動組裝Generation(只有context內容)
        #############################################
        # # 組裝 Prompt
        # prompt = PROMPTS["rag_response"].format(
        #     context_data=context_str,
        #     response_type="Multiple Paragraphs",
        #     user_prompt=query,
        #     query=query,
        #     retrieved_contexts=context_str
        # )
        
        # # 使用 LLM 生成答案（不限制生成 token）
        # llm_response = await settings.llm.acomplete(prompt)
        # response = llm_response.text
        #############################################
        # 新版官方端到端Generation
        #############################################
        param_generate = QueryParam(mode=self.mode, only_need_context=False)
        
        if hasattr(self.rag, "aquery"):
            raw_response = await self.rag.aquery(query, param=param_generate)
        elif inspect.iscoroutinefunction(self.rag.query):
            raw_response = await self.rag.query(query, param=param_generate)
        else:
            raw_response = self.rag.query(query, param=param_generate)
        
        # 相容處理：如果回傳的是 LlamaIndex 物件才取 .text
        if hasattr(raw_response, "text"):
            response = raw_response.text
        elif not isinstance(raw_response, str):
            response = str(raw_response)
        else:
            response = raw_response
        
        print(f"📊 Retrieved {len(retrieved_ids)} chunk IDs | Tokens: {total_tokens} (E:{entity_tokens}, R:{relation_tokens}, C:{chunk_tokens})")
        
        return {
            "generated_answer": str(response) if response else "找不到答案",
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": retrieved_ids,
            "retrieved_entities": entity_contexts,
            "retrieved_relations": relation_contexts,
            "source_nodes": [],
            # Token統計
            "context_tokens": total_tokens,
            "context_token_details": {
                "total_tokens": total_tokens,
                "entity_tokens": entity_tokens,
                "relation_tokens": relation_tokens,
                "chunk_tokens": chunk_tokens,
                "num_entities": len(entity_contexts),
                "num_relations": len(relation_contexts),
                "num_chunks": len(chunk_contexts)
            }
        }


# 向後兼容別名
class LightRAGWrapper_Original(LightRAGWrapper):
    """
    LightRAG 官方端到端基線：單次 aquery / query（only_need_context=False）。
    不另匯出結構化 retrieved_contexts，依賴該欄位之檢索指標可能為空或不適用。
    """
    
    def __init__(self, name: str, rag_instance, mode: str = "hybrid", schema_info: Dict[str, Any] = None):
        super().__init__(
            name=name,
            rag_instance=rag_instance,
            mode=mode,
            use_context=False,
            schema_info=schema_info
        )


class LightRAGStrategyWrapper(LightRAGWrapper):
    """
    使用 LightRAGGraphRetriever（Entity Linking + Graph Traversal Strategy）的 Wrapper。

    保留與 LightRAGWrapper 相同的 prompt 格式、token budget、token 統計，
    但以 LightRAGGraphRetriever.aretrieve() 取代 aquery_data() 作為檢索來源。
    """

    def __init__(
        self,
        name: str,
        rag_instance,
        retriever,
        mode: str = "hybrid",
        model_type: str = "small",
        schema_info: Dict[str, Any] = None,
    ):
        super().__init__(
            name=name,
            rag_instance=rag_instance,
            mode=mode,
            use_context=True,
            model_type=model_type,
            schema_info=schema_info,
        )
        self._retriever = retriever

    async def _execute_context_mode(
        self, query: str, settings: "ModelSettings"
    ) -> Dict[str, Any]:
        """
        以 LightRAGGraphRetriever 取得 contexts，再走與 LightRAGWrapper 相同的
        prompt 組裝 + LLM 生成流程。
        """
        from lightrag.prompt import PROMPTS

        # --- 檢索 ---
        default_top_k = max(20, int(getattr(self._retriever, "el_top_k", 10)) * 10)
        top_k = (
            int(self.token_budget.get("chunk_top_k", default_top_k))
            if self.token_budget
            else default_top_k
        )
        retrieval_result = await self._retriever.aretrieve(
            query=query,
            top_k=top_k,
        )

        retrieved_contexts = retrieval_result.get("contexts", [])
        metadata = retrieval_result.get("metadata", {})

        # strategy-based path：由 LightRAGGraphRetriever 補上 chunk->REF 映射
        retrieved_ids: list = retrieval_result.get("retrieved_ids", []) or []
        # 去重且保留順序（避免 nodes/edges 擷取造成重複）
        seen_ids = set()
        deduped_ids: list = []
        for rid in retrieved_ids:
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            deduped_ids.append(rid)
        retrieved_ids = deduped_ids

        entity_contexts = [
            c for c in retrieved_contexts
            if not c.startswith(("The relation", "->"))
        ]
        relation_contexts = [
            c for c in retrieved_contexts
            if c not in entity_contexts
        ]
        chunk_contexts: list = []

        # --- token budget / retrieval_max_tokens 截斷 ---
        chunk_top_k = int(self.token_budget.get("chunk_top_k", 0)) if self.token_budget else 0
        if chunk_top_k > 0:
            retrieved_contexts = retrieved_contexts[:chunk_top_k]
            retrieved_ids = retrieved_ids[:chunk_top_k]

        if self.retrieval_max_tokens > 0:
            retrieved_contexts = self._truncate_contexts_by_tokens(retrieved_contexts)
        elif self.token_budget:
            max_total_tokens = int(self.token_budget.get("max_total_tokens", 0))
            if max_total_tokens > 0 and retrieved_contexts:
                token_counter = get_token_counter()
                kept, running = [], 0
                for ctx in retrieved_contexts:
                    t = token_counter.count_tokens(ctx)
                    if kept and (running + t) > max_total_tokens:
                        break
                    kept.append(ctx)
                    running += t
                retrieved_contexts = kept

        context_str = "\n".join(retrieved_contexts) if retrieved_contexts else ""

        # --- token 統計 ---
        token_counter = get_token_counter()
        entity_tokens = token_counter.count_tokens_batch(entity_contexts) if entity_contexts else 0
        relation_tokens = token_counter.count_tokens_batch(relation_contexts) if relation_contexts else 0
        chunk_tokens = token_counter.count_tokens_batch(chunk_contexts) if chunk_contexts else 0
        total_tokens = token_counter.count_tokens(context_str)

        # --- Prompt + LLM ---
        prompt = PROMPTS["rag_response"].format(
            context_data=context_str,
            response_type="Multiple Paragraphs",
            user_prompt=query,
            query=query,
            retrieved_contexts=context_str,
        )
        llm_response = await settings.llm.acomplete(prompt)
        response = llm_response.text

        strategy_name = metadata.get("strategy", "unknown")
        print(
            f"[{self.name}] strategy={strategy_name} | "
            f"contexts={len(retrieved_contexts)} | tokens={total_tokens}"
        )

        return {
            "generated_answer": str(response) if response else "找不到答案",
            "retrieved_contexts": retrieved_contexts,
            "retrieved_ids": retrieved_ids,
            "retrieved_entities": entity_contexts,
            "retrieved_relations": relation_contexts,
            "source_nodes": [],
            "context_tokens": total_tokens,
            "context_token_details": {
                "total_tokens": total_tokens,
                "entity_tokens": entity_tokens,
                "relation_tokens": relation_tokens,
                "chunk_tokens": chunk_tokens,
                "num_entities": len(entity_contexts),
                "num_relations": len(relation_contexts),
                "num_chunks": len(chunk_contexts),
            },
            "traversal_metadata": metadata,
        }
