import os
import json
import logging
import numpy as np
import nest_asyncio
import asyncio
import requests
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from src.config.settings import get_settings
from src.data.processors import data_processing
from src.storage import get_storage_path

nest_asyncio.apply()

_logger = logging.getLogger(__name__)

# 嘗試匯入 LightRAG 內建的 LlamaIndex 適配器 (建議更新至最新版 LightRAG 以支援)
try:
    from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
    HAS_LLAMA_INDEX_IMPL = True
except ImportError:
    HAS_LLAMA_INDEX_IMPL = False


def _query_embed_context_length(
    ollama_url: str, model_name: str, fallback_chars: int = 2560
) -> int:
    """
    向 Ollama API 查詢 embedding model 的實際 context length，
    並以保守比率換算為安全字元上限。
    若查詢失敗則回退到 fallback_chars。
    """
    try:
        resp = requests.post(
            f"{ollama_url}/api/show",
            json={"name": model_name},
            timeout=10,
        )
        resp.raise_for_status()
        info = resp.json()

        # Ollama 回傳格式：model_info 或 parameters 中含 num_ctx
        num_ctx = None
        model_info = info.get("model_info", {})
        for key, val in model_info.items():
            if "context_length" in key:
                num_ctx = int(val)
                break
        if num_ctx is None:
            params = info.get("parameters", "")
            if isinstance(params, str):
                for line in params.splitlines():
                    if "num_ctx" in line:
                        num_ctx = int(line.split()[-1])
                        break

        if num_ctx and num_ctx > 0:
            # 中文平均 ~1.5 tokens/char；取 90% 安全邊際
            safe_chars = int(num_ctx / 1.5 * 0.9)
            _logger.info(
                "Ollama embedding model %s context_length=%d → 安全字元上限=%d",
                model_name, num_ctx, safe_chars,
            )
            print(
                f"🔍 Embedding model '{model_name}' context_length={num_ctx} "
                f"→ 動態安全字元上限={safe_chars}"
            )
            return safe_chars

        _logger.warning(
            "無法從 Ollama API 解析 context_length，使用 fallback=%d", fallback_chars
        )
    except Exception as exc:
        _logger.warning(
            "查詢 Ollama embedding context length 失敗 (%s)，使用 fallback=%d",
            exc, fallback_chars,
        )

    return fallback_chars


def _build_llm_and_embed(Settings):
    """建立 LightRAG 所需的 LLM / Embedding 函式（共用邏輯）。"""
    print("正在測量 Embedding 實際維度...")
    dummy_emb = Settings.embed_model.get_text_embedding("test_dimension")
    EMBEDDING_DIM = len(dummy_emb)
    print(f"檢測到 Embedding 維度為: {EMBEDDING_DIM}")

    async def custom_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = await Settings.builder_llm.acomplete(full_prompt)
        return response.text

    config_max = int(getattr(Settings.lightrag_config, "embed_max_input_chars", 2560))
    ollama_url = getattr(Settings.embed_model, "base_url", "") or ""
    embed_model_name = getattr(Settings.embed_model, "model_name", "") or ""

    if ollama_url and embed_model_name:
        max_chars = _query_embed_context_length(ollama_url, embed_model_name, config_max)
    else:
        max_chars = config_max

    # 同步寫回 config 供 entity merge 等下游元件使用
    if hasattr(Settings, "lightrag_config"):
        Settings.lightrag_config.embed_max_input_chars = max_chars

    async def manual_embed_func(texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        truncated = []
        for t in texts:
            if len(t) > max_chars:
                _logger.debug(
                    "嵌入輸入截斷: 原長度 %d → %d 字元",
                    len(t),
                    max_chars,
                )
                truncated.append(t[:max_chars])
            else:
                truncated.append(t)
        embeddings = await Settings.embed_model.aget_text_embedding_batch(truncated)
        return np.array(embeddings)

    custom_embed_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=max_chars,
        func=manual_embed_func
    )
    return custom_llm_func, custom_embed_func


def _create_lightrag_at_path(Settings, working_dir: str):
    """
    在指定的 working_dir 建立 LightRAG 實例並初始化 storages。
    供 Retriever 直接以 storage_path 載入已存在的索引。
    """
    custom_llm_func, custom_embed_func = _build_llm_and_embed(Settings)

    os.makedirs(working_dir, exist_ok=True)
    print(f"📂 LightRAG 儲存路徑: {working_dir}")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=custom_llm_func,
        embedding_func=custom_embed_func,
        addon_params={
            "entity_types": Settings.lightrag_config.entity_types
        }
    )

    loop = asyncio.get_event_loop()
    if hasattr(rag, "initialize_storages"):
        loop.run_until_complete(rag.initialize_storages())

    return rag


def get_lightrag_engine(
    Settings,
    data_type: str = "DI",
    sup: str = "",
    fast_test: bool = False,
    mode: str = "",
    custom_tag: str = "",
):
    """
    建立 LightRAG 實例（透過 StorageManager 推算路徑）
    
    Args:
        Settings: 設定物件
        data_type: 資料類型（DI 或 GEN）
        sup: 自訂標籤（用於區分不同實驗）
        fast_test: 是否為快速測試模式
        mode: 檢索模式（local/global/hybrid 等，用於區分不同 mode 的 storage）
        custom_tag: StorageManager 自訂標籤（例如相似實體合併實驗目錄）
    
    Returns:
        LightRAG 實例
    """
    working_dir = get_storage_path(
        storage_type="lightrag",
        data_type=data_type,
        method=sup if sup else "default",
        mode=mode,
        fast_test=fast_test,
        custom_tag=custom_tag,
    )
    return _create_lightrag_at_path(Settings, working_dir)


def build_lightrag_index(
    Settings,
    mode: str = "natural_text",
    data_type: str = "DI",
    sup: str = "",
    fast_build: bool = False,
    lightrag_mode: str = "",
    custom_tag: str = "",
) -> None:
    """
    執行資料插入與圖譜建立
    
    Args:
        Settings: 設定物件
        mode: 資料處理模式（natural_text, markdown 等）
        data_type: 資料類型（DI, GEN）
        sup: 自訂標籤
        fast_build: 是否快速建立
        lightrag_mode: LightRAG 檢索模式（用於區分 storage）
        custom_tag: 寫入之索引目錄 custom_tag（預設 baseline 為空字串）
    """
    rag = get_lightrag_engine(
        Settings,
        data_type=data_type,
        sup=sup,
        fast_test=fast_build,
        mode=lightrag_mode,
        custom_tag=custom_tag,
    )
    print("正在處理文本並匯入 LightRAG (此過程包含 LLM 實體與關係抽取，可能較久)...")
    docs = data_processing(mode=mode, data_type=data_type)
    if not docs:
        raise ValueError(f"Failed to load documents. Check if data_mode '{mode}' and data_type '{data_type}' are correct.")
    if docs is not None:
        if fast_build:
            print("⚡ [LightRAG] 啟用微型建圖模式：僅抽取前 2 筆文本進行快速建圖...")
            docs = docs[:2]
        texts = [doc.text for doc in docs] if hasattr(docs[0], "text") else docs
        
        # 插入文本到 LightRAG
        rag.insert(texts)
        
        # 建立 chunk_id 到 doc_id 的映射
        print("📝 建立 chunk ID 映射...")
        from src.rag.graph.lightrag_id_mapper import ChunkIDMapper
        mapper = ChunkIDMapper(rag.working_dir)
        mapper.build_mapping_from_documents(docs, texts)
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