"""
LightRAG 建圖後之實體描述整合（Post-hoc Entity Description Consolidation, EDC）

- baseline 索引不修改；自 source storage 複製整個 working_dir 至含 "edc" 的 custom_tag 目錄後執行整合。
- 對描述長度超過閾值（含多段碎片）的 entity，用 LLM 整合成單一連貫描述。
- 使用 LightRAG graph storage 的 upsert_node API 寫回 description 欄位。
- 支援 source_custom_tag 參數，可串接在其他 plugin（如 PETN）之後。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.storage import get_storage_path

_COPY_EXCLUDE_PATTERNS = ("kv_store_llm_response_cache.json",)
MARKER_FILENAME = ".lightrag_edc_meta.json"
LOG_FILENAME = "lightrag_edc_log.json"


# ─── tag ──────────────────────────────────────────────────────────────────────

def build_edc_custom_tag() -> str:
    """EDC plugin 的 custom_tag；固定為 'edc'。"""
    return "edc"


# ─── 輔助：marker / 持久化 ────────────────────────────────────────────────────

def _lightrag_dir_populated(dir_path: str) -> bool:
    if not os.path.isdir(dir_path):
        return False
    try:
        return len(os.listdir(dir_path)) > 0
    except OSError:
        return False


def _marker_path(plugin_dir: str) -> str:
    return os.path.join(plugin_dir, MARKER_FILENAME)


def _read_marker(plugin_dir: str) -> Optional[Dict[str, Any]]:
    p = _marker_path(plugin_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_marker(
    plugin_dir: str,
    *,
    source_path: str,
    min_desc_len: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "source_path": source_path,
        "min_desc_len": min_desc_len,
        "updated_at": int(time.time()),
    }
    if extra:
        payload.update(extra)
    with open(_marker_path(plugin_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _marker_matches(marker: Dict[str, Any], min_desc_len: int) -> bool:
    return marker.get("min_desc_len") == min_desc_len


# ─── 核心：LLM 描述整合 ──────────────────────────────────────────────────────────

@dataclass
class EDCResult:
    source_path: str
    plugin_path: str
    num_entities: int
    num_consolidated: int
    skipped_reason: str = ""


def _consolidate_description(entity_name: str, description: str, llm) -> Optional[str]:
    """
    呼叫 LLM 整合單一 entity 的碎片描述。失敗時回傳 None。
    """
    from llama_index.core import PromptTemplate
    from pydantic import BaseModel, Field

    class ConsolidatedDescription(BaseModel):
        consolidated: str = Field(description="整合後的實體描述（最多 300 字）")

    prompt_str = """
    你是一個知識庫策展人。
    實體名稱："{name}"

    以下是從不同文件片段收集到的描述，可能有重複或互相補充的資訊：
    ---
    {description}
    ---

    請整合上述描述成一份連貫、無重複、資訊完整的描述（最多 300 字）。
    只保留與此實體直接相關的重要事實。
    只回傳整合後的描述文字，不要加標題或額外說明。
    """

    prompt_template = PromptTemplate(template=prompt_str)
    try:
        response_obj = llm.structured_predict(
            ConsolidatedDescription,
            prompt=prompt_template,
            name=entity_name,
            description=description[:2000],  # 避免超出 context window
        )
        result = (response_obj.consolidated or "").strip()
        return result if result else None
    except Exception as e:
        print(f">> [EDC] 描述整合失敗（entity={entity_name}）：{e}")
        return None


def run_entity_desc_consolidate(
    rag,
    llm,
    min_desc_len: int = 200,
    max_chars_per_desc: int = 2000,
) -> EDCResult:
    """
    對已載入之 LightRAG 實例執行 entity description 整合。

    Args:
        rag:              LightRAG 實例
        llm:              LLM 實例
        min_desc_len:     只處理描述長度超過此值的 entity（過濾短描述的孤立 entity）
        max_chars_per_desc: 傳給 LLM 的描述最大字元數

    Returns:
        EDCResult（統計摘要）
    """
    loop = asyncio.get_event_loop()
    graph_inst = rag.chunk_entity_relation_graph
    nodes: List[Dict[str, Any]] = loop.run_until_complete(graph_inst.get_all_nodes())

    if not nodes:
        return EDCResult(source_path="", plugin_path="", num_entities=0,
                         num_consolidated=0, skipped_reason="no nodes")

    # 篩選需要整合的節點（描述夠長，可能包含多段碎片）
    candidates = [
        n for n in nodes
        if len(str(n.get("description", ""))) >= min_desc_len
    ]
    print(
        f">> [EDC] 共 {len(nodes)} 個實體節點，"
        f"其中 {len(candidates)} 個描述長度 >= {min_desc_len}，將進行整合..."
    )

    num_consolidated = 0
    for i, node in enumerate(candidates):
        node_id = str(node.get("id", ""))
        original_desc = str(node.get("description", ""))
        consolidated = _consolidate_description(
            node_id, original_desc[:max_chars_per_desc], llm
        )
        if consolidated and consolidated != original_desc:
            loop.run_until_complete(
                graph_inst.upsert_node(node_id, {"description": consolidated})
            )
            num_consolidated += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(candidates):
            print(f">> [EDC] 已處理 {i + 1}/{len(candidates)} 個候選節點")

    # 持久化到磁碟
    if hasattr(graph_inst, "index_done_callback"):
        loop.run_until_complete(graph_inst.index_done_callback())

    print(f">> [EDC] 完成，共整合 {num_consolidated}/{len(candidates)} 個節點的描述")
    return EDCResult(
        source_path="",
        plugin_path="",
        num_entities=len(nodes),
        num_consolidated=num_consolidated,
    )


# ─── ensure（複製 + 執行 + marker）─────────────────────────────────────────────

def ensure_edc_lightrag_index(
    Settings: Any,
    *,
    data_type: str,
    method: str,
    mode: str,
    fast_test: bool,
    llm,
    source_custom_tag: str = "",
    min_desc_len: int = 200,
    force_recopy: bool = False,
) -> Tuple[str, str]:
    """
    確保 EDC plugin storage 存在且已完成。

    Args:
        source_custom_tag: 來源 storage 的 custom_tag（""=baseline，或前一個 plugin 的 tag）
        min_desc_len:      只整合描述長度超過此值的 entity

    Returns:
        (source_path, plugin_path)
    """
    from src.rag.graph.lightrag import _create_lightrag_at_path

    source_path = get_storage_path(
        storage_type="lightrag",
        data_type=data_type,
        method=method,
        mode=mode,
        fast_test=fast_test,
        custom_tag=source_custom_tag,
    )
    edc_tag = build_edc_custom_tag()
    full_tag = f"{source_custom_tag}_{edc_tag}" if source_custom_tag else edc_tag
    plugin_path = get_storage_path(
        storage_type="lightrag",
        data_type=data_type,
        method=method,
        mode=mode,
        fast_test=fast_test,
        custom_tag=full_tag,
    )

    if not _lightrag_dir_populated(source_path):
        raise FileNotFoundError(
            f"[EDC] 來源 LightRAG 索引不存在或為空: {source_path}，請先建圖。"
        )

    # 快取命中檢查
    marker = _read_marker(plugin_path)
    if (
        marker
        and _marker_matches(marker, min_desc_len)
        and _lightrag_dir_populated(plugin_path)
        and not force_recopy
    ):
        print(f"✅ [EDC] 已存在符合參數之索引，略過 → {plugin_path}")
        return source_path, plugin_path

    if force_recopy and os.path.isdir(plugin_path):
        print(f"🗑️ [EDC] force_recopy，移除 {plugin_path}")
        shutil.rmtree(plugin_path, ignore_errors=True)

    # marker 不符或目錄損壞：重建
    if _lightrag_dir_populated(plugin_path):
        m = _read_marker(plugin_path)
        if not (m and _marker_matches(m, min_desc_len)):
            print("⚠️ [EDC] plugin 目錄與 marker 不符，刪除後重建")
            shutil.rmtree(plugin_path, ignore_errors=True)

    if not _lightrag_dir_populated(plugin_path):
        if os.path.isdir(plugin_path):
            shutil.rmtree(plugin_path, ignore_errors=True)
        print(f"📋 [EDC] 複製 source → plugin\n   {source_path}\n   → {plugin_path}")
        shutil.copytree(
            source_path,
            plugin_path,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns(*_COPY_EXCLUDE_PATTERNS),
        )

    rag = _create_lightrag_at_path(Settings, plugin_path)
    print(f"📝 [EDC] 開始實體描述整合（min_desc_len={min_desc_len}）...")
    result = run_entity_desc_consolidate(rag, llm, min_desc_len=min_desc_len)
    result.source_path = source_path
    result.plugin_path = plugin_path

    _write_marker(
        plugin_path,
        source_path=source_path,
        min_desc_len=min_desc_len,
        extra={
            "num_entities": result.num_entities,
            "num_consolidated": result.num_consolidated,
        },
    )
    print(f"✅ [EDC] 完成 → {plugin_path}")
    return source_path, plugin_path
