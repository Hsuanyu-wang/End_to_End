"""
LightRAG 建圖後之實體類型標準化（Post-hoc Entity Type Normalization, PETN）

- baseline 索引不修改；自 source storage 複製整個 working_dir 至含 "petn" 的 custom_tag 目錄後執行重新標注。
- 使用 LightRAG graph storage 的 upsert_node API 更新 entity_type 欄位。
- 支援 source_custom_tag 參數，可串接在其他 plugin（如 EDC）之後。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.storage import get_storage_path

_COPY_EXCLUDE_PATTERNS = ("kv_store_llm_response_cache.json",)
MARKER_FILENAME = ".lightrag_petn_meta.json"
LOG_FILENAME = "lightrag_petn_log.json"


# ─── tag ──────────────────────────────────────────────────────────────────────

def build_petn_custom_tag() -> str:
    """PETN plugin 的 custom_tag；固定為 'petn'。"""
    return "petn"


# ─── 輔助：持久化 ────────────────────────────────────────────────────────────

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
    schema_entities: List[str],
    schema_method: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "source_path": source_path,
        "schema_entities": schema_entities,
        "schema_method": schema_method,
        "updated_at": int(time.time()),
    }
    if extra:
        payload.update(extra)
    with open(_marker_path(plugin_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _marker_matches(
    marker: Dict[str, Any],
    schema_entities: List[str],
    schema_method: str,
) -> bool:
    if marker.get("schema_method") != schema_method:
        return False
    return sorted(marker.get("schema_entities", [])) == sorted(schema_entities)


# ─── 核心：批次 LLM 分類 ───────────────────────────────────────────────────────

@dataclass
class PETNResult:
    source_path: str
    plugin_path: str
    num_entities: int
    num_relabeled: int
    skipped_reason: str = ""


def _batch_classify_entities(
    names_descs: List[Tuple[str, str]],
    schema_entities: List[str],
    llm,
) -> Dict[str, str]:
    """
    對一批 (name, description) 呼叫 LLM，回傳 {name: entity_type} 的映射。
    失敗時回傳空 dict。
    """
    from llama_index.core import PromptTemplate
    from pydantic import BaseModel, Field

    types_str = ", ".join(schema_entities)

    lines = []
    for idx, (name, desc) in enumerate(names_descs, 1):
        desc_truncated = desc[:200].replace("\n", " ")
        lines.append(f'{idx}. 名稱: "{name}"，描述: "{desc_truncated}"')
    entities_str = "\n".join(lines)

    class EntityTypeMapping(BaseModel):
        mappings: Dict[str, str] = Field(
            description="實體名稱到類型的映射，格式 {entity_name: entity_type}"
        )

    prompt_str = """
    你是一個知識圖譜架構師。
    以下是可用的實體類型清單：{types}

    請將下列實體各自歸類到最適合的類型。
    若無任何類型符合，請選 Concept。
    以 JSON 格式回傳，格式：{{"entity_name": "entity_type", ...}}（key 為實體名稱，value 為類型）

    【實體列表】
    {entities}
    """

    prompt_template = PromptTemplate(template=prompt_str)
    try:
        response_obj = llm.structured_predict(
            EntityTypeMapping,
            prompt=prompt_template,
            types=types_str,
            entities=entities_str,
        )
        return response_obj.mappings or {}
    except Exception as e:
        print(f">> [PETN] LLM 批次分類失敗（{e}），本批次跳過")
        return {}


def run_entity_type_normalize(
    rag,
    schema_entities: List[str],
    llm,
    batch_size: int = 20,
) -> PETNResult:
    """
    對已載入之 LightRAG 實例執行 entity type 重新標注。

    Args:
        rag:             LightRAG 實例
        schema_entities: 目標 entity type 清單（來自 evolved schema）
        llm:             LLM 實例（使用 structured_predict）
        batch_size:      每批次呼叫 LLM 的實體數

    Returns:
        PETNResult（統計摘要）
    """
    loop = asyncio.get_event_loop()

    graph_inst = rag.chunk_entity_relation_graph
    nodes: List[Dict[str, Any]] = loop.run_until_complete(graph_inst.get_all_nodes())

    if not nodes:
        return PETNResult(source_path="", plugin_path="", num_entities=0,
                          num_relabeled=0, skipped_reason="no nodes")

    valid_type_set = {s.lower() for s in schema_entities}
    num_relabeled = 0

    print(f">> [PETN] 共 {len(nodes)} 個實體節點，分批進行類型標注（batch_size={batch_size}）...")

    for batch_start in range(0, len(nodes), batch_size):
        batch = nodes[batch_start: batch_start + batch_size]
        names_descs = [
            (str(n.get("id", "")), str(n.get("description", ""))[:300])
            for n in batch
        ]

        mappings = _batch_classify_entities(names_descs, schema_entities, llm)
        if not mappings:
            continue

        for node in batch:
            node_id = str(node.get("id", ""))
            new_type = mappings.get(node_id)
            if not new_type:
                continue
            # 若 LLM 回傳的 type 不在清單中，強制設為 Concept
            if new_type.lower() not in valid_type_set:
                new_type = "Concept"
            old_type = node.get("entity_type", "")
            if new_type != old_type:
                loop.run_until_complete(
                    graph_inst.upsert_node(node_id, {"entity_type": new_type})
                )
                num_relabeled += 1

        print(f">> [PETN] 已處理 {min(batch_start + batch_size, len(nodes))}/{len(nodes)} 個節點")

    # 持久化到磁碟
    if hasattr(graph_inst, "index_done_callback"):
        loop.run_until_complete(graph_inst.index_done_callback())

    print(f">> [PETN] 完成，共重新標注 {num_relabeled}/{len(nodes)} 個節點")
    return PETNResult(
        source_path="",
        plugin_path="",
        num_entities=len(nodes),
        num_relabeled=num_relabeled,
    )


# ─── ensure（複製 + 執行 + marker）─────────────────────────────────────────────

def ensure_petn_lightrag_index(
    Settings: Any,
    *,
    data_type: str,
    method: str,
    mode: str,
    fast_test: bool,
    schema_entities: List[str],
    llm,
    schema_method: str = "anchored_additive_evolution",
    source_custom_tag: str = "",
    force_recopy: bool = False,
) -> Tuple[str, str]:
    """
    確保 PETN plugin storage 存在且已完成。

    Args:
        source_custom_tag: 來源 storage 的 custom_tag（""=baseline，或前一個 plugin 的 tag）

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
    petn_tag = build_petn_custom_tag()
    full_tag = f"{source_custom_tag}_{petn_tag}" if source_custom_tag else petn_tag
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
            f"[PETN] 來源 LightRAG 索引不存在或為空: {source_path}，請先建圖。"
        )

    # 快取命中檢查
    marker = _read_marker(plugin_path)
    if (
        marker
        and _marker_matches(marker, schema_entities, schema_method)
        and _lightrag_dir_populated(plugin_path)
        and not force_recopy
    ):
        print(f"✅ [PETN] 已存在符合參數之索引，略過 → {plugin_path}")
        return source_path, plugin_path

    if force_recopy and os.path.isdir(plugin_path):
        print(f"🗑️ [PETN] force_recopy，移除 {plugin_path}")
        shutil.rmtree(plugin_path, ignore_errors=True)

    # marker 不符或目錄損壞：重建
    if _lightrag_dir_populated(plugin_path):
        m = _read_marker(plugin_path)
        if not (m and _marker_matches(m, schema_entities, schema_method)):
            print("⚠️ [PETN] plugin 目錄與 marker 不符，刪除後重建")
            shutil.rmtree(plugin_path, ignore_errors=True)

    if not _lightrag_dir_populated(plugin_path):
        if os.path.isdir(plugin_path):
            shutil.rmtree(plugin_path, ignore_errors=True)
        print(f"📋 [PETN] 複製 source → plugin\n   {source_path}\n   → {plugin_path}")
        shutil.copytree(
            source_path,
            plugin_path,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns(*_COPY_EXCLUDE_PATTERNS),
        )

    rag = _create_lightrag_at_path(Settings, plugin_path)
    print(f"🏷️ [PETN] 開始實體類型標注，schema types: {schema_entities}")
    result = run_entity_type_normalize(rag, schema_entities, llm)
    result.source_path = source_path
    result.plugin_path = plugin_path

    _write_marker(
        plugin_path,
        source_path=source_path,
        schema_entities=schema_entities,
        schema_method=schema_method,
        extra={
            "num_entities": result.num_entities,
            "num_relabeled": result.num_relabeled,
        },
    )
    print(f"✅ [PETN] 完成 → {plugin_path}")
    return source_path, plugin_path
