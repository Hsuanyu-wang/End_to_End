"""
LightRAG SER（相似實體合併）串接版本。

與 lightrag_similar_entity_merge.py 的差異：
  - 接受 source_custom_tag 參數，可從任何前置 plugin（PETN / EDC）的 storage 複製，
    而非硬連結到 custom_tag="" 的 baseline。
  - 不修改原始 ensure_similar_merged_lightrag_index，完全獨立的新函式。
  - 內部仍呼叫 run_similar_entity_merge（現有函式，不修改）。

Tag 格式範例：
  source_custom_tag="petn"   → plugin custom_tag = "petn_simmerge_t0_90_name_desc"
  source_custom_tag="edc"    → plugin custom_tag = "edc_simmerge_t0_90_name_desc"
  source_custom_tag=""       → plugin custom_tag = "simmerge_t0_90_name_desc"（與原版相同）
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Optional, Tuple

from src.storage import get_storage_path
from src.rag.plugins.lightrag_similar_entity_merge import (
    build_simmerge_custom_tag,
    run_similar_entity_merge,
    LOG_FILENAME,
    STEPS_FILENAME,
    ERROR_FILENAME,
    _lightrag_dir_populated,
    _read_marker,
    _write_marker,
    _marker_matches,
)

_COPY_EXCLUDE_PATTERNS = ("kv_store_llm_response_cache.json",)


def ensure_simmerge_from_source_lightrag_index(
    Settings: Any,
    *,
    data_type: str,
    method: str,
    mode: str,
    fast_test: bool,
    threshold: float,
    threshold_max: Optional[float] = None,
    text_mode: str,
    force_recopy: bool,
    dry_run: bool,
    source_custom_tag: str = "",
    use_llm_verify: bool = False,
) -> Tuple[str, str]:
    """
    確保 SER plugin storage 存在且已完成，支援從任意 source_custom_tag 複製。

    source_custom_tag 決定複製來源：
      ""       → 從 baseline（custom_tag=""）複製，行為與原版相同
      "petn"   → 從 PETN 結果複製
      "edc"    → 從 EDC 結果複製

    Tag 格式：f"{source_custom_tag}_{simmerge_tag}" or simmerge_tag（source 為空時）

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

    simmerge_tag = build_simmerge_custom_tag(threshold, text_mode, threshold_max)
    full_tag = f"{source_custom_tag}_{simmerge_tag}" if source_custom_tag else simmerge_tag

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
            f"[SimMerge-Chained] 來源 LightRAG 索引不存在或為空: {source_path}，請先建圖。"
        )

    # 快取命中（沿用 _read_marker / _marker_matches，marker 格式與原版一致）
    if not dry_run:
        marker = _read_marker(plugin_path)
        if (
            marker
            and _marker_matches(marker, threshold, text_mode, dry_run, threshold_max)
            and _lightrag_dir_populated(plugin_path)
            and not force_recopy
        ):
            print(
                f"✅ [SimMerge-Chained] 已存在符合參數之索引，略過 → {plugin_path}"
            )
            return source_path, plugin_path

    if force_recopy and os.path.isdir(plugin_path):
        print(f"🗑️ [SimMerge-Chained] force_recopy，移除 {plugin_path}")
        shutil.rmtree(plugin_path, ignore_errors=True)

    # marker 不符或缺失：整包重建
    if not dry_run and _lightrag_dir_populated(plugin_path):
        m = _read_marker(plugin_path)
        if not (
            m and _marker_matches(m, threshold, text_mode, dry_run, threshold_max)
        ):
            print(
                "⚠️ [SimMerge-Chained] plugin 目錄與 marker 不符或缺失，刪除後自 source 複製"
            )
            shutil.rmtree(plugin_path, ignore_errors=True)

    if not _lightrag_dir_populated(plugin_path):
        if os.path.isdir(plugin_path):
            shutil.rmtree(plugin_path, ignore_errors=True)
        print(
            f"📋 [SimMerge-Chained] 複製 source → plugin（來源: custom_tag='{source_custom_tag}'）\n"
            f"   {source_path}\n   → {plugin_path}"
        )
        shutil.copytree(
            source_path,
            plugin_path,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns(*_COPY_EXCLUDE_PATTERNS),
        )

    log_path = os.path.join(plugin_path, LOG_FILENAME)
    rag = _create_lightrag_at_path(Settings, plugin_path)
    print(
        f"🔀 [SimMerge-Chained] threshold={threshold} threshold_max={threshold_max!r} "
        f"text_mode={text_mode} dry_run={dry_run}"
    )
    run_similar_entity_merge(
        rag,
        Settings,
        threshold=threshold,
        threshold_max=threshold_max,
        text_mode=text_mode,
        dry_run=dry_run,
        log_file=log_path,
        use_llm_verify=use_llm_verify,
        baseline_path=source_path,
    )
    if not dry_run:
        _write_marker(
            plugin_path,
            threshold=threshold,
            threshold_max=threshold_max,
            text_mode=text_mode,
            dry_run=dry_run,
            baseline_path=source_path,
            extra={
                "log_file": LOG_FILENAME,
                "steps_file": STEPS_FILENAME,
                "error_file": ERROR_FILENAME,
                "source_custom_tag": source_custom_tag,
            },
        )
    print(f"✅ [SimMerge-Chained] 完成 → {plugin_path}")
    return source_path, plugin_path
