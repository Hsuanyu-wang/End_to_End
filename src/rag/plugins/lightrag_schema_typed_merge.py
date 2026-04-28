"""
LightRAG Schema-Typed SimMerge（STSM）plugin

核心思想：
  先以 schema evolution 取得領域 entity types，再用這些 type embeddings
  作為分群錨點，對圖中 entity 做 type-aware 合併：
    1. 以餘弦相似度把每個 entity 歸類到最近的 schema type
    2. 同 type 群內的 entity pair 若 sim >= sim_threshold 則合併

優勢：
  - 不約束 extraction（graph 以 lightrag_default 建立，保留完整覆蓋率）
  - 可串接在 SimMerge 之後，補捉同語義類別但名稱表達不同的 entity 重複
  - schema 只用於 merge guidance，schema 品質問題不影響 extraction recall
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.storage import get_storage_path
from src.storage.storage_manager import _safe_slug
from src.rag.plugins.lightrag_similar_entity_merge import (
    _UnionFind,
    _entity_text_for_embedding,
    _summarize_merge_entity_result,
    _write_simmerge_error_json,
    _lightrag_dir_populated,
)


STSM_MARKER_FILENAME = ".lightrag_stsm_meta.json"
STSM_LOG_FILENAME = "lightrag_stsm_log.json"
STSM_STEPS_FILENAME = "lightrag_stsm_steps.jsonl"
STSM_ERROR_FILENAME = "lightrag_stsm_error.json"

_COPY_EXCLUDE_PATTERNS = ("kv_store_llm_response_cache.json",)


# ─────────────────────────────────────────────────────────────
# Tag / Marker
# ─────────────────────────────────────────────────────────────


def build_schema_typed_merge_tag(
    schema_method: str,
    sim_threshold: float,
    type_threshold: float,
    entity_text_mode: str = "name",
    type_text_mode: str = "name",
) -> str:
    """
    生成 STSM plugin 的 custom_tag，例如：stm_aae_s0_82_t0_60
    不同參數對應不同 storage，避免共用錯圖。
    """
    method_slug = _safe_slug(schema_method or "aae")
    s_norm = f"{float(sim_threshold):.2f}".replace(".", "_")
    t_norm = f"{float(type_threshold):.2f}".replace(".", "_")
    etm_slug = _safe_slug(entity_text_mode or "name")
    ttm_slug = _safe_slug(type_text_mode or "name")
    return f"stm_{method_slug}_s{s_norm}_t{t_norm}_e{etm_slug}_y{ttm_slug}"


def _stsm_marker_path(plugin_dir: str) -> str:
    return os.path.join(plugin_dir, STSM_MARKER_FILENAME)


def _read_stsm_marker(plugin_dir: str) -> Optional[Dict[str, Any]]:
    p = _stsm_marker_path(plugin_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_stsm_marker(
    plugin_dir: str,
    *,
    schema_method: str,
    sim_threshold: float,
    type_threshold: float,
    entity_text_mode: str,
    type_text_mode: str,
    dry_run: bool,
    source_custom_tag: str,
    baseline_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "schema_method": schema_method,
        "sim_threshold": float(sim_threshold),
        "type_threshold": float(type_threshold),
        "entity_text_mode": entity_text_mode,
        "type_text_mode": type_text_mode,
        "dry_run": dry_run,
        "source_custom_tag": source_custom_tag,
        "baseline_path": baseline_path,
        "updated_at": int(time.time()),
    }
    if extra:
        payload.update(extra)
    with open(_stsm_marker_path(plugin_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _stsm_marker_matches(
    marker: Dict[str, Any],
    schema_method: str,
    sim_threshold: float,
    type_threshold: float,
    entity_text_mode: str,
    type_text_mode: str,
    dry_run: bool,
) -> bool:
    if marker.get("dry_run") != dry_run:
        return False
    if str(marker.get("schema_method", "")) != str(schema_method):
        return False
    if abs(float(marker.get("sim_threshold", -1)) - float(sim_threshold)) > 1e-9:
        return False
    if abs(float(marker.get("type_threshold", -1)) - float(type_threshold)) > 1e-9:
        return False
    if str(marker.get("entity_text_mode", "name")) != str(entity_text_mode):
        return False
    if str(marker.get("type_text_mode", "name")) != str(type_text_mode):
        return False
    return True


def _compose_entity_embed_text(
    node: Dict[str, Any], text_mode: str, max_chars: int
) -> Tuple[str, bool]:
    name = str(
        node.get("name")
        or node.get("id")
        or node.get("entity_name")
        or node.get("entity_id")
        or ""
    ).strip()
    if text_mode == "name_desc":
        desc = str(node.get("description") or "").strip()
        if desc:
            text = f"{name}\n{desc}" if name else desc
            if len(text) > max_chars:
                text = text[:max_chars]
            return text, False
    text = name[:max_chars] if len(name) > max_chars else name
    return text, (text_mode == "name_desc")


def _normalize_schema_type_items(
    schema_types: List[Any],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in schema_types:
        if isinstance(item, str):
            name = item.strip()
            if name:
                out.append({"name": name, "description": ""})
            continue
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("type") or item.get("label") or "").strip()
            if not name:
                continue
            desc = str(item.get("description") or item.get("desc") or "").strip()
            out.append({"name": name, "description": desc})
    return out


def _compose_type_embed_text(
    type_item: Dict[str, str], text_mode: str, max_chars: int
) -> Tuple[str, bool]:
    name = str(type_item.get("name") or "").strip()
    if text_mode == "name_desc":
        desc = str(type_item.get("description") or "").strip()
        if desc:
            text = f"{name}\n{desc}" if name else desc
            if len(text) > max_chars:
                text = text[:max_chars]
            return text, False
    text = name[:max_chars] if len(name) > max_chars else name
    return text, (text_mode == "name_desc")


# ─────────────────────────────────────────────────────────────
# Async 輔助
# ─────────────────────────────────────────────────────────────


async def _async_gather_degrees_stsm(
    graph_inst: Any, names: List[str]
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for n in names:
        try:
            out[n] = int(await graph_inst.node_degree(n))
        except Exception:
            out[n] = 0
    return out


# ─────────────────────────────────────────────────────────────
# 核心合併邏輯
# ─────────────────────────────────────────────────────────────


def run_schema_typed_entity_merge(
    rag: Any,
    settings: Any,
    *,
    schema_types: List[Any],
    sim_threshold: float = 0.82,
    type_threshold: float = 0.60,
    entity_text_mode: str = "name",
    type_text_mode: str = "name",
    dry_run: bool = False,
    log_file: Optional[str] = None,
    baseline_path: str = "",
) -> Dict[str, Any]:
    """
    Schema-Typed Entity Merge：

    1. 對圖中所有 entity names 做 embedding，並對 schema type names 做 embedding
    2. 將每個 entity 歸類到餘弦相似度最大的 schema type（>= type_threshold）
    3. 同 type 群內的 entity pair 若相互餘弦相似度 >= sim_threshold 則加入合併集
    4. Union-Find 聚類 + degree-based canonical 選擇 + rag.merge_entities 執行

    與 SimMerge 的差異：
    - SimMerge 是全局 pairwise 相似度（無型別意識）
    - STSM 只合併同 schema type 群的 entity，允許使用較低 sim_threshold，
      同時避免跨型別的假陽性合併
    """
    started_iso = datetime.now(timezone.utc).isoformat()
    started_ts = time.time()
    max_embed = int(getattr(settings.lightrag_config, "embed_max_input_chars", 2560))
    entity_desc_fallback_count = 0
    type_desc_fallback_count = 0
    schema_type_items = _normalize_schema_type_items(schema_types)

    plugin_path = getattr(rag, "working_dir", "")
    loop = asyncio.get_event_loop()
    graph_inst = rag.chunk_entity_relation_graph

    def _write_log(
        *,
        pairs_count: int,
        merge_ops: List[Dict[str, Any]],
        names: List[str],
        num_merges: int,
        skipped_reason: str = "",
    ) -> None:
        if not log_file:
            return
        payload = {
            "started_at": started_iso,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_sec": round(time.time() - started_ts, 3),
            "baseline_path": baseline_path or None,
            "plugin_path": plugin_path or None,
            "sim_threshold": float(sim_threshold),
            "type_threshold": float(type_threshold),
            "schema_type_count": len(schema_type_items),
            "schema_types": [t.get("name", "") for t in schema_type_items],
            "entity_text_mode": entity_text_mode,
            "type_text_mode": type_text_mode,
            "dry_run": dry_run,
            "embed_max_input_chars": max_embed,
            "entity_desc_fallback_count": entity_desc_fallback_count,
            "entity_desc_fallback_ratio": (
                round(entity_desc_fallback_count / len(names), 4) if names else 0.0
            ),
            "type_desc_fallback_count": type_desc_fallback_count,
            "type_desc_fallback_ratio": (
                round(type_desc_fallback_count / len(schema_type_items), 4)
                if schema_type_items
                else 0.0
            ),
            "num_entities": len(names),
            "type_aware_pairs_found": pairs_count,
            "num_merge_groups": len([op for op in merge_ops if op.get("sources")]),
            "num_merges_called": num_merges,
            "skipped_reason": skipped_reason,
        }
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"⚠️ [STSM] 無法寫入 log: {log_file} ({e})")

    # --- 1. 讀取所有 entities ---
    nodes: List[Dict[str, Any]] = loop.run_until_complete(
        graph_inst.get_all_nodes()
    )
    id_to_node = {str(n.get("id")): n for n in nodes if n.get("id") is not None}
    names = sorted([n for n in id_to_node.keys() if n.strip()])

    if len(names) < 2:
        _write_log(
            pairs_count=0, merge_ops=[], names=names, num_merges=0,
            skipped_reason="少於 2 個實體，略過合併",
        )
        print("[STSM] 少於 2 個實體，略過")
        return {"num_entities": len(names), "num_merges": 0, "skipped": True}

    if not schema_type_items:
        _write_log(
            pairs_count=0, merge_ops=[], names=names, num_merges=0,
            skipped_reason="schema_types 為空，略過",
        )
        print("[STSM] schema_types 為空，略過")
        return {"num_entities": len(names), "num_merges": 0, "skipped": True}

    # --- 2. entity dict 備齊 name 欄位 ---
    entity_dicts = [id_to_node[n] for n in names]
    for d, name in zip(entity_dicts, names):
        d.setdefault("name", name)

    # --- 3. 取得 embed_model ---
    embed_model = settings.embed_model
    if embed_model is None:
        _write_log(
            pairs_count=0, merge_ops=[], names=names, num_merges=0,
            skipped_reason="embed_model 為 None，略過",
        )
        print("[STSM] embed_model 不可用，略過")
        return {"num_entities": len(names), "num_merges": 0, "skipped": True}

    # --- 4. 批次嵌入：entity names + schema type names ---
    entity_texts: List[str] = []
    for node in entity_dicts:
        txt, is_fallback = _compose_entity_embed_text(node, entity_text_mode, max_embed)
        entity_texts.append(txt)
        if is_fallback:
            entity_desc_fallback_count += 1

    schema_type_texts: List[str] = []
    for t in schema_type_items:
        txt, is_fallback = _compose_type_embed_text(t, type_text_mode, max_embed)
        schema_type_texts.append(txt)
        if is_fallback:
            type_desc_fallback_count += 1

    print(
        f"[STSM] 嵌入 {len(names)} 個 entity + {len(schema_type_items)} 個 schema types ..."
    )
    all_texts = entity_texts + schema_type_texts
    try:
        all_embs = np.array(embed_model.get_text_embedding_batch(all_texts))
    except Exception as e:
        print(f"[STSM] embedding 失敗: {e}，略過")
        return {"num_entities": len(names), "num_merges": 0, "skipped": True}

    entity_embs = all_embs[: len(names)]   # [n_entities, dim]
    type_embs = all_embs[len(names) :]     # [n_types, dim]

    # --- 5. L2 正規化 ---
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1.0, norms)

    entity_embs_n = _normalize(entity_embs)
    type_embs_n = _normalize(type_embs)

    # --- 6. Entity → type assignment（各取最近 type）---
    entity_type_sim = entity_embs_n @ type_embs_n.T  # [n_entities, n_types]
    best_type_idxs = entity_type_sim.argmax(axis=1)
    best_type_sims = entity_type_sim.max(axis=1)

    type_groups: Dict[int, List[int]] = {}
    for i, (bt_sim, bt_idx) in enumerate(zip(best_type_sims, best_type_idxs)):
        if float(bt_sim) >= type_threshold:
            type_groups.setdefault(int(bt_idx), []).append(i)

    assigned = sum(len(v) for v in type_groups.values())
    print(
        f"[STSM] {assigned}/{len(names)} 個 entity 歸類到 "
        f"{len(type_groups)} 個 type 群（type_threshold={type_threshold}）"
    )

    # --- 7. 計算 entity-entity 相似度（只在 type 群內配對）---
    entity_entity_sim = entity_embs_n @ entity_embs_n.T  # [n_entities, n_entities]

    uf = _UnionFind()
    pair_count = 0

    for type_idx, group_idxs in type_groups.items():
        if len(group_idxs) < 2:
            continue
        for pi, i in enumerate(group_idxs):
            for j in group_idxs[pi + 1 :]:
                sim_val = float(entity_entity_sim[i, j])
                if sim_val >= sim_threshold:
                    uf.union(names[i], names[j])
                    pair_count += 1

    # --- 8. 組裝 merge ops（Union-Find 分群 + degree-based canonical）---
    groups_map: Dict[str, List[str]] = {}
    for name in names:
        r = uf.find(name)
        groups_map.setdefault(r, []).append(name)

    degrees = loop.run_until_complete(
        _async_gather_degrees_stsm(graph_inst, names)
    )

    merge_ops: List[Dict[str, Any]] = []
    for _root, members in groups_map.items():
        if len(members) < 2:
            continue
        canonical = sorted(members, key=lambda x: (-degrees.get(x, 0), x))[0]
        sources = [m for m in sorted(members) if m != canonical]
        merge_ops.append({
            "canonical": canonical,
            "sources": sources,
            "degrees": {m: degrees.get(m, 0) for m in sorted(members)},
        })

    print(
        f"[STSM] type-aware 相似對: {pair_count}，"
        f"合併群: {len(merge_ops)}，sim_threshold={sim_threshold}"
    )

    # --- 9. 執行合併 ---
    num_merges = 0
    if not dry_run:
        steps_path = (
            os.path.join(os.path.dirname(log_file), STSM_STEPS_FILENAME)
            if log_file
            else None
        )
        err_p = os.path.join(plugin_path, STSM_ERROR_FILENAME) if plugin_path else None
        if err_p and os.path.isfile(err_p):
            try:
                os.remove(err_p)
            except OSError:
                pass

        steps_f = None
        if steps_path and merge_ops:
            try:
                steps_f = open(steps_path, "w", encoding="utf-8")
            except OSError as e:
                print(f"⚠️ [STSM] 無法建立步驟紀錄: {steps_path} ({e})")

        step_i = 0
        try:
            for op in merge_ops:
                src = op["sources"]
                if not src:
                    continue
                try:
                    result = rag.merge_entities(
                        source_entities=list(src),
                        target_entity=str(op["canonical"]),
                    )
                    summary = _summarize_merge_entity_result(result)
                    line = {
                        "step": step_i,
                        "canonical": op["canonical"],
                        "sources": list(src),
                        "ok": True,
                        "result_summary": summary,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    if steps_f:
                        steps_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                        steps_f.flush()
                    num_merges += 1
                    step_i += 1
                except Exception as e:
                    if plugin_path:
                        _write_simmerge_error_json(
                            plugin_path, step=step_i, op=op, exc=e
                        )
                    line = {
                        "step": step_i,
                        "canonical": op["canonical"],
                        "sources": list(src),
                        "ok": False,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    if steps_f:
                        steps_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                        steps_f.flush()
                    raise
        finally:
            if steps_f:
                steps_f.close()

    _write_log(
        pairs_count=pair_count,
        merge_ops=merge_ops,
        names=names,
        num_merges=num_merges,
    )

    return {
        "num_entities": len(names),
        "type_aware_pairs": pair_count,
        "merge_groups": len(merge_ops),
        "num_merges": num_merges,
        "skipped": False,
        "dry_run": dry_run,
    }


# ─────────────────────────────────────────────────────────────
# 確保 plugin storage 存在
# ─────────────────────────────────────────────────────────────


def ensure_schema_typed_merge_lightrag_index(
    Settings: Any,
    *,
    data_type: str,
    method: str,
    mode: str,
    fast_test: bool,
    schema_types: List[Any],
    schema_method_name: str,
    sim_threshold: float = 0.82,
    type_threshold: float = 0.60,
    entity_text_mode: str = "name",
    type_text_mode: str = "name",
    source_custom_tag: str = "",
    force_recopy: bool = False,
    dry_run: bool = False,
) -> Tuple[str, str]:
    """
    確保 STSM plugin storage 存在且已完成。

    從 source_custom_tag 對應目錄複製（source 為空時視同 baseline），
    再執行 type-aware entity merge。

    Tag 格式：
      source_custom_tag="simmerge_t0_90_name_desc"
      → full_tag = "simmerge_t0_90_name_desc_stm_aae_s0_82_t0_60"

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

    stsm_tag = build_schema_typed_merge_tag(
        schema_method_name,
        sim_threshold,
        type_threshold,
        entity_text_mode,
        type_text_mode,
    )
    full_tag = f"{source_custom_tag}_{stsm_tag}" if source_custom_tag else stsm_tag

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
            f"[STSM] 來源 LightRAG 索引不存在或為空: {source_path}，"
            f"請先建圖（或執行 SimMerge 後再啟用 STSM）。"
        )

    # 快取命中
    if not dry_run:
        marker = _read_stsm_marker(plugin_path)
        if (
            marker
            and _stsm_marker_matches(
                marker,
                schema_method_name,
                sim_threshold,
                type_threshold,
                entity_text_mode,
                type_text_mode,
                dry_run,
            )
            and _lightrag_dir_populated(plugin_path)
            and not force_recopy
        ):
            print(f"✅ [STSM] 已存在符合參數之索引，略過 → {plugin_path}")
            return source_path, plugin_path

    if force_recopy and os.path.isdir(plugin_path):
        print(f"🗑️ [STSM] force_recopy，移除 {plugin_path}")
        shutil.rmtree(plugin_path, ignore_errors=True)

    # marker 不符或缺失：整包重建以免重複 merge 壞圖
    if not dry_run and _lightrag_dir_populated(plugin_path):
        m = _read_stsm_marker(plugin_path)
        if not (
            m
            and _stsm_marker_matches(
                m,
                schema_method_name,
                sim_threshold,
                type_threshold,
                entity_text_mode,
                type_text_mode,
                dry_run,
            )
        ):
            print("⚠️ [STSM] plugin 目錄與 marker 不符或缺失，刪除後自 source 複製")
            shutil.rmtree(plugin_path, ignore_errors=True)

    if not _lightrag_dir_populated(plugin_path):
        if os.path.isdir(plugin_path):
            shutil.rmtree(plugin_path, ignore_errors=True)
        print(
            f"📋 [STSM] 複製 source → plugin\n"
            f"   source_tag='{source_custom_tag}'\n"
            f"   {source_path}\n   → {plugin_path}"
        )
        shutil.copytree(
            source_path,
            plugin_path,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns(*_COPY_EXCLUDE_PATTERNS),
        )

    log_path = os.path.join(plugin_path, STSM_LOG_FILENAME)
    rag = _create_lightrag_at_path(Settings, plugin_path)
    print(
        f"🔷 [STSM] sim_threshold={sim_threshold} type_threshold={type_threshold} "
        f"schema_method={schema_method_name} n_types={len(schema_types)} "
        f"entity_text_mode={entity_text_mode} type_text_mode={type_text_mode} dry_run={dry_run}"
    )
    run_schema_typed_entity_merge(
        rag,
        Settings,
        schema_types=schema_types,
        sim_threshold=sim_threshold,
        type_threshold=type_threshold,
        entity_text_mode=entity_text_mode,
        type_text_mode=type_text_mode,
        dry_run=dry_run,
        log_file=log_path,
        baseline_path=source_path,
    )

    if not dry_run:
        _write_stsm_marker(
            plugin_path,
            schema_method=schema_method_name,
            sim_threshold=sim_threshold,
            type_threshold=type_threshold,
            entity_text_mode=entity_text_mode,
            type_text_mode=type_text_mode,
            dry_run=dry_run,
            source_custom_tag=source_custom_tag,
            baseline_path=source_path,
            extra={
                "log_file": STSM_LOG_FILENAME,
                "steps_file": STSM_STEPS_FILENAME,
                "error_file": STSM_ERROR_FILENAME,
            },
        )

    print(f"✅ [STSM] 完成 → {plugin_path}")
    return source_path, plugin_path
