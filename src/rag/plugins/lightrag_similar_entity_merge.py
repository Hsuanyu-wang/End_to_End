"""
LightRAG 建圖後之相似實體合併（可選 plugin）

- baseline 索引不修改；自 baseline 複製整個 working_dir 至含 threshold／text_mode 的 custom_tag 目錄後執行合併。
- 使用官方 LightRAG.merge_entities 更新圖與向量儲存。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.storage import get_storage_path
from src.storage.storage_manager import _safe_slug


MARKER_FILENAME = ".lightrag_simmerge_meta.json"
LOG_FILENAME = "lightrag_simmerge_log.json"
STEPS_FILENAME = "lightrag_simmerge_steps.jsonl"
ERROR_FILENAME = "lightrag_simmerge_error.json"

# copytree 時排除的檔案（非圖相關，複製後會污染 simmerge 環境）
_COPY_EXCLUDE_PATTERNS = ("kv_store_llm_response_cache.json",)


def build_simmerge_custom_tag(
    threshold: float, text_mode: str, threshold_max: Optional[float] = None
) -> str:
    """
    依閾值與 text_mode 產生 custom_tag（不同參數對應不同 storage，避免共用錯圖）。
    threshold 固定兩位小數再轉 slug，避免 0.8 與 0.80 變成兩套目錄。
    threshold_max 有設定時附加 _um{上界}（不含該值之合併帶）；None 時與舊版 tag 完全相同。
    """
    t_norm = f"{float(threshold):.2f}".replace(".", "_")
    mode_slug = _safe_slug(text_mode or "name")
    base = f"simmerge_t{t_norm}_{mode_slug}"
    if threshold_max is None:
        return base
    u_norm = f"{float(threshold_max):.2f}".replace(".", "_")
    return f"{base}_um{u_norm}"


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
    threshold: float,
    threshold_max: Optional[float],
    text_mode: str,
    dry_run: bool,
    baseline_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "threshold": float(threshold),
        "threshold_max": threshold_max,
        "text_mode": text_mode,
        "dry_run": dry_run,
        "baseline_path": baseline_path,
        "updated_at": int(time.time()),
    }
    if extra:
        payload.update(extra)
    with open(_marker_path(plugin_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _marker_threshold_max(marker: Dict[str, Any]) -> Optional[float]:
    if "threshold_max" not in marker:
        return None
    v = marker.get("threshold_max")
    if v is None:
        return None
    return float(v)


def _marker_matches(
    marker: Dict[str, Any],
    threshold: float,
    text_mode: str,
    dry_run: bool,
    threshold_max: Optional[float] = None,
) -> bool:
    if marker.get("dry_run") != dry_run:
        return False
    if abs(float(marker.get("threshold", -1)) - float(threshold)) > 1e-9:
        return False
    if str(marker.get("text_mode", "")) != str(text_mode):
        return False
    mm = _marker_threshold_max(marker)
    if threshold_max is None and mm is None:
        return True
    if threshold_max is not None and mm is not None:
        return abs(mm - float(threshold_max)) <= 1e-9
    return False


class _UnionFind:
    """字串 key 版本，供 LightRAG 實體名稱直接使用"""

    def __init__(self) -> None:
        self._p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._p:
            self._p[x] = x
        if self._p[x] != x:
            self._p[x] = self.find(self._p[x])
        return self._p[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


@dataclass
class SimilarMergeResult:
    """合併執行摘要（供日誌與除錯）"""

    baseline_path: str
    plugin_path: str
    num_entities: int
    num_merge_groups: int
    num_merges_called: int
    skipped_reason: str = ""
    dry_run: bool = False


def _entity_text_for_embedding(
    node: Dict[str, Any], text_mode: str, max_chars: int = 2560
) -> str:
    """組合相似度用嵌入文字；長度上限與 LightRAG 嵌入截斷一致，避免 Ollama context 錯誤。"""
    eid = str(node.get("id", "") or node.get("entity_id", ""))
    if text_mode == "name_desc":
        desc = (node.get("description") or "").strip()
        if desc:
            prefix = f"{eid}\n"
            budget = max(0, max_chars - len(prefix))
            if len(desc) > budget:
                desc = desc[:budget]
            return prefix + desc
    if len(eid) > max_chars:
        return eid[:max_chars]
    return eid


def _summarize_merge_entity_result(result: Any) -> Dict[str, Any]:
    """LightRAG merge_entities 回傳 dict 之精簡摘要，避免 log 過大。"""
    if not isinstance(result, dict):
        return {"repr": str(result)[:500]}
    out: Dict[str, Any] = {}
    for k in ("entity_name", "entity_type", "source_id"):
        if k in result:
            out[k] = result[k]
    if "description" in result and isinstance(result["description"], str):
        d = result["description"]
        out["description"] = d if len(d) <= 500 else d[:500] + "..."
    return out


def _write_simmerge_error_json(
    plugin_dir: str,
    *,
    step: int,
    op: Dict[str, Any],
    exc: BaseException,
) -> None:
    p = os.path.join(plugin_dir, ERROR_FILENAME)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": step,
                    "canonical": op.get("canonical"),
                    "sources": op.get("sources"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except OSError as e:
        print(f"⚠️ 無法寫入合併錯誤紀錄: {p} ({e})")


async def _async_gather_degrees(
    graph_inst, names: List[str]
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for n in names:
        try:
            out[n] = int(await graph_inst.node_degree(n))
        except Exception:
            out[n] = 0
    return out


def run_similar_entity_merge(
    rag: Any,
    settings: Any,
    *,
    threshold: float,
    threshold_max: Optional[float] = None,
    text_mode: str = "name",
    dry_run: bool = False,
    log_file: Optional[str] = None,
    use_llm_verify: bool = False,
    baseline_path: str = "",
) -> SimilarMergeResult:
    """
    對已載入之 LightRAG 實例執行相似度合併（同步包裝 async 圖 API）。

    相似度計算使用 EntityDisambiguator 核心引擎，
    合併執行使用 LightRAG 專屬的 rag.merge_entities() API。
    完成後寫入 lightrag_simmerge_log.json；實際合併步驟寫入 lightrag_simmerge_steps.jsonl。
    """
    started_iso = datetime.now(timezone.utc).isoformat()
    started_ts = time.time()
    max_embed = int(getattr(settings.lightrag_config, "embed_max_input_chars", 2560))

    loop = asyncio.get_event_loop()
    graph_inst = rag.chunk_entity_relation_graph
    plugin_path = getattr(rag, "working_dir", "")
    steps_path = (
        os.path.join(os.path.dirname(log_file), STEPS_FILENAME) if log_file else None
    )

    def _write_main_log(
        *,
        pair_logs: List[Dict[str, Any]],
        merge_ops: List[Dict[str, Any]],
        names: List[str],
        groups: Dict[str, List[str]],
        num_merges: int,
        skipped_reason: str = "",
    ) -> None:
        if not log_file:
            return
        finished_iso = datetime.now(timezone.utc).isoformat()
        use_steps = bool(
            steps_path
            and (not dry_run)
            and merge_ops
            and any(op.get("sources") for op in merge_ops)
        )
        payload = {
            "started_at": started_iso,
            "finished_at": finished_iso,
            "duration_sec": round(time.time() - started_ts, 3),
            "baseline_path": baseline_path or None,
            "plugin_path": plugin_path or None,
            "threshold": float(threshold),
            "threshold_max": threshold_max,
            "text_mode": text_mode,
            "dry_run": dry_run,
            "embed_max_input_chars": max_embed,
            "num_entities": len(names),
            "pairs": pair_logs,
            "merge_ops": merge_ops,
            "skipped_reason": skipped_reason,
            "num_merge_groups": len([g for g in groups.values() if len(g) > 1]),
            "num_merges_called": num_merges,
            "steps_file": STEPS_FILENAME if use_steps else None,
            "error_file": ERROR_FILENAME,
        }
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"⚠️ 無法寫入相似合併 log: {log_file} ({e})")

    nodes: List[Dict[str, Any]] = loop.run_until_complete(
        graph_inst.get_all_nodes()
    )
    id_to_node = {str(n.get("id")): n for n in nodes if n.get("id") is not None}
    names = sorted(id_to_node.keys())
    names = [n for n in names if n.strip()]
    if len(names) < 2:
        _write_main_log(
            pair_logs=[],
            merge_ops=[],
            names=names,
            groups={},
            num_merges=0,
            skipped_reason="少於 2 個實體，略過合併",
        )
        return SimilarMergeResult(
            baseline_path=baseline_path,
            plugin_path=plugin_path,
            num_entities=len(names),
            num_merge_groups=0,
            num_merges_called=0,
            skipped_reason="少於 2 個實體，略過合併",
            dry_run=dry_run,
        )

    # 使用 EntityDisambiguator 核心引擎計算相似度
    from src.rag.schema.entity_disambiguation import EntityDisambiguator

    entity_dicts = [id_to_node[n] for n in names]
    for d, name in zip(entity_dicts, names):
        d["_embed_text"] = _entity_text_for_embedding(d, text_mode, max_embed)
        d.setdefault("name", name)

    disambiguator = EntityDisambiguator(
        llm=settings.llm if use_llm_verify else None,
        embed_model=settings.embed_model,
        similarity_threshold=threshold,
        similarity_threshold_max=threshold_max,
        use_llm_verification=use_llm_verify,
    )
    pairs = disambiguator._find_similar_pairs(entity_dicts)

    if use_llm_verify and settings.llm:
        pairs = disambiguator._verify_with_llm(entity_dicts, pairs)

    uf = _UnionFind()
    pair_logs: List[Dict[str, Any]] = []
    for i, j, score in pairs:
        uf.union(names[i], names[j])
        pair_logs.append({"a": names[i], "b": names[j], "similarity": score})

    groups: Dict[str, List[str]] = {}
    for name in names:
        r = uf.find(name)
        groups.setdefault(r, []).append(name)

    degrees = loop.run_until_complete(_async_gather_degrees(graph_inst, names))

    merge_ops: List[Dict[str, Any]] = []
    for _root, members in groups.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members)
        # 代表節點：度最大，平手取字串序較小者
        canonical = sorted(
            members_sorted,
            key=lambda x: (-degrees.get(x, 0), x),
        )[0]
        sources = [m for m in members_sorted if m != canonical]
        merge_ops.append(
            {
                "canonical": canonical,
                "sources": sources,
                "degrees": {m: degrees.get(m, 0) for m in members_sorted},
            }
        )

    num_merges = 0
    if not dry_run:
        err_p = os.path.join(plugin_path, ERROR_FILENAME)
        if os.path.isfile(err_p):
            try:
                os.remove(err_p)
            except OSError:
                pass

        steps_f = None
        if steps_path and any(op.get("sources") for op in merge_ops):
            try:
                steps_f = open(steps_path, "w", encoding="utf-8")
            except OSError as e:
                print(f"⚠️ 無法建立步驟紀錄: {steps_path} ({e})")

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

    _write_main_log(
        pair_logs=pair_logs,
        merge_ops=merge_ops,
        names=names,
        groups=groups,
        num_merges=num_merges,
        skipped_reason="",
    )

    return SimilarMergeResult(
        baseline_path=baseline_path,
        plugin_path=plugin_path,
        num_entities=len(names),
        num_merge_groups=len([g for g in groups.values() if len(g) > 1]),
        num_merges_called=num_merges,
        skipped_reason="",
        dry_run=dry_run,
    )


def ensure_similar_merged_lightrag_index(
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
    use_llm_verify: bool = False,
) -> Tuple[str, str]:
    """
    確保 plugin storage 存在且已完成（或 dry_run 下僅產生 log）。

    Returns:
        (baseline_path, plugin_path) — 後續應以 plugin_path 載入 LightRAG。
    """
    from src.rag.graph.lightrag import _create_lightrag_at_path

    baseline_path = get_storage_path(
        storage_type="lightrag",
        data_type=data_type,
        method=method,
        mode=mode,
        fast_test=fast_test,
        custom_tag="",
    )
    custom_tag = build_simmerge_custom_tag(threshold, text_mode, threshold_max)
    plugin_path = get_storage_path(
        storage_type="lightrag",
        data_type=data_type,
        method=method,
        mode=mode,
        fast_test=fast_test,
        custom_tag=custom_tag,
    )

    if not _lightrag_dir_populated(baseline_path):
        raise FileNotFoundError(
            f"LightRAG baseline 索引不存在或為空: {baseline_path}，請先建圖。"
        )

    # dry_run 不重複使用快取（避免無 marker 時誤刪目錄）
    if not dry_run:
        marker = _read_marker(plugin_path)
        if (
            marker
            and _marker_matches(
                marker, threshold, text_mode, dry_run, threshold_max
            )
            and _lightrag_dir_populated(plugin_path)
            and not force_recopy
        ):
            print(
                f"✅ similar_entity_merge：已存在符合參數之索引，略過複製與合併 → {plugin_path}"
            )
            return baseline_path, plugin_path

    if force_recopy and os.path.isdir(plugin_path):
        print(f"🗑️ similar_entity_merge：force_recopy，移除 {plugin_path}")
        shutil.rmtree(plugin_path, ignore_errors=True)

    # 目錄在但 marker 不符／缺失：非 dry_run 時整包重建以免重複 merge 壞圖
    if not dry_run and _lightrag_dir_populated(plugin_path):
        m = _read_marker(plugin_path)
        if not (
            m
            and _marker_matches(m, threshold, text_mode, dry_run, threshold_max)
        ):
            print(
                "⚠️ similar_entity_merge：plugin 目錄與 marker 不符或缺失，將刪除後自 baseline 複製"
            )
            shutil.rmtree(plugin_path, ignore_errors=True)

    if not _lightrag_dir_populated(plugin_path):
        # get_storage_path 會先 makedirs 出空目錄，copytree 要求 dst 不存在，須先移除空資料夾
        if os.path.isdir(plugin_path):
            shutil.rmtree(plugin_path, ignore_errors=True)
        print(
            f"📋 similar_entity_merge：複製 baseline → plugin（排除 LLM response cache）\n   {baseline_path}\n   → {plugin_path}"
        )
        shutil.copytree(
            baseline_path,
            plugin_path,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns(*_COPY_EXCLUDE_PATTERNS),
        )

    log_path = os.path.join(plugin_path, LOG_FILENAME)
    rag = _create_lightrag_at_path(Settings, plugin_path)
    print(
        f"🔀 similar_entity_merge：threshold={threshold} threshold_max={threshold_max!r} "
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
        baseline_path=baseline_path,
    )
    if not dry_run:
        _write_marker(
            plugin_path,
            threshold=threshold,
            threshold_max=threshold_max,
            text_mode=text_mode,
            dry_run=dry_run,
            baseline_path=baseline_path,
            extra={
                "log_file": LOG_FILENAME,
                "steps_file": STEPS_FILENAME,
                "error_file": ERROR_FILENAME,
            },
        )
    print(f"✅ similar_entity_merge 完成 → {plugin_path}")
    return baseline_path, plugin_path
