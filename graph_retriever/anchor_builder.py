import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class AnchorBuildResult:
    query_entities: List[str]
    vector_doc_ids: List[str]
    doc_entities: List[str]
    anchor_node_ids: List[str]


def _dedup_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def extract_entities_rule_based(query: str) -> List[str]:
    """
    高精確（保守）規則抽取：
    - CVE-XXXX-YYYY
    - 版本號/代碼（含 . _ -）
    - 英數詞（長度 >= 3）
    - 括號/引號內片段
    """
    q = query or ""
    entities: List[str] = []

    entities.extend(re.findall(r"\bCVE-\d{4}-\d{4,7}\b", q, flags=re.IGNORECASE))

    for m in re.findall(r"[\"'「『](.+?)[\"'」『]", q):
        if 1 <= len(m.strip()) <= 50:
            entities.append(m.strip())

    for m in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9._-]{2,}\b", q):
        # 避免把純數字當 entity
        if re.fullmatch(r"\d+(\.\d+)*", m):
            continue
        entities.append(m)

    return _dedup_keep_order(entities)


def _parse_json_list_maybe(text: str) -> List[str]:
    if not text:
        return []
    t = text.strip()
    # 嘗試擷取第一段 JSON array
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        t = m.group(0)
    try:
        data = json.loads(t)
    except Exception:
        return []
    if isinstance(data, list):
        out = []
        for x in data:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return _dedup_keep_order(out)
    return []


async def extract_entities_llm(
    llm: Any,
    query: str,
    schema_hint: Optional[Dict[str, Any]] = None,
    rule_entities: Optional[List[str]] = None,
) -> List[str]:
    """
    用 LLM 補齊 entities，回傳 JSON array of strings。
    相容 llama_index 的 Settings.llm（通常有 acomplete/complete）。
    """
    schema_part = ""
    if schema_hint:
        try:
            schema_part = f"\n可參考的 schema（JSON）:\n{json.dumps(schema_hint, ensure_ascii=False)}\n"
        except Exception:
            schema_part = ""
    rule_part = ""
    if rule_entities:
        rule_part = f"\n已由規則抽到的 entities（請保留並可補齊）:\n{json.dumps(rule_entities, ensure_ascii=False)}\n"

    prompt = (
        "你是一個資訊抽取器。請從使用者查詢中抽取『實體/關鍵名詞』，"
        "例如：客戶名稱、工程師、系統/產品、版本、主機代號、錯誤代碼、CVE、維護分類等。\n"
        "要求：\n"
        "1) 只輸出 JSON array（例如 [\"Splunk\",\"FWD02\"]），不要輸出其他文字。\n"
        "2) 每個元素是字串，去除多餘空白。\n"
        "3) 以繁體中文思考，但輸出可混合中英。\n"
        f"{schema_part}{rule_part}\n"
        f"查詢：{query}\n"
    )

    raw = ""
    if hasattr(llm, "acomplete"):
        res = await llm.acomplete(prompt)
        raw = getattr(res, "text", None) or str(res)
    else:
        res = llm.complete(prompt)
        raw = getattr(res, "text", None) or str(res)

    return _parse_json_list_maybe(raw)


async def extract_doc_entities_llm(llm: Any, documents: Sequence[Any], *, schema_hint: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    從（少量）documents 內容抽取補強 entities。只回傳 JSON array。
    為避免成本暴增，建議上層先限制 documents 數量。
    """
    snippets: List[str] = []
    for d in documents:
        meta = getattr(d, "metadata", None) or {}
        doc_id = meta.get("NO") or meta.get("atom_id") or meta.get("id") or ""
        text = getattr(d, "text", None) or getattr(d, "page_content", None) or ""
        if not text and hasattr(d, "get_content"):
            text = d.get_content()
        snippets.append(f"[doc_id={doc_id}]\n{text}"[:1200])

    schema_part = ""
    if schema_hint:
        try:
            schema_part = f"\n可參考的 schema（JSON）:\n{json.dumps(schema_hint, ensure_ascii=False)}\n"
        except Exception:
            schema_part = ""

    prompt = (
        "你是一個資訊抽取器。請從下列文件片段中抽取『可作為圖譜 anchor 的實體/關鍵名詞』，"
        "例如：客戶名稱、工程師、系統/產品、主機代號、版本、錯誤代碼、CVE、維護分類。\n"
        "要求：只輸出 JSON array（例如 [\"Splunk\",\"FWD02\"]），不要輸出其他文字。\n"
        f"{schema_part}\n"
        f"文件片段：\n{chr(10).join(snippets)}\n"
    )

    raw = ""
    if hasattr(llm, "acomplete"):
        res = await llm.acomplete(prompt)
        raw = getattr(res, "text", None) or str(res)
    else:
        res = llm.complete(prompt)
        raw = getattr(res, "text", None) or str(res)

    return _parse_json_list_maybe(raw)


def extract_doc_entities_metadata_only(documents: Sequence[Any]) -> Tuple[List[str], List[str]]:
    """
    從 documents 的 metadata 抽出：
    - doc_ids（NO/atom_id）
    - entities（Customer/Engineers/維護類別/event_date 等）
    """
    doc_ids: List[str] = []
    entities: List[str] = []

    for d in documents:
        meta = getattr(d, "metadata", None) or {}
        doc_id = meta.get("NO") or meta.get("atom_id") or meta.get("id")
        if doc_id is not None:
            doc_ids.append(str(doc_id))

        for key in ("Customer", "Engineers", "維護類別", "doc_type", "event_date"):
            val = meta.get(key)
            if not val:
                continue
            if isinstance(val, list):
                entities.extend([str(x).strip() for x in val if str(x).strip()])
            else:
                entities.append(str(val).strip())

    return _dedup_keep_order(doc_ids), _dedup_keep_order(entities)


class AnchorBuilder:
    def __init__(
        self,
        settings: Any,
        *,
        use_vector_docs: bool = False,
        doc_entity_mode: str = "metadata_only",
        use_schema_hint: bool = False,
    ):
        self.settings = settings
        self.use_vector_docs = bool(use_vector_docs)
        self.doc_entity_mode = doc_entity_mode
        self.use_schema_hint = bool(use_schema_hint)

    async def build(
        self,
        *,
        query: str,
        graph: Any,
        documents_for_vector: Optional[Sequence[Any]] = None,
        top_k: int = 2,
        schema_hint: Optional[Dict[str, Any]] = None,
        resolve_anchors_fn: Optional[Any] = None,
    ) -> AnchorBuildResult:
        """
        - query_entities: rule + llm（Hybrid）
        - vector_doc_ids: 可選，由外部 vector retriever 產生後傳入 documents_for_vector
        - doc_entities: 可選，從 vector docs metadata 抽
        - anchor_node_ids: 由 query/doc entities 映射到 graph node ids
        """
        rule_entities = extract_entities_rule_based(query)
        llm_entities = await extract_entities_llm(
            self.settings.builder_llm,
            query=query,
            schema_hint=schema_hint,
            rule_entities=rule_entities,
        )
        query_entities = _dedup_keep_order(rule_entities + llm_entities)

        vector_doc_ids: List[str] = []
        doc_entities: List[str] = []
        doc_contents: List[str] = []

        if self.use_vector_docs and documents_for_vector:
            vector_doc_ids, meta_entities = extract_doc_entities_metadata_only(documents_for_vector)
            if self.doc_entity_mode in ("metadata_only", "metadata"):
                doc_entities = meta_entities
            elif self.doc_entity_mode in ("llm", "richer"):
                # 先用 metadata 當保底（LLM richer 之後再補）
                doc_entities = meta_entities
                try:
                    # 避免成本暴增：只取前 top_k 份文件做補強
                    doc_llm_entities = await extract_doc_entities_llm(
                        self.settings.builder_llm,
                        list(documents_for_vector)[: max(1, top_k)],
                        schema_hint=schema_hint,
                    )
                    doc_entities = _dedup_keep_order(doc_entities + doc_llm_entities)
                except Exception:
                    # LLM 抽取失敗就退回 metadata_only
                    pass
            for d in documents_for_vector:
                text = getattr(d, "text", None) or getattr(d, "page_content", None) or ""
                if text:
                    doc_contents.append(str(text))

        entity_list = _dedup_keep_order(query_entities + doc_entities)

        anchor_node_ids: List[str] = []
        if resolve_anchors_fn is not None and entity_list:
            try:
                anchor_node_ids = resolve_anchors_fn(query, entity_list, graph, doc_contents=doc_contents)
            except TypeError:
                # 兼容舊介面（沒有 doc_contents 參數）
                anchor_node_ids = resolve_anchors_fn(query, entity_list, graph)

        # doc_id anchors（補強）：record_{NO} / NO 直接餵給 expand 的 fallback
        anchor_node_ids = _dedup_keep_order(anchor_node_ids + vector_doc_ids)

        return AnchorBuildResult(
            query_entities=query_entities,
            vector_doc_ids=vector_doc_ids,
            doc_entities=doc_entities,
            anchor_node_ids=anchor_node_ids,
        )

