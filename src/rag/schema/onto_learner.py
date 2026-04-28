import json
import re
from typing import Any, Iterable, List, Optional
from uuid import uuid4


def _iter_texts(text_corpus: Iterable[Any], sample_docs: int, max_chars_per_doc: int) -> List[str]:
    texts: List[str] = []
    for doc in list(text_corpus)[: max(0, int(sample_docs))]:
        if hasattr(doc, "text"):
            text = getattr(doc, "text")
        else:
            text = str(doc)
        text = (text or "").strip()
        if not text:
            continue
        texts.append(text[: max(0, int(max_chars_per_doc))])
    return texts


def _extract_json_array(text: str) -> Optional[List[str]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    # 嘗試從雜訊中抓出第一個 JSON array
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        return None
    return None


def _is_over_specific(name: str) -> bool:
    # 過度具體：像 Server_001、ErrorCode404、IP_10_0_0_1 這類
    if re.search(r"\d{3,}", name):
        return True
    if re.search(r"[_-]\d+$", name):
        return True
    return False


def _normalize_entity_types(entity_types: List[str], max_entity_types: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in entity_types:
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if not name:
            continue
        if len(name) < 2:
            continue
        if _is_over_specific(name):
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= max(1, int(max_entity_types)):
            break
    return out


def learn_entity_types(
    text_corpus: Iterable[Any],
    llm: Any,
    *,
    max_entity_types: int = 30,
    sample_docs: int = 5,
    max_chars_per_doc: int = 900,
    seed_types: Optional[List[str]] = None,
) -> List[str]:
    """
    OntoLearner（entity-only 版本）

    目標：從語料中歸納 Entity Types（概念/類型），並以 JSON list 回傳。
    約束：速度優先，只產出 entity types，不產出 relations/validation_schema。
    """
    if text_corpus is None:
        raise ValueError("learn_entity_types 需要提供 text_corpus")
    if llm is None:
        raise ValueError("learn_entity_types 需要提供 llm")

    texts = _iter_texts(text_corpus, sample_docs=sample_docs, max_chars_per_doc=max_chars_per_doc)
    if not texts:
        return []

    seed = seed_types or []
    seed_str = json.dumps(seed, ensure_ascii=False)
    sample_text = "\n\n".join(texts)

    prompt = f"""
你是一個領域本體論（Ontology）學習器。
任務：從下列文本歸納「Entity Types（概念/類型）」清單，供後續 in-context-learning 提示使用。

【重要約束】
1. 只輸出 JSON array of strings，例如：["Server","Customer","Issue"]
2. 請輸出「概念/類型」而不是「實例/專有名詞」（例如輸出 "Database" 而不是 "Splunk"）
3. 請保持抽象：不要輸出帶編號或過度具體的型別（例如 "Server_001"）
4. 最多輸出 {int(max_entity_types)} 個

【可參考的 seed types（可用可不用）】
{seed_str}

【文本】
{sample_text}
""".strip()

    # 盡量使用 llama-index LLM 的 complete 介面（Ollama）
    try:
        completion = llm.complete(prompt)
        llm_text = getattr(completion, "text", None) or str(completion)
    except Exception:
        # 避免整個流程失敗：回退成空
        return []

    parsed = _extract_json_array(llm_text)
    if parsed is None:
        # 若 LLM 不守規則，退一步用極小規則抽取（保守）
        # 只抓雙引號字串，避免把大量雜訊當成 entity
        parsed = re.findall(r"\"([^\"]+)\"", llm_text or "")

    return _normalize_entity_types(parsed, max_entity_types=max_entity_types)

