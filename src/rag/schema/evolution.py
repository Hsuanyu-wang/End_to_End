import json
import copy
import math
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from llama_index.core import PromptTemplate
from src.config.settings import get_settings
from src.data.processors import data_processing

# 初始 Base Schema（固定錨點，供迭代演化與外部呼叫共用）
# _BASE_SCHEMA = {
#     "entities": ["Record", "Engineer", "Customer", "Action", "Issue"],
#     "relations": ["BELONGS_TO", "HANDLED", "TAKEN_ACTION", "HAS_ISSUE"],
#     "validation_schema": {
#         "Record": ["BELONGS_TO", "TAKEN_ACTION", "HAS_ISSUE"],
#         "Engineer": ["HANDLED"],
#         "Customer": ["BELONGS_TO"],
#     },
# }

_BASE_SCHEMA = {
    "entities": [
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject"
  ]
}

# 1. 定義期望的 Schema 資料結構 (保留您原本的設計)
class DynamicKGSchema(BaseModel):
    entities: List[str] = Field(description="領域專屬的實體類型列表 (Entity Types)")
    relations: List[str] = Field(description="實體之間的關係類型列表 (Relation Types)")
    validation_schema: Dict[str, List[str]] = Field(description="實體與關係的合法連接規則，格式如 {'實體A': ['關係1', '關係2']}")

# 2. 核心：結合 LLM 進行 Schema 的動態演化
def evolve_schema_with_pydantic(current_schema: dict, batch_docs: list, llm) -> dict:
    # 取適當長度的文本，避免超出 Token 限制 (可依據您的 LLM Context Window 調整)
    sample_text = "\n\n".join([doc.text[:] if hasattr(doc, 'text') else str(doc)[:] for doc in batch_docs[:]]) 
    
    # 強化 Prompt：明確告知 LLM 需要「融合」新舊 Schema，而不是每次都重新發明
    prompt_str = """
    你是一個專業的領域本體論（Ontology）與知識圖譜架構師。
    任務：從給定的領域文本中，萃取出專屬的「實體類型」、「關係類型」與「連接規則」。

    【現有 Schema 狀態】
    {current_schema}

    【任務要求】
    1. 閱讀下方文本，找出重要且具代表性的業務實體與關聯。
    2. 若文本中出現現有 Schema 未涵蓋的新概念，請將其「擴充」至 Schema 中。
    3. 若文本內容未提供新資訊，請「保留」現有 Schema，切勿隨意刪除舊有結構。
    4. 提取的類別必須具備「抽象性」（例如提取 "Server" 而不是特定的 "Server_001"）。

    【文本內容】
    {text}
    """
    # """
    # You are a professional ontology and knowledge graph architect.
    # Task: From the given domain text, extract the specific "entity types," "relation types," and "connection rules."

    # [Current Schema Status]
    # {current_schema}

    # [Task Requirements]
    # 1. Read the text below and identify important and representative business entities and associations.
    # 2. If there are new concepts in the text that are not covered by the current schema, please "expand" the schema to include them.
    # 3. If the text does not provide new information, "retain" the existing schema and do not arbitrarily delete existing structures.
    # 4. The extracted classes must possess "abstraction" (for example, extract "Server" rather than a specific "Server_001").

    # [Text Content]
    # {text}
    # """
    
    prompt_template = PromptTemplate(template=prompt_str)
    
    try:
        # 呼叫 LlamaIndex 的 structured_predict，強迫 LLM 吐出符合 Pydantic 格式的 JSON
        response_obj = llm.structured_predict(
            DynamicKGSchema,
            prompt=prompt_template,
            current_schema=json.dumps(current_schema, ensure_ascii=False),
            text=sample_text
        )
        return response_obj.model_dump()
    except Exception as e:
        print(f">> Schema 更新失敗，沿用上一輪狀態: {e}")
        return current_schema

# ─────────────────────────────────────────────────────────
# Anchored Additive Evolution (AAE) — 新方法
# ─────────────────────────────────────────────────────────

# 只回傳「新增建議」的精簡 Pydantic model，降低 structured_predict 失敗率
class SuggestedEntities(BaseModel):
    new_entity_suggestions: List[str] = Field(
        description="本批次文本中出現、但現有類型清單尚未涵蓋的新實體類型建議（若無則回傳空清單）"
    )


def evolve_schema_additive(
    current_entities: list,
    batch_docs: list,
    llm,
    max_chars_per_doc: int = 1000,
) -> list:
    """
    讓 LLM 只回傳「本批次新增建議」，由程式碼 union 合併，避免 LLM 覆寫現有 schema。

    Args:
        current_entities: 目前已確認的實體類型清單
        batch_docs:       本批次文件
        llm:              LLM 實例
        max_chars_per_doc: 每份文件最多取幾個字元，防止超出 context window

    Returns:
        LLM 建議的新增實體類型清單（不含已有的）
    """
    sample_text = "\n\n".join([
        (doc.text[:max_chars_per_doc] if hasattr(doc, "text") else str(doc)[:max_chars_per_doc])
        for doc in batch_docs
    ])

    prompt_str = """
    你是一個知識圖譜架構師。
    以下是現有的實體類型清單，請只回傳「本段文本中出現、但清單尚未涵蓋的新領域概念類型」。

    【現有類型清單】
    {current_entities}

    【文本內容】
    {text}

    【規則】
    1. 只新增，不修改、不刪除現有類型。
    2. 提取的類別必須具備「抽象性」（例如 "Server" 而非 "Server_001"）。
    3. 若文本中無超出現有清單的新概念，回傳空清單。
    """

    prompt_template = PromptTemplate(template=prompt_str)

    try:
        response_obj = llm.structured_predict(
            SuggestedEntities,
            prompt=prompt_template,
            current_entities=json.dumps(current_entities, ensure_ascii=False),
            text=sample_text,
        )
        suggestions = response_obj.new_entity_suggestions or []
        # 過濾掉已存在的（防止 LLM 重複回傳）
        existing_set = {e.lower() for e in current_entities}
        return [s for s in suggestions if s.lower() not in existing_set]
    except Exception as e:
        print(f">> [AAE] LLM 建議取得失敗，本輪跳過: {e}")
        return []


def cluster_sample_documents(documents: list, n_clusters: int, embed_model) -> list:
    """
    對 documents 做 K-means 分群，每群取距離群心最近的代表文件。

    Args:
        documents:  文件列表
        n_clusters: 群數（通常等於 batch 數）
        embed_model: LlamaIndex embed_model 實例

    Returns:
        代表文件列表（長度 ≤ n_clusters）
    """
    try:
        import numpy as np
        from sklearn.cluster import KMeans

        texts = [
            doc.text if hasattr(doc, "text") else str(doc)
            for doc in documents
        ]

        print(f">> [AAE] 對 {len(texts)} 份文件做 K-means 分群（K={n_clusters}）...")
        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)
        embeddings_np = np.array(embeddings)

        actual_k = min(n_clusters, len(documents))
        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init="auto")
        kmeans.fit(embeddings_np)

        representatives = []
        for cluster_id in range(actual_k):
            indices = np.where(kmeans.labels_ == cluster_id)[0]
            center = kmeans.cluster_centers_[cluster_id]
            # 取與群心距離最近的文件
            dists = np.linalg.norm(embeddings_np[indices] - center, axis=1)
            best_idx = indices[np.argmin(dists)]
            representatives.append(documents[best_idx])

        print(f">> [AAE] 分群完成，選出 {len(representatives)} 份代表文件")
        return representatives

    except ImportError as e:
        print(f">> [AAE] 分群所需套件未安裝（{e}），改用順序切片")
        return documents[:n_clusters]
    except Exception as e:
        print(f">> [AAE] 分群失敗（{e}），改用順序切片")
        return documents[:n_clusters]


def run_anchored_additive_evolution(
    documents: list,
    llm,
    max_ratio: float = 1.0,
    batch_ratio: float = 0.1,
    min_frequency: int = 2,
    use_cluster: bool = False,
    embed_model=None,
    max_chars_per_doc: int = 1000,
    use_bootstrap: bool = True,
    use_consolidate: bool = True,
    adaptive_min_frequency: bool = True,
) -> dict:
    """
    Anchored Additive Evolution：以不可變錨點為基礎，透過跨批次頻率計數決定新增實體類型。

    Args:
        documents:             完整文件列表
        llm:                   LLM 實例
        max_ratio:             最多使用多少比例的文件（0.0~1.0）
        batch_ratio:           每批次使用多少比例的文件（預設 0.1）
        min_frequency:         新增 entity type 需在幾個批次中被建議（adaptive=False 時生效）
        use_cluster:           是否使用 embedding 分群取代順序切片
        embed_model:           embedding 模型（use_cluster 或 use_bootstrap=True 時必須提供）
        max_chars_per_doc:     每份文件截斷字元數
        use_bootstrap:         是否先用 embedding cluster 取代表文件做 domain bootstrap schema
        use_consolidate:       是否在迭代完成後執行 schema consolidation（清理值型態、統一英文）
        adaptive_min_frequency: 是否依批次數動態調整 min_frequency（True 時忽略 min_frequency 參數）

    Returns:
        收斂後的 schema dict（含 entities / relations / validation_schema）
    """
    if max_ratio <= 0.0 or not documents:
        return copy.deepcopy(_BASE_SCHEMA)

    total_docs = len(documents)
    max_docs = max(1, int(total_docs * min(max_ratio, 1.0)))
    batch_size = max(1, int(total_docs * batch_ratio))
    subset = documents[:max_docs]

    # ── 修正二：Domain Bootstrap ──────────────────────────────────────────────
    # 用 embedding cluster 取 5 篇代表文件產生領域初始 schema，取代通用 _BASE_SCHEMA
    if use_bootstrap and embed_model is not None:
        print("[AAE] 執行 Domain Bootstrap（embedding cluster 取代表文件）...")
        boot_schema = bootstrap_domain_schema(
            documents=subset,
            llm=llm,
            embed_model=embed_model,
            n_representatives=5,
        )
        initial_entities = boot_schema.get("entities", [])
        if initial_entities:
            print(f"[AAE] Bootstrap schema ({len(initial_entities)} types): {initial_entities}")
        else:
            print("[AAE] Bootstrap 未產生有效 schema，回退至 _BASE_SCHEMA")
            initial_entities = list(_BASE_SCHEMA.get("entities", []))
    else:
        initial_entities = list(_BASE_SCHEMA.get("entities", []))

    base_set = set(initial_entities)

    # 決定要迭代的 batch 列表
    if use_cluster and embed_model is not None:
        n_batches = max(1, math.ceil(max_docs / batch_size))
        representatives = cluster_sample_documents(subset, n_batches, embed_model)
        batches = [[rep] for rep in representatives]
    else:
        batches = [
            subset[i: i + batch_size]
            for i in range(0, max_docs, batch_size)
        ]

    # ── 修正一：Adaptive min_frequency ───────────────────────────────────────
    n_batches = len(batches)
    if adaptive_min_frequency:
        effective_min_freq = max(1, int(n_batches * 0.3))
        print(f"[AAE] Adaptive min_frequency: n_batches={n_batches} → effective={effective_min_freq}")
    else:
        effective_min_freq = min_frequency

    # 跨批次頻率計數
    candidate_counts: dict[str, int] = {}
    current_entities = list(initial_entities)

    pbar = tqdm(enumerate(batches), total=len(batches), desc="AAE Evolving Schema")
    for round_idx, batch in pbar:
        suggestions = evolve_schema_additive(current_entities, batch, llm, max_chars_per_doc)
        for s in suggestions:
            candidate_counts[s] = candidate_counts.get(s, 0) + 1
        pbar.set_postfix({
            "round": round_idx + 1,
            "candidates": len(candidate_counts),
        })
        if suggestions:
            print(f"\n[AAE Round {round_idx + 1}] LLM 建議新增: {suggestions}")

    # 只保留出現頻率 >= effective_min_freq 的候選
    frequent_candidates = {t for t, c in candidate_counts.items() if c >= effective_min_freq}
    pre_consolidate_entities = list(base_set | frequent_candidates)

    print(
        f"\n[AAE] 迭代完成。初始: {len(base_set)} 類，"
        f"新增(freq>={effective_min_freq}): {frequent_candidates}"
    )

    # ── 修正三：Post-evolution Schema Consolidation ───────────────────────────
    if use_consolidate and pre_consolidate_entities:
        print("[AAE] 執行 Schema Consolidation（清理值型態、統一英文）...")
        consolidated = consolidate_schema(pre_consolidate_entities, llm)
        final_entities = consolidated.get("entities", pre_consolidate_entities)
        removed = consolidated.get("removed", [])
        if removed:
            print(f"[AAE] Consolidation 剔除: {removed}")
    else:
        final_entities = pre_consolidate_entities

    print(f"[AAE] 最終 entity types ({len(final_entities)}): {final_entities}")

    return {
        "entities": final_entities,
        "relations": [],
        "validation_schema": {},
    }


# ─────────────────────────────────────────────────────────
# Domain Bootstrap 與 Schema Consolidation（新增）
# ─────────────────────────────────────────────────────────

class BootstrapSchema(BaseModel):
    entities: List[str] = Field(
        description="領域初始實體類型清單（10~20 個，英文 PascalCase，只含實體型態）"
    )


class ConsolidatedSchema(BaseModel):
    entities: List[str] = Field(
        description="整理後保留的實體類型（英文 PascalCase）"
    )
    removed: List[str] = Field(
        description="被剔除的類型（值型態、語意重複、非英文等）"
    )


def bootstrap_domain_schema(
    documents: list,
    llm,
    embed_model,
    n_representatives: int = 5,
    max_chars_per_doc: int = 1500,
) -> dict:
    """
    用 embedding cluster 取 n_representatives 篇代表文件，讓 LLM 自由產生領域初始 schema。
    此 schema 取代通用 _BASE_SCHEMA 作為 AAE 的起始錨點。

    Args:
        documents:         完整文件列表
        llm:               LLM 實例
        embed_model:       embedding 模型（K-means 分群用）
        n_representatives: 取幾篇代表文件（預設 5）
        max_chars_per_doc: 每份文件截斷字元數

    Returns:
        {"entities": [...]} 形式的 schema dict；失敗時回傳空 entities
    """
    representatives = cluster_sample_documents(documents, n_representatives, embed_model)
    if not representatives:
        print(">> [Bootstrap] 無法取得代表文件，跳過 bootstrap")
        return {"entities": []}

    sample_text = "\n\n".join([
        (doc.text[:max_chars_per_doc] if hasattr(doc, "text") else str(doc)[:max_chars_per_doc])
        for doc in representatives
    ])

    prompt_str = """
    你是一個知識圖譜架構師，正在為特定領域建立知識圖譜的 schema。

    請分析以下領域文本，歸納出 10~20 個最重要的「領域實體類型」。

    【要求】
    1. 全部使用英文命名，採用 PascalCase 格式（例如：MaintenanceRecord, Engineer, Customer）
    2. 只包含「實體」型態，嚴格禁止包含以下屬性型態：
       - 時間 / 日期 / 期間（如 Time, Date, TimeRange, Period）
       - 狀態 / 描述 / 說明（如 Status, Description, StatusDescription）
       - 動作 / 流程 / 步驟（如 Action, Process, Step, Operation）
    3. 具備領域專屬性，避免過於通用（不要只寫 "Thing", "Object", "Item"）
    4. 數量控制在 10~20 個

    【領域文本】
    {text}
    """

    prompt_template = PromptTemplate(template=prompt_str)
    try:
        response_obj = llm.structured_predict(
            BootstrapSchema,
            prompt=prompt_template,
            text=sample_text,
        )
        entities = [e.strip() for e in (response_obj.entities or []) if e.strip()]
        return {"entities": entities}
    except Exception as e:
        print(f">> [Bootstrap] LLM 呼叫失敗（{e}），回退至 _BASE_SCHEMA")
        return {"entities": []}


def consolidate_schema(candidate_entities: List[str], llm) -> dict:
    """
    對迭代演化後的候選 entity types 清單執行 LLM 整理：
    剔除值型態、統一英文 PascalCase、合併語意重複項。

    Args:
        candidate_entities: 待整理的 entity types 清單（可能含中文、值型態）
        llm:                LLM 實例

    Returns:
        {"entities": [...], "removed": [...]}；失敗時回傳原清單（無剔除）
    """
    if not candidate_entities:
        return {"entities": [], "removed": []}

    candidates_str = "\n".join(f"- {e}" for e in candidate_entities)

    prompt_str = """
    以下是從領域文本迭代萃取出的實體類型候選清單，請進行整理與清洗：

    【候選清單】
    {candidates}

    【整理要求】
    1. 剔除值型態：時間/日期/狀態/描述/動作/流程/步驟等「不是事物本身」的類型
    2. 全部翻譯成英文，使用 PascalCase 命名（例如 MaintenanceRecord, Engineer）
    3. 合併語意重複的類型（例如 Customer 和 客户 → 保留 Customer）
    4. 確保抽象層級一致（不接受 instance-level 名稱如 "Server_001"）
    5. 回傳兩個清單：
       - entities：整理後保留的類型（英文 PascalCase）
       - removed：被剔除或合併的原始類型名稱
    """

    prompt_template = PromptTemplate(template=prompt_str)
    try:
        response_obj = llm.structured_predict(
            ConsolidatedSchema,
            prompt=prompt_template,
            candidates=candidates_str,
        )
        entities = [e.strip() for e in (response_obj.entities or []) if e.strip()]
        removed = [r.strip() for r in (response_obj.removed or []) if r.strip()]
        if not entities:
            print(">> [Consolidate] LLM 回傳空結果，保留原清單")
            return {"entities": candidate_entities, "removed": []}
        return {"entities": entities, "removed": removed}
    except Exception as e:
        print(f">> [Consolidate] LLM 呼叫失敗（{e}），保留原清單")
        return {"entities": candidate_entities, "removed": []}


# 3. 可重用的迭代演化主邏輯
def run_iterative_evolution(
    documents: list,
    llm,
    max_ratio: float = 1.0,
    batch_ratio: float = 0.1,
) -> dict:
    """
    對 documents 子集進行迭代 Schema 演化。

    Args:
        documents:   完整文件列表（順序需穩定）
        llm:         LLM 實例
        max_ratio:   最多使用多少比例的文件（0.0~1.0）；0.0 直接回傳初始 base schema
        batch_ratio: 每批次使用多少比例的文件（預設 0.1 = 10%）

    Returns:
        收斂後的 schema dict
    """
    import copy

    dynamic_schema = copy.deepcopy(_BASE_SCHEMA)

    if max_ratio <= 0.0 or not documents:
        return dynamic_schema

    total_docs = len(documents)
    max_docs = max(1, int(total_docs * min(max_ratio, 1.0)))
    batch_size = max(1, int(total_docs * batch_ratio))
    subset = documents[:max_docs]

    pbar = tqdm(range(0, max_docs, batch_size), desc="Evolving Schema")
    for i in pbar:
        batch = subset[i : i + batch_size]
        old_entities = set(dynamic_schema.get("entities", []))
        dynamic_schema = evolve_schema_with_pydantic(dynamic_schema, batch, llm)
        new_entities = set(dynamic_schema.get("entities", []))
        added = new_entities - old_entities
        pbar.set_postfix({"round": i // batch_size + 1, "entities": len(new_entities)})
        if added:
            print(f"\n[Round {i // batch_size + 1}] 新增實體: {added}")

    return dynamic_schema


# 4. 主流程：文件批次處理與 Schema 收斂
def main():
    print(">> 正在載入文件...")
    documents = data_processing(mode="natural_text", data_type="DI")
    total_docs = len(documents)
    print(f">> 總文件數: {total_docs}")
    
    # 取得設定
    settings = get_settings()

    print(f">> 開始進行 Schema 動態演化 (每批次 10% 文件)...")
    dynamic_schema = run_iterative_evolution(
        documents=documents,
        llm=settings.llm,
        max_ratio=1.0,
        batch_ratio=0.1,
    )

    print("\n>> 領域本體論 (Schema) 萃取完成！")
    
    # 4. 將收斂完成的 Schema 存出
    output_path = "/home/End_to_End_RAG/custom_domain_schema.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dynamic_schema, f, indent=4, ensure_ascii=False)
        
    print(f">> 最終 Schema 已儲存至: {output_path}")
    print("\n最終 Schema 預覽:")
    print(json.dumps(dynamic_schema, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()