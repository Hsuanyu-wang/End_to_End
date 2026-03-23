"""
實體消歧核心引擎

提供通用的實體相似度計算、Union-Find 分群、LLM 驗證與合併邏輯。
可被 LightRAG similar_entity_merge 等上層模組呼叫。
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class _UnionFind:
    """路徑壓縮 Union-Find"""

    def __init__(self) -> None:
        self._p: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._p:
            self._p[x] = x
        if self._p[x] != x:
            self._p[x] = self.find(self._p[x])
        return self._p[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


class EntityDisambiguator:
    """
    實體消歧器（核心引擎）

    流程：
    1. _find_similar_pairs — embedding 相似度 + 字串 fallback
    2. _verify_with_llm    — 可選 LLM 驗證
    3. _perform_merge       — Union-Find 分群 + 代表選擇
    """

    def __init__(
        self,
        llm=None,
        embed_model=None,
        similarity_threshold: float = 0.85,
        use_llm_verification: bool = False,
        degree_fn: Optional[Callable[[Any], int]] = None,
    ):
        """
        Args:
            llm: LLM 實例（用於可選驗證）
            embed_model: Embedding 模型（需有 get_text_embedding_batch）
            similarity_threshold: 餘弦相似度閾值
            use_llm_verification: 是否在合併前用 LLM 驗證
            degree_fn: 計算實體 degree 的函式（用於代表選擇），
                       簽名 degree_fn(entity) -> int
        """
        self.llm = llm
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.use_llm_verification = use_llm_verification
        self.degree_fn = degree_fn

    def merge_entities(self, entities: List[Any]) -> List[Any]:
        """
        合併重複的實體

        Returns:
            去重後的實體列表
        """
        if len(entities) < 2:
            return entities

        print(f"📊 開始實體消歧: {len(entities)} 個實體")

        pairs = self._find_similar_pairs(entities)
        if not pairs:
            print("✅ 沒有發現相似實體")
            return entities

        print(f"🔍 發現 {len(pairs)} 對相似實體")

        if self.use_llm_verification and self.llm:
            pairs = self._verify_with_llm(entities, pairs)
            print(f"✅ LLM 驗證後剩餘 {len(pairs)} 對")

        merged = self._perform_merge(entities, pairs)
        print(f"✅ 消歧完成: {len(entities)} → {len(merged)} 個實體")
        return merged

    # ------------------------------------------------------------------
    # 1. 相似度計算
    # ------------------------------------------------------------------

    def _find_similar_pairs(
        self, entities: List[Any]
    ) -> List[Tuple[int, int, float]]:
        if self.embed_model is not None:
            try:
                return self._embedding_similarity(entities)
            except Exception as e:
                print(f"⚠️  Embedding 計算失敗: {e}，使用字串 fallback")
        return self._string_similarity(entities)

    def _embedding_similarity(
        self, entities: List[Any]
    ) -> List[Tuple[int, int, float]]:
        texts = [self._get_entity_text(e) for e in entities]
        embeddings = np.array(self.embed_model.get_text_embedding_batch(texts))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb_n = embeddings / norms
        sim = emb_n @ emb_n.T

        pairs = []
        n = len(entities)
        for i in range(n):
            for j in range(i + 1, n):
                if float(sim[i, j]) >= self.similarity_threshold:
                    pairs.append((i, j, float(sim[i, j])))
        return pairs

    def _string_similarity(
        self, entities: List[Any]
    ) -> List[Tuple[int, int, float]]:
        texts = [self._get_entity_text(e).lower() for e in entities]
        pairs = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                t1, t2 = texts[i], texts[j]
                if t1 in t2 or t2 in t1:
                    pairs.append((i, j, 0.9))
                else:
                    score = self._jaccard(t1, t2)
                    if score >= self.similarity_threshold:
                        pairs.append((i, j, score))
        return pairs

    @staticmethod
    def _jaccard(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        a, b = set(s1), set(s2)
        return len(a & b) / len(a | b)

    # ------------------------------------------------------------------
    # 2. LLM 驗證
    # ------------------------------------------------------------------

    def _verify_with_llm(
        self,
        entities: List[Any],
        pairs: List[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        """用 LLM 逐對確認是否為同一實體，過濾掉不是的"""
        if not self.llm or not pairs:
            return pairs

        verified = []
        for idx_a, idx_b, score in pairs:
            name_a = self._get_entity_text(entities[idx_a])
            name_b = self._get_entity_text(entities[idx_b])
            desc_a = self._get_entity_desc(entities[idx_a])
            desc_b = self._get_entity_desc(entities[idx_b])

            prompt = (
                f"判斷以下兩個實體是否指同一事物，只回答 YES 或 NO。\n\n"
                f"實體 A: {name_a}"
            )
            if desc_a:
                prompt += f"\n  描述: {desc_a}"
            prompt += f"\n\n實體 B: {name_b}"
            if desc_b:
                prompt += f"\n  描述: {desc_b}"
            prompt += "\n\n回答 (YES/NO):"

            try:
                response = str(self.llm.complete(prompt)).strip().upper()
                if "YES" in response:
                    verified.append((idx_a, idx_b, score))
            except Exception:
                verified.append((idx_a, idx_b, score))

        return verified

    # ------------------------------------------------------------------
    # 3. Union-Find 分群與代表選擇
    # ------------------------------------------------------------------

    def _perform_merge(
        self,
        entities: List[Any],
        pairs: List[Tuple[int, int, float]],
    ) -> List[Any]:
        uf = _UnionFind()
        for i, j, _ in pairs:
            uf.union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(len(entities)):
            root = uf.find(i)
            groups.setdefault(root, []).append(i)

        merged = []
        for members in groups.values():
            group_entities = [entities[i] for i in members]
            representative = self._merge_entity_group(group_entities)
            merged.append(representative)
        return merged

    def _merge_entity_group(self, group: List[Any]) -> Any:
        """
        選擇代表實體：degree 最大者優先，平手取名稱較短者。
        合併所有 descriptions。
        """
        if len(group) == 1:
            return group[0]

        def _sort_key(e):
            deg = self.degree_fn(e) if self.degree_fn else 0
            name = self._get_entity_text(e)
            return (-deg, len(name), name)

        sorted_group = sorted(group, key=_sort_key)
        representative = sorted_group[0]

        all_descs = []
        for e in group:
            desc = self._get_entity_desc(e)
            if desc and desc not in all_descs:
                all_descs.append(desc)

        if len(all_descs) > 1:
            merged_desc = " | ".join(all_descs)
            if hasattr(representative, "description"):
                representative.description = merged_desc
            elif isinstance(representative, dict):
                representative["description"] = merged_desc

        return representative

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_entity_text(entity: Any) -> str:
        if isinstance(entity, dict) and entity.get("_embed_text"):
            return str(entity["_embed_text"])
        for attr in ("name", "text", "label", "id"):
            val = getattr(entity, attr, None)
            if val:
                return str(val)
        if isinstance(entity, dict):
            return str(entity.get("name") or entity.get("id") or entity)
        return str(entity)

    @staticmethod
    def _get_entity_desc(entity: Any) -> str:
        if hasattr(entity, "description"):
            return str(entity.description or "").strip()
        if isinstance(entity, dict):
            return str(entity.get("description", "")).strip()
        return ""

    def resolve_coreference(self, text: str) -> Dict[str, str]:
        """
        共指消解（使用 LLM 識別代詞和實體指涉）

        TODO: 實作共指消解邏輯（屬於文本預處理階段，範疇不同）
        """
        return {}
