"""
CypherCapablePropertyGraphStore

繼承 SimplePropertyGraphStore，補足 Cypher 查詢支援，
使 LlamaIndex 原生 TextToCypherRetriever 可直接運作。
"""

import re
from typing import Any, Dict, List, Optional

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore


class CypherCapablePropertyGraphStore(SimplePropertyGraphStore):
    """
    在 SimplePropertyGraphStore 基礎上實作 get_schema / structured_query，
    讓 TextToCypherRetriever 能在不依賴 Neo4j 的情況下運作。
    """

    @classmethod
    def from_simple_store(cls, store: SimplePropertyGraphStore) -> "CypherCapablePropertyGraphStore":
        """從既有的 SimplePropertyGraphStore 實例建立包裝器（共用同一份 graph）。"""
        wrapper = cls.__new__(cls)
        SimplePropertyGraphStore.__init__(wrapper, graph=store.graph)
        return wrapper

    @property
    def supports_structured_queries(self) -> bool:  # type: ignore[override]
        return True

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self, refresh: bool = False) -> str:
        """掃描圖中所有節點 / 關係，產生類 Neo4j 風格 schema 字串。"""
        all_nodes = self.graph.get_all_nodes()
        all_relations = self.graph.get_all_relations()

        node_labels: Dict[str, set] = {}
        for node in all_nodes:
            label = getattr(node, "label", "Entity") or "Entity"
            if label not in node_labels:
                node_labels[label] = set()
            props = getattr(node, "properties", None)
            if props:
                node_labels[label].update(props.keys())

        rel_types: Dict[str, set] = {}
        for rel in all_relations:
            label = getattr(rel, "label", "RELATED_TO") or "RELATED_TO"
            if label not in rel_types:
                rel_types[label] = set()
            props = getattr(rel, "properties", None)
            if props:
                rel_types[label].update(props.keys())

        parts: List[str] = ["Node properties:"]
        for label in sorted(node_labels):
            props = sorted(node_labels[label])
            parts.append(f"  {label}: [{', '.join(props)}]")

        parts.append("Relationship types:")
        for rtype in sorted(rel_types):
            parts.append(f"  {rtype}")

        parts.append("Relationship properties:")
        for rtype in sorted(rel_types):
            props = sorted(rel_types[rtype])
            if props:
                parts.append(f"  {rtype}: [{', '.join(props)}]")

        return "\n".join(parts)

    def get_schema_str(self, refresh: bool = False) -> str:
        return self.get_schema(refresh=refresh)

    # ------------------------------------------------------------------
    # Structured Query（將 Cypher 轉為 get_triplets / get 呼叫）
    # ------------------------------------------------------------------

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        從 LLM 生成的 Cypher 中提取查詢意圖，
        透過 get_triplets / get 等 API 在記憶體圖上執行。
        """
        query = query.strip()

        entity_names = self._extract_entity_names_from_cypher(query)
        relation_names = self._extract_relation_names_from_cypher(query)
        node_labels = self._extract_node_labels_from_cypher(query)
        prop_filters = self._extract_property_filters_from_cypher(query)

        results: List[Dict[str, Any]] = []

        # 1) 用 entity / relation 過濾取得三元組
        if entity_names or relation_names:
            triplets = self.get_triplets(
                entity_names=entity_names or None,
                relation_names=relation_names or None,
            )
            for subj, rel, obj in triplets:
                results.append(
                    {
                        "subject": getattr(subj, "name", getattr(subj, "id", str(subj))),
                        "subject_label": getattr(subj, "label", ""),
                        "relation": getattr(rel, "label", str(rel)),
                        "object": getattr(obj, "name", getattr(obj, "id", str(obj))),
                        "object_label": getattr(obj, "label", ""),
                    }
                )

        # 2) 用屬性過濾取得節點
        if prop_filters and not results:
            nodes = self.get(properties=prop_filters)
            for node in nodes:
                results.append(
                    {
                        "name": getattr(node, "name", getattr(node, "id", str(node))),
                        "label": getattr(node, "label", "Entity"),
                        "properties": dict(getattr(node, "properties", {})),
                    }
                )

        # 3) Fallback：按 label 取得節點
        if not results and node_labels:
            all_nodes = self.graph.get_all_nodes()
            for node in all_nodes:
                nlabel = getattr(node, "label", "")
                if nlabel in node_labels:
                    results.append(
                        {
                            "name": getattr(node, "name", getattr(node, "id", str(node))),
                            "label": nlabel,
                            "properties": dict(getattr(node, "properties", {})),
                        }
                    )

        # 4) 如果 label 也沒撈到，嘗試用 label 查三元組
        if not results and node_labels:
            all_triplets = self._get_triplets_by_labels(node_labels)
            for subj, rel, obj in all_triplets:
                results.append(
                    {
                        "subject": getattr(subj, "name", getattr(subj, "id", str(subj))),
                        "relation": getattr(rel, "label", str(rel)),
                        "object": getattr(obj, "name", getattr(obj, "id", str(obj))),
                    }
                )

        return results

    # ------------------------------------------------------------------
    # 內部 Cypher 解析工具
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entity_names_from_cypher(cypher: str) -> List[str]:
        """從 WHERE / 屬性區塊提取實體名稱。"""
        names: List[str] = []
        patterns = [
            r"\.name\s*=\s*['\"]([^'\"]+)['\"]",
            r"\.id\s*=\s*['\"]([^'\"]+)['\"]",
            r"\{name:\s*['\"]([^'\"]+)['\"]",
            r"\{id:\s*['\"]([^'\"]+)['\"]",
            r"CONTAINS\s+['\"]([^'\"]+)['\"]",
            r"=~\s*['\"].*?([^'\"]+).*?['\"]",
        ]
        for pattern in patterns:
            names.extend(re.findall(pattern, cypher, re.IGNORECASE))
        return names

    @staticmethod
    def _extract_relation_names_from_cypher(cypher: str) -> List[str]:
        """提取 [r:REL_TYPE] 或 [:REL_TYPE] 中的關係類型。"""
        matches = re.findall(r"\[\w*:(\w+)\]", cypher)
        return [m for m in matches if m.upper() == m and len(m) > 1]

    @staticmethod
    def _extract_node_labels_from_cypher(cypher: str) -> List[str]:
        """提取 (n:Label) 或 (:Label) 中的節點 label。"""
        matches = re.findall(r"\(\w*:(\w+)", cypher)
        return [m for m in matches if m[0].isupper()]

    @staticmethod
    def _extract_property_filters_from_cypher(cypher: str) -> Dict[str, Any]:
        """提取 {key: 'value'} 格式的屬性過濾條件。"""
        filters: Dict[str, Any] = {}
        for key, value in re.findall(r"\{(\w+):\s*['\"]([^'\"]+)['\"]\}", cypher):
            filters[key] = value
        return filters

    def _get_triplets_by_labels(self, labels: List[str]) -> list:
        """取得指定 label 節點相關的所有三元組。"""
        all_triplets = self.graph.get_triplets()
        matched = []
        label_set = set(labels)
        for subj, rel, obj in all_triplets:
            subj_label = getattr(subj, "label", "")
            obj_label = getattr(obj, "label", "")
            if subj_label in label_set or obj_label in label_set:
                matched.append((subj, rel, obj))
        return matched
