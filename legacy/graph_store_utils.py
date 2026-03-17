# graph_store_utils.py
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

class SafeNeo4jStore(Neo4jPropertyGraphStore):
    def upsert_nodes(self, nodes):
        for node in nodes:
            if hasattr(node, "metadata") and isinstance(node.metadata, dict):
                node.metadata.pop("kg_nodes", None)
                node.metadata.pop("kg_relations", None)
            if hasattr(node, "properties") and isinstance(node.properties, dict):
                node.properties.pop("kg_nodes", None)
                node.properties.pop("kg_relations", None)
        super().upsert_nodes(nodes)

    def upsert_relations(self, relations):
        for rel in relations:
            if hasattr(rel, "properties") and isinstance(rel.properties, dict):
                rel.properties.pop("kg_nodes", None)
                rel.properties.pop("kg_relations", None)
            if hasattr(rel, "metadata") and isinstance(rel.metadata, dict):
                rel.metadata.pop("kg_nodes", None)
                rel.metadata.pop("kg_relations", None)
        super().upsert_relations(relations)

    def delete_all(self):
        print(">> 正在清空 Neo4j 舊資料...")
        cypher_query = "MATCH (n) DETACH DELETE n"
        with self.client.session(database=self._database) as session:
            session.run(cypher_query)
        print(">> 清理完成。")