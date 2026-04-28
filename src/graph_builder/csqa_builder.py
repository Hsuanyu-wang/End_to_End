# src/graph_builder/csqa_builder.py
from src.graph_builder.base_builder import BaseGraphBuilder

class CSQAGraphBuilder(BaseGraphBuilder):
    def __init__(self, llm_client, vector_db, graph_db):
        self.llm = llm_client
        self.vector_db = vector_db  # 例如 Qdrant
        self.graph_db = graph_db    # 例如 Neo4j

    def build_graph(self, documents):
        for doc in documents:
            # 1. 內部解析：將文件解析為樹狀結構 (Rule-based + LLM)
            tree_nodes, tree_edges = self._intra_ticket_parsing(doc)
            self.graph_db.add_nodes(tree_nodes)
            self.graph_db.add_edges(tree_edges)
            
            # 2. 生成節點向量並存入 Vector DB
            self._generate_and_store_embeddings(tree_nodes)
            
        # 3. 建立工單間的顯性與隱性連結
        self._build_inter_ticket_connections()
        
    def _intra_ticket_parsing(self, doc):
        # 實作 Rule-based 與 LLM YAML 模板解析
        pass

    def _build_inter_ticket_connections(self):
        # 實作 Embedding Cosine Similarity 計算與顯性欄位連結
        pass