"""
AutoSchemaKG End-to-End Wrapper

完整的 AutoSchemaKG 建圖 + 檢索 Pipeline
"""

import os
import networkx as nx
from typing import Dict, Any, Optional
from src.rag.wrappers.base_wrapper import BaseRAGWrapper
from src.graph_builder.autoschema_builder import AutoSchemaKGBuilder
from llama_index.core import Document


class AutoSchemaWrapper(BaseRAGWrapper):
    """
    AutoSchemaKG 端到端 Wrapper
    
    提供完整的 AutoSchemaKG 建圖與檢索功能
    
    Attributes:
        name: Wrapper 名稱
        builder: AutoSchemaKG Builder 實例
        graph: NetworkX 圖實例
        output_dir: 輸出目錄
    """
    
    def __init__(
        self,
        name: str = "AutoSchemaKG",
        output_dir: str = None,
        documents: list = None,
        model_type: str = "small",
        settings=None,
        schema_info: Dict[str, Any] = None,
        top_k: int = 2,
        builder_config: Dict[str, Any] = None
    ):
        """
        初始化 AutoSchemaKG Wrapper
        
        Args:
            name: Wrapper 名稱
            output_dir: 輸出目錄
            documents: 文檔列表(如需要立即建圖)
            model_type: 模型類型
            schema_info: Schema 資訊
            top_k: 檢索數量
            builder_config: Builder 配置參數
        """
        super().__init__(name, schema_info=schema_info)
        self.model_type = model_type
        self.top_k = top_k
        self.output_dir = output_dir
        self.graph = None
        self.graphml_path = None
        self.settings = settings
        
        # 建立 Builder
        if self.settings is None:
            from src.config.settings import get_settings
            self.settings = get_settings(model_type=self.model_type)

        self.builder = AutoSchemaKGBuilder(
            graph_store=None,
            settings=self.settings,
            output_dir=output_dir
        )
        
        # 初始化 Builder
        if builder_config:
            self.builder.initialize(builder_config)
        
        # 如果提供了文檔,立即建圖
        if documents:
            self._build_graph(documents)
    
    def _build_graph(self, documents: list):
        """
        使用 AutoSchemaKG 建立圖譜
        
        Args:
            documents: 文檔列表
        """
        print(f"🔨 [AutoSchemaKG Wrapper] 開始建圖流程...")
        
        # 檢查是否已有 cache
        if self.output_dir:
            graphml_dir = os.path.join(self.output_dir, "kg_graphml")
            if os.path.exists(graphml_dir):
                graphml_files = [f for f in os.listdir(graphml_dir) if f.endswith('.graphml')]
                if graphml_files:
                    print(f"✅ [AutoSchemaKG Wrapper] 發現已存在的圖譜，直接載入")
                    print(f"📂 GraphML 位置: {graphml_dir}")
                    
                    # 直接載入而不呼叫 builder.build()
                    self._load_graph()
                    
                    # 更新 schema_info (從圖譜推斷或使用預設值)
                    if self.graph:
                        entities = set()
                        relations = set()
                        for u, v, data in self.graph.edges(data=True):
                            label = data.get('label', 'RELATED')
                            relations.add(label)
                        
                        for node in self.graph.nodes():
                            entities.add(self.graph.nodes[node].get('label', 'Entity'))
                        
                        self.schema_info = {
                            "entities": list(entities),
                            "relations": list(relations),
                            "method": "autoschema",
                            "hierarchical": True,
                            "num_nodes": self.graph.number_of_nodes(),
                            "num_edges": self.graph.number_of_edges()
                        }
                    
                    print(f"✅ [AutoSchemaKG Wrapper] 圖譜載入完成")
                    return
        
        # 若無 cache，執行建圖
        print(f"🔨 [AutoSchemaKG] 未找到 cache，開始建立知識圖譜...")
        
        # 轉換為 LlamaIndex Document 格式
        if documents and not isinstance(documents[0], Document):
            documents = [Document(text=str(doc)) for doc in documents]
        
        # 建立圖譜
        try:
            graph_data = self.builder.build(documents)
        except ImportError as e:
            print("❌ [AutoSchemaKG] 缺少必要依賴，請先安裝 `atlas-rag` 與 `networkx`。")
            raise
        except Exception as e:
            print(f"❌ [AutoSchemaKG] 建圖失敗，output_dir={self.output_dir}，錯誤: {e}")
            raise
        
        # 更新 schema_info
        if graph_data.get("schema_info"):
            self.schema_info = graph_data["schema_info"]
        
        self.output_dir = graph_data.get("storage_path")
        
        # 載入 GraphML(如果存在)
        self._load_graph()
        
        print(f"✅ [AutoSchemaKG] 知識圖譜建立完成")
    
    def _load_graph(self):
        """載入 GraphML 圖譜"""
        if not self.output_dir:
            return
        
        # 檢查 kg_graphml 子目錄
        graphml_dir = os.path.join(self.output_dir, "kg_graphml")
        if os.path.exists(graphml_dir):
            graphml_files = [f for f in os.listdir(graphml_dir) if f.endswith('.graphml')]
            if graphml_files:
                self.graphml_path = os.path.join(graphml_dir, graphml_files[0])
                try:
                    self.graph = nx.read_graphml(self.graphml_path)
                    print(f"✅ 已載入 GraphML: {self.graphml_path}")
                    print(f"📊 節點數: {self.graph.number_of_nodes()}, 邊數: {self.graph.number_of_edges()}")
                except Exception as e:
                    print(f"⚠️  載入 GraphML 失敗: {e}")
                return
        
        # 如果 kg_graphml 子目錄不存在，嘗試在 output_dir 直接查找
        graphml_files = [f for f in os.listdir(self.output_dir) if f.endswith('.graphml')]
        
        if graphml_files:
            self.graphml_path = os.path.join(self.output_dir, graphml_files[0])
            try:
                self.graph = nx.read_graphml(self.graphml_path)
                print(f"✅ 已載入 GraphML: {self.graphml_path}")
                print(f"📊 節點數: {self.graph.number_of_nodes()}, 邊數: {self.graph.number_of_edges()}")
            except Exception as e:
                print(f"⚠️  載入 GraphML 失敗: {e}")
        else:
            print("⚠️  未找到 GraphML 檔案")
    
    async def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        執行 AutoSchemaKG 查詢
        
        實作基於概念層級的語義檢索
        
        Args:
            query: 使用者查詢
        
        Returns:
            查詢結果字典
        """
        if self.graph is None:
            # 如果圖譜未建立,嘗試載入
            self._load_graph()
            
            if self.graph is None:
                return {
                    "generated_answer": "圖譜尚未建立或載入失敗",
                    "retrieved_contexts": [],
                    "retrieved_ids": [],
                    "source_nodes": []
                }
        
        print(f"🔍 [AutoSchemaKG] 開始檢索...")
        
        # TODO: 實作完整的 AutoSchemaKG 檢索邏輯
        # 1. 實體連結 (Entity Linking)
        # 2. 基於概念層級的子圖檢索
        # 3. 路徑推理(利用 is-a 關係)
        
        # 簡化版圖譜檢索
        contexts = self._simple_graph_retrieval(query)
        
        # 應用 retrieval token 限制
        contexts = self._truncate_contexts_by_tokens(contexts)
        
        # 使用 LLM 生成答案
        from src.config.settings import get_settings
        Settings = get_settings(model_type=self.model_type)
        
        if contexts:
            context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
            
            prompt = f"""基於以下從知識圖譜檢索到的資訊回答問題:

{context_str}

問題: {query}

請直接回答問題,使用繁體中文。"""
        else:
            prompt = f"""問題: {query}

由於知識圖譜中未找到相關資訊,請基於你的知識回答,並說明這是基於一般知識的回答。使用繁體中文。"""
        
        # 不限制生成 token
        response = self.settings.llm.complete(prompt)
        generated_answer = str(response)
        
        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": contexts,
            "retrieved_ids": [f"autoschema_{i}" for i in range(len(contexts))],
            "source_nodes": [],
            "metadata": {
                "method": "autoschema",
                "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
                "graph_edges": self.graph.number_of_edges() if self.graph else 0
            }
        }
    
    def _simple_graph_retrieval(self, query: str) -> list:
        """
        簡化版圖譜檢索
        
        Args:
            query: 查詢文本
            
        Returns:
            檢索到的上下文列表
        """
        if self.graph is None:
            return []
        
        contexts = []
        
        # 簡單的節點匹配檢索
        query_lower = query.lower()
        matched_nodes = []
        
        for node, data in self.graph.nodes(data=True):
            node_text = str(node).lower()
            # 如果節點文本包含查詢關鍵字
            if any(word in node_text for word in query_lower.split()):
                matched_nodes.append((node, data))
                
                # 取得鄰居節點資訊
                neighbors = list(self.graph.neighbors(node))
                if neighbors:
                    neighbor_info = ", ".join([str(n) for n in neighbors[:3]])
                    context = f"實體 '{node}' 相關連結: {neighbor_info}"
                    contexts.append(context)
        
        # 限制返回數量
        return contexts[:self.top_k]
