"""
AutoSchemaKG Builder

整合 AutoSchemaKG 的建圖邏輯作為 Builder 模組
"""

import os
import json
import glob
import tempfile
from typing import List, Dict, Any
from llama_index.core import Document
from src.graph_builder.base_builder import BaseGraphBuilder


class AutoSchemaKGBuilder(BaseGraphBuilder):
    """
    AutoSchemaKG Graph Builder
    
    使用 AutoSchemaKG 的三元組抽取和概念化層級組織建立知識圖譜
    
    參考: https://github.com/hkust-knowcomp/autoschemakg
    """
    
    def __init__(self, graph_store: Any = None, settings: Any = None, output_dir: str = None, force_rebuild: bool = False):
        """
        初始化 AutoSchemaKG Builder
        
        Args:
            graph_store: 圖譜儲存後端(可選)
            settings: 配置設定(可選)
            output_dir: 輸出目錄(可選,預設使用臨時目錄)
            force_rebuild: 是否強制重建圖譜(預設 False)
        """
        super().__init__(graph_store, settings)
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="autoschema_")
        self.kg_extractor = None
        self.force_rebuild = force_rebuild
        if output_dir:
            print(f"📂 [AutoSchemaKG] 使用固定輸出路徑: {self.output_dir}")
        else:
            print(f"⚠️ [AutoSchemaKG] 未指定輸出路徑，改用暫存路徑: {self.output_dir}")
    
    def get_name(self) -> str:
        return "AutoSchemaKG"

    def _normalize_openai_base_url(self, url: str) -> str:
        """將 Ollama/OpenAI 相容 URL 正規化為以 /v1 結尾。"""
        if not url:
            return url
        url = url.rstrip("/")
        if url.endswith("/v1"):
            return url
        return f"{url}/v1"

    def _infer_model_and_base_url_from_settings(self) -> Dict[str, str]:
        """
        從 injected Settings 推導 autoschema 需要的 base_url / model_name。

        - base_url: 以 Settings.builder_llm.base_url 為主（若可取得）
        - model_name: 以 Settings.builder_llm.model 為主（若可取得）
        """
        model_name = None
        base_url = None

        llm = getattr(self.settings, "builder_llm", None) if self.settings is not None else None
        if llm is not None:
            base_url = getattr(llm, "base_url", None) or getattr(llm, "base_url_", None)
            model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)

        if base_url:
            base_url = self._normalize_openai_base_url(str(base_url))

        return {
            "base_url": base_url,
            "model_name": model_name,
        }
    
    def initialize(self, config: Dict[str, Any] = None):
        """
        初始化 AutoSchemaKG 組件
        
        Args:
            config: 配置參數,可包含:
                - model_name: LLM 模型名稱
                - batch_size_triple: 三元組抽取批次大小
                - batch_size_concept: 概念生成批次大小
                - max_workers: 並發工作數
        """
        try:
            from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
            from atlas_rag.kg_construction.triple_config import ProcessingConfig
            from atlas_rag.llm_generator import LLMGenerator
            from openai import OpenAI
        except ImportError:
            print("⚠️  AutoSchemaKG 相關套件未安裝,無法使用此 Builder")
            print("   請安裝: pip install atlas-rag")
            raise
        
        config = config or {}

        inferred = self._infer_model_and_base_url_from_settings()
        base_url = config.get("base_url") or inferred.get("base_url") or "http://192.168.63.174:11434/v1"
        model_name = config.get("model_name") or inferred.get("model_name") or "llama3.3:latest"
        base_url = self._normalize_openai_base_url(base_url)
        
        # 設定 LLM 客戶端
        client = OpenAI(
            base_url=base_url,
            api_key=config.get('api_key', 'ollama')
        )
        
        triple_generator = LLMGenerator(client, model_name=model_name)
        
        # 建立配置
        kg_config = ProcessingConfig(
            model_path=model_name,
            data_directory=self.output_dir,
            filename_pattern="input_documents.jsonl",
            batch_size_triple=config.get('batch_size_triple', 3),
            batch_size_concept=config.get('batch_size_concept', 16),
            output_directory=self.output_dir,
            max_new_tokens=config.get('max_new_tokens', 2048),
            max_workers=config.get('max_workers', 3),
            remove_doc_spaces=True,
        )
        
        self.kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_config)
        print(f"✅ AutoSchemaKG 初始化完成,輸出目錄: {self.output_dir}")
        print(f"   - base_url: {base_url}")
        print(f"   - model_name: {model_name}")
    
    def build(self, documents: List[Document]) -> Dict[str, Any]:
        """
        使用 AutoSchemaKG 建立知識圖譜
        
        Args:
            documents: 文檔列表
            
        Returns:
            標準化的圖譜資料字典
        """
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 檢查 cache: 若 GraphML 檔案已存在且不強制重建，直接載入
        graphml_dir = os.path.join(self.output_dir, "kg_graphml")
        if not self.force_rebuild and os.path.exists(graphml_dir):
            graphml_files = [f for f in os.listdir(graphml_dir) if f.endswith('.graphml')]
            if graphml_files:
                print(f"✅ [AutoSchemaKG] 發現已存在的圖譜，跳過建圖流程")
                print(f"📂 GraphML 位置: {graphml_dir}")
                print(f"📊 找到 {len(graphml_files)} 個 GraphML 檔案")
                print(f"💡 提示: 若需重新建圖，請設定 force_rebuild=True")
                
                # 直接解析並返回
                nodes, edges, schema_info = self._parse_autoschema_output()
                
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "metadata": {
                        "num_documents": len(documents),
                        "output_dir": self.output_dir,
                        "cached": True
                    },
                    "schema_info": schema_info,
                    "storage_path": self.output_dir,
                    "graph_format": "autoschema"
                }
        print(f"ℹ️ [AutoSchemaKG] Cache miss，將進行建圖流程。目錄: {self.output_dir}")
        
        if self.kg_extractor is None:
            print("🔧 自動初始化 AutoSchemaKG...")
            self.initialize()
        
        print(f"🔨 [AutoSchemaKG] 開始建立知識圖譜...")
        print(f"📝 處理 {len(documents)} 份文檔...")
        
        if self.force_rebuild:
            print("⚠️  強制重建模式已啟用")
        
        # 1. 準備 JSONL 格式輸入 (簡化格式: {id, text})
        input_file = os.path.join(self.output_dir, "input_documents.jsonl")
        print(f"📄 準備輸入檔案: {input_file}")
        
        with open(input_file, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(documents):
                # 從 LlamaIndex Document 提取純文字內容
                if hasattr(doc, 'text'):
                    text_content = doc.text
                elif hasattr(doc, 'get_content'):
                    text_content = doc.get_content()
                else:
                    text_content = str(doc)
                
                # 提取 metadata (如果有的話)
                metadata = {}
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata = doc.metadata
                
                # AutoSchemaKG 期待的格式 (需包含 metadata 欄位)
                entry = {
                    "id": f"doc_{i}",
                    "text": text_content,
                    "metadata": metadata
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✅ 輸入檔案已準備: {input_file}")
        
        # 驗證檔案內容
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            print(f"📋 首筆資料預覽: {first_line[:100]}...")
        
        file_size = os.path.getsize(input_file)
        print(f"📊 檔案大小: {file_size / 1024:.2f} KB")
        
        # 2. 執行三元組抽取
        print("🔬 開始三元組抽取...")
        try:
            self.kg_extractor.run_extraction()
            self.kg_extractor.convert_json_to_csv()
            print("✅ 三元組抽取完成")
            
            # 檢查生成的三元組檔案
            extraction_dir = os.path.join(self.output_dir, "kg_extraction")
            if os.path.exists(extraction_dir):
                extraction_files = os.listdir(extraction_dir)
                print(f"   生成的抽取檔案: {extraction_files}")
        except Exception as e:
            print(f"⚠️  三元組抽取過程出現錯誤: {e}")
            print(f"   錯誤類型: {type(e).__name__}")
            print(f"   輸出目錄: {self.output_dir}")
            import traceback
            print(f"   錯誤堆疊: {traceback.format_exc()}")
            print("   將使用已有數據繼續...")
        
        # 3. 執行概念生成
        print("🧠 開始概念生成與層級化...")
        try:
            self.kg_extractor.generate_concept_csv_temp(batch_size=64)
            self.kg_extractor.create_concept_csv()
            print("✅ 概念生成完成")
            
            # 檢查生成的概念檔案
            concepts_dir = os.path.join(self.output_dir, "concepts")
            if os.path.exists(concepts_dir):
                concept_files = os.listdir(concepts_dir)
                print(f"   生成的概念檔案: {concept_files}")
        except Exception as e:
            print(f"⚠️  概念生成過程出現錯誤: {e}")
            print(f"   錯誤類型: {type(e).__name__}")
            import traceback
            print(f"   錯誤堆疊: {traceback.format_exc()}")
        
        # 4. 轉換為 GraphML
        print("📊 轉換為 GraphML 格式...")
        try:
            self.kg_extractor.convert_to_graphml()
            print("✅ GraphML 轉換完成")
            
            # 檢查生成的 GraphML 檔案
            graphml_dir = os.path.join(self.output_dir, "kg_graphml")
            if os.path.exists(graphml_dir):
                graphml_files = [f for f in os.listdir(graphml_dir) if f.endswith('.graphml')]
                print(f"   生成的 GraphML 檔案: {graphml_files}")
            else:
                print(f"⚠️  未找到 GraphML 目錄: {graphml_dir}")
        except Exception as e:
            print(f"⚠️  GraphML 轉換出現錯誤: {e}")
            print(f"   錯誤類型: {type(e).__name__}")
            import traceback
            print(f"   錯誤堆疊: {traceback.format_exc()}")
        
        # 5. 解析輸出並轉換為標準格式
        nodes, edges, schema_info = self._parse_autoschema_output()
        
        print(f"✅ [AutoSchemaKG] 知識圖譜建立完成")
        print(f"   節點數: {len(nodes)}, 邊數: {len(edges)}")
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "num_documents": len(documents),
                "output_dir": self.output_dir,
                "cached": False
            },
            "schema_info": schema_info,
            "storage_path": self.output_dir,
            "graph_format": "autoschema"
        }
    
    def _parse_autoschema_output(self) -> tuple:
        """
        解析 AutoSchemaKG 輸出檔案
        
        Returns:
            (nodes, edges, schema_info) 元組
        """
        nodes = []
        edges = []
        entities = set()
        relations = set()
        
        print(f"🔍 開始解析 AutoSchemaKG 輸出...")
        print(f"   輸出目錄: {self.output_dir}")
        
        # 檢查輸出目錄結構
        if os.path.exists(self.output_dir):
            all_files = []
            for root, dirs, files in os.walk(self.output_dir):
                for f in files:
                    rel_path = os.path.relpath(os.path.join(root, f), self.output_dir)
                    all_files.append(rel_path)
            print(f"   所有生成檔案: {all_files}")
        
        # 檢查 GraphML 目錄
        graphml_dir = os.path.join(self.output_dir, "kg_graphml")
        if not os.path.exists(graphml_dir):
            print(f"⚠️  未找到 GraphML 目錄: {graphml_dir}")
            schema_info = {
                "entities": [],
                "relations": [],
                "method": "autoschema",
                "hierarchical": True,
                "status": "no_graphml"
            }
            return nodes, edges, schema_info
        
        # 尋找 GraphML 檔案
        graphml_files = glob.glob(os.path.join(graphml_dir, "*.graphml"))
        if not graphml_files:
            print(f"⚠️  GraphML 目錄中未找到 .graphml 檔案")
            print(f"   目錄內容: {os.listdir(graphml_dir)}")
            schema_info = {
                "entities": [],
                "relations": [],
                "method": "autoschema",
                "hierarchical": True,
                "status": "no_graphml_files"
            }
            return nodes, edges, schema_info
        
        print(f"📊 找到 {len(graphml_files)} 個 GraphML 檔案")
        
        # 解析 GraphML 檔案
        try:
            import networkx as nx
            
            for gml_file in graphml_files:
                print(f"   正在解析: {os.path.basename(gml_file)}")
                
                try:
                    G = nx.read_graphml(gml_file)
                    print(f"   節點數: {G.number_of_nodes()}, 邊數: {G.number_of_edges()}")
                    
                    # 提取節點
                    for node_id in G.nodes():
                        node_data = G.nodes[node_id]
                        node_dict = {
                            "id": node_id,
                            **dict(node_data)
                        }
                        nodes.append(node_dict)
                        
                        # 收集實體類型
                        if "type" in node_data:
                            entities.add(node_data["type"])
                        elif "label" in node_data:
                            entities.add(node_data["label"])
                    
                    # 提取邊
                    for edge in G.edges(data=True):
                        source, target, edge_data = edge
                        edge_dict = {
                            "source": source,
                            "target": target,
                            **dict(edge_data)
                        }
                        edges.append(edge_dict)
                        
                        # 收集關係類型
                        if "relation" in edge_data:
                            relations.add(edge_data["relation"])
                        elif "label" in edge_data:
                            relations.add(edge_data["label"])
                        elif "type" in edge_data:
                            relations.add(edge_data["type"])
                
                except Exception as e:
                    print(f"⚠️  解析 {gml_file} 時出錯: {e}")
                    import traceback
                    print(f"   錯誤堆疊: {traceback.format_exc()}")
            
            print(f"✅ GraphML 解析完成")
            print(f"   總節點數: {len(nodes)}")
            print(f"   總邊數: {len(edges)}")
            print(f"   實體類型: {len(entities)}")
            print(f"   關係類型: {len(relations)}")
            
        except ImportError:
            print("⚠️  NetworkX 未安裝,無法解析 GraphML")
            print("   請安裝: pip install networkx")
        
        schema_info = {
            "entities": list(entities),
            "relations": list(relations),
            "method": "autoschema",
            "hierarchical": True,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "status": "success"
        }
        
        return nodes, edges, schema_info
    
    def cleanup(self):
        """清理臨時檔案"""
        # 保留輸出檔案以供後續檢索使用
        print(f"💾 AutoSchemaKG 輸出已保留於: {self.output_dir}")
