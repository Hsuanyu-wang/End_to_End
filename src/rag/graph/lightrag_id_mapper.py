"""
ChunkIDMapper 模組

提供 LightRAG chunk ID 到原始文檔 NO 的映射機制
"""

import json
import os
from typing import Dict, Optional, List, Any
from pathlib import Path


class ChunkIDMapper:
    """
    Chunk ID 映射器
    
    負責記錄和查詢 LightRAG chunk ID 到原始 Document NO 的映射關係
    """
    
    def __init__(self, storage_dir: str):
        """
        初始化 ChunkIDMapper
        
        Args:
            storage_dir: LightRAG storage 目錄路徑
        """
        self.storage_dir = storage_dir
        self.mapping_file = os.path.join(storage_dir, "chunk_id_mapping.json")
        self.mapping: Dict[str, str] = {}
        self._load_mapping()
    
    def _load_mapping(self):
        """從檔案載入映射表"""
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    self.mapping = json.load(f)
                print(f"📖 已載入 {len(self.mapping)} 個 chunk ID 映射")
            except Exception as e:
                print(f"⚠️  載入映射表失敗: {e}")
                self.mapping = {}
        else:
            self.mapping = {}
    
    def _save_mapping(self):
        """儲存映射表到檔案"""
        try:
            os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)
            with open(self.mapping_file, "w", encoding="utf-8") as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=2)
            print(f"💾 已儲存 {len(self.mapping)} 個 chunk ID 映射")
        except Exception as e:
            print(f"⚠️  儲存映射表失敗: {e}")
    
    def add_mapping(self, chunk_id: str, original_no: str):
        """
        新增映射關係
        
        Args:
            chunk_id: LightRAG 生成的 chunk ID
            original_no: 原始文檔的 NO 欄位
        """
        self.mapping[chunk_id] = original_no
    
    def add_mappings_batch(self, mappings: Dict[str, str]):
        """
        批次新增映射關係
        
        Args:
            mappings: chunk_id -> original_no 的字典
        """
        self.mapping.update(mappings)
        self._save_mapping()
    
    def get_original_no(self, chunk_id: str, debug: bool = False) -> Optional[str]:
        """
        取得原始文檔 NO
        
        Args:
            chunk_id: LightRAG chunk ID
            debug: 是否輸出debug資訊
        
        Returns:
            原始文檔 NO，若不存在則返回 None
        """
        result = self.mapping.get(chunk_id, None)
        
        if debug:
            if result:
                print(f"✅ 映射成功: {chunk_id} -> {result}")
            else:
                print(f"⚠️  映射失敗: {chunk_id} (未找到對應的原始NO)")
                if len(self.mapping) == 0:
                    print(f"   提示: 映射表為空，可能需要重新建立映射")
        
        return result
    
    def get_original_nos(self, chunk_ids: List[str], debug: bool = False) -> List[str]:
        """
        批次取得原始文檔 NO
        
        Args:
            chunk_ids: LightRAG chunk ID 列表
            debug: 是否輸出debug資訊
        
        Returns:
            原始文檔 NO 列表（保留順序，未找到的返回 chunk_id 本身）
        """
        result = []
        unmapped_count = 0
        
        for chunk_id in chunk_ids:
            original_no = self.get_original_no(chunk_id, debug=False)
            if original_no:
                result.append(original_no)
            else:
                # 如果找不到映射，可能 chunk_id 本身就是 NO（向後相容）
                result.append(chunk_id)
                unmapped_count += 1
        
        if debug and unmapped_count > 0:
            print(f"⚠️  有 {unmapped_count}/{len(chunk_ids)} 個chunk ID無法映射")
            print(f"   映射表大小: {len(self.mapping)}")
            print(f"   建議檢查chunk_id_mapping.json是否正確建立")
        
        return result
    
    def validate_mapping(self) -> Dict[str, Any]:
        """
        驗證映射表的完整性
        
        Returns:
            驗證結果字典
        """
        validation = {
            "total_mappings": len(self.mapping),
            "mapping_file_exists": os.path.exists(self.mapping_file),
            "storage_dir_exists": os.path.exists(self.storage_dir),
            "sample_mappings": dict(list(self.mapping.items())[:5]) if self.mapping else {},
            "is_valid": len(self.mapping) > 0
        }
        
        return validation
    
    def extract_and_save_from_lightrag(self, rag_instance):
        """
        從 LightRAG 實例中提取映射關係並儲存
        
        這個方法需要訪問 LightRAG 的內部資料結構
        
        Args:
            rag_instance: LightRAG 實例
        """
        try:
            # 嘗試從 LightRAG 的 chunk storage 中提取映射
            # 注意：這需要根據 LightRAG 的實際資料結構調整
            
            # 方法 1: 從 chunk_vdb 中提取
            if hasattr(rag_instance, "chunk_vdb"):
                # LightRAG 的 chunk vector database
                # 需要找出如何從 chunk 取得 full_doc_id 和原始 NO
                pass
            
            # 方法 2: 從 key-value storage 中提取
            if hasattr(rag_instance, "key_string_value_json_storage_cls"):
                # 可能需要讀取特定的 JSON 檔案
                pass
            
            print("⚠️  extract_and_save_from_lightrag 需要根據 LightRAG 版本調整實作")
            
        except Exception as e:
            print(f"⚠️  從 LightRAG 提取映射失敗: {e}")
    
    def build_mapping_from_documents(self, documents: List, chunk_texts: List[str]):
        """
        從文檔列表和 chunk 文本建立映射
        
        使用內容匹配來建立 chunk ID 到原始 NO 的映射
        
        Args:
            documents: LlamaIndex Document 列表（包含 doc_id 和 metadata["NO"]）
            chunk_texts: LightRAG 切分的 chunk 文本列表
        """
        from lightrag.lightrag import compute_mdhash_id
        
        mappings = {}
        
        for chunk_text in chunk_texts:
            # 計算 LightRAG 的 chunk ID
            chunk_id = compute_mdhash_id(chunk_text, prefix="chunk-")
            
            # 嘗試從 chunk 文本中找出對應的文檔
            # 方法：檢查哪個文檔的文本包含這個 chunk
            for doc in documents:
                if chunk_text in doc.text or doc.text in chunk_text:
                    # 優先使用 doc_id，否則使用 metadata["NO"]
                    original_no = doc.doc_id or doc.metadata.get("NO")
                    if original_no:
                        mappings[chunk_id] = original_no
                        break
        
        self.add_mappings_batch(mappings)
        print(f"✅ 已建立 {len(mappings)} 個 chunk ID 映射")
    
    def clear(self):
        """清空映射表"""
        self.mapping = {}
        if os.path.exists(self.mapping_file):
            os.remove(self.mapping_file)
    
    def __len__(self) -> int:
        """返回映射數量"""
        return len(self.mapping)
    
    def __repr__(self) -> str:
        return f"ChunkIDMapper(storage_dir={self.storage_dir}, mappings={len(self.mapping)})"
