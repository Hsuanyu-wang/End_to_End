"""
統一的 Schema Cache 管理系統

支援所有 ontology learning 方法的 schema 結果快取，
確保相同語料和配置下可以重複使用 schema，加速實驗流程。
"""

import os
import json
import hashlib
import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class SchemaCacheKey:
    """Schema Cache 鍵值"""
    method: str  # lightrag_default, iterative_evolution, llm_dynamic, autoschema, dynamic_schema
    data_type: str  # DI, GEN
    data_mode: str  # natural_text, structured, etc.
    text_corpus_hash: str  # 文本語料的 hash（確保一致性）
    config_hash: str  # 方法特定配置的 hash
    
    def to_filename(self) -> str:
        """生成檔案名稱"""
        return f"{self.method}_{self.data_type}_{self.data_mode}_{self.text_corpus_hash[:8]}_{self.config_hash[:8]}.json"


class SchemaCacheManager:
    """統一的 Schema Cache 管理器"""
    
    def __init__(self, cache_root: str = "/home/End_to_End_RAG/storage/schema_cache"):
        """
        初始化 Schema Cache 管理器
        
        Args:
            cache_root: 快取根目錄路徑
        """
        self.cache_root = cache_root
        os.makedirs(cache_root, exist_ok=True)
        print(f"📂 [Schema Cache] 初始化快取管理器: {cache_root}")
        
    def _get_cache_path(self, method: str) -> str:
        """根據方法名稱取得子目錄路徑"""
        method_dir = os.path.join(self.cache_root, method)
        os.makedirs(method_dir, exist_ok=True)
        return method_dir
        
    def compute_corpus_hash(self, text_corpus: list) -> str:
        """
        計算文本語料的 hash
        
        Args:
            text_corpus: 文本語料列表
            
        Returns:
            MD5 hash 字串
        """
        if not text_corpus:
            return "empty"
            
        # 提取文本內容
        texts = []
        for doc in text_corpus:
            if hasattr(doc, 'text'):
                texts.append(doc.text)
            elif hasattr(doc, 'get_content'):
                texts.append(doc.get_content())
            else:
                texts.append(str(doc))
        
        corpus_str = "".join(texts)
        return hashlib.md5(corpus_str.encode('utf-8')).hexdigest()
        
    def compute_config_hash(self, config: Dict) -> str:
        """
        計算配置的 hash
        
        Args:
            config: 配置字典
            
        Returns:
            MD5 hash 字串
        """
        if not config:
            return "default"
            
        # 排序 key 確保一致性
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
        
    def load(self, cache_key: SchemaCacheKey) -> Optional[Dict[str, Any]]:
        """
        載入 cached schema
        
        Args:
            cache_key: Cache 鍵值
            
        Returns:
            Schema 字典，若不存在則返回 None
        """
        cache_dir = self._get_cache_path(cache_key.method)
        cache_file = os.path.join(cache_dir, cache_key.to_filename())
        
        if os.path.exists(cache_file):
            print(f"✅ [Schema Cache] 載入快取: {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  [Schema Cache] 載入快取失敗: {e}")
                return None
        
        print(f"📭 [Schema Cache] 快取不存在: {cache_file}")
        return None
        
    def save(self, cache_key: SchemaCacheKey, schema: Dict[str, Any]) -> None:
        """
        儲存 schema 到快取
        
        Args:
            cache_key: Cache 鍵值
            schema: Schema 字典
        """
        cache_dir = self._get_cache_path(cache_key.method)
        cache_file = os.path.join(cache_dir, cache_key.to_filename())
        
        # 加入 metadata
        schema_with_meta = {
            "metadata": asdict(cache_key),
            "schema": schema,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(schema_with_meta, f, ensure_ascii=False, indent=2)
            print(f"💾 [Schema Cache] 儲存快取: {cache_file}")
        except Exception as e:
            print(f"⚠️  [Schema Cache] 儲存快取失敗: {e}")
        
    def list_caches(self, method: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        列出所有快取檔案
        
        Args:
            method: 可選的方法名稱，若指定則只列出該方法的快取
            
        Returns:
            [(method, filename), ...] 列表
        """
        if method:
            cache_dir = self._get_cache_path(method)
            if not os.path.exists(cache_dir):
                return []
            files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
            return [(method, f) for f in files]
        else:
            all_caches = []
            if not os.path.exists(self.cache_root):
                return []
                
            for method_dir in os.listdir(self.cache_root):
                method_path = os.path.join(self.cache_root, method_dir)
                if os.path.isdir(method_path):
                    files = [f for f in os.listdir(method_path) if f.endswith('.json')]
                    all_caches.extend([(method_dir, f) for f in files])
            return all_caches
    
    def clear_cache(self, method: Optional[str] = None) -> int:
        """
        清理快取
        
        Args:
            method: 可選的方法名稱，若指定則只清理該方法的快取
            
        Returns:
            清理的檔案數量
        """
        count = 0
        if method:
            cache_dir = self._get_cache_path(method)
            if os.path.exists(cache_dir):
                for f in os.listdir(cache_dir):
                    if f.endswith('.json'):
                        os.remove(os.path.join(cache_dir, f))
                        count += 1
        else:
            if os.path.exists(self.cache_root):
                for method_dir in os.listdir(self.cache_root):
                    method_path = os.path.join(self.cache_root, method_dir)
                    if os.path.isdir(method_path):
                        for f in os.listdir(method_path):
                            if f.endswith('.json'):
                                os.remove(os.path.join(method_path, f))
                                count += 1
        return count
    
    def get_cache_info(self, method: Optional[str] = None) -> Dict[str, Any]:
        """
        取得快取資訊統計
        
        Args:
            method: 可選的方法名稱
            
        Returns:
            快取資訊字典
        """
        caches = self.list_caches(method)
        
        info = {
            "total_caches": len(caches),
            "methods": {}
        }
        
        for method_name, filename in caches:
            if method_name not in info["methods"]:
                info["methods"][method_name] = {
                    "count": 0,
                    "files": []
                }
            info["methods"][method_name]["count"] += 1
            info["methods"][method_name]["files"].append(filename)
        
        return info
