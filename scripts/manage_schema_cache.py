#!/usr/bin/env python3
"""
Schema Cache 管理工具

提供命令列介面來管理 Schema Cache：
- 列出所有快取
- 清理快取
- 匯出快取報告
"""

import argparse
import json
import os
import sys

# 加入專案路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.schema.schema_cache import SchemaCacheManager


def list_caches(manager: SchemaCacheManager, method: str = None):
    """列出所有快取"""
    caches = manager.list_caches(method)
    
    if not caches:
        print("📭 沒有找到任何快取")
        return
    
    print(f"📋 找到 {len(caches)} 個快取檔案")
    print()
    
    # 按方法分組顯示
    methods = {}
    for method_name, filename in caches:
        if method_name not in methods:
            methods[method_name] = []
        methods[method_name].append(filename)
    
    for method_name, files in sorted(methods.items()):
        print(f"📦 {method_name}: {len(files)} 個快取")
        for f in files:
            cache_file = os.path.join(manager._get_cache_path(method_name), f)
            size = os.path.getsize(cache_file) / 1024  # KB
            print(f"  - {f} ({size:.1f} KB)")
        print()


def clean_caches(manager: SchemaCacheManager, method: str = None):
    """清理快取"""
    if method:
        count = manager.clear_cache(method)
        print(f"🗑️  已清理 {method} 的 {count} 個快取檔案")
    else:
        confirm = input("⚠️  確定要清理所有快取嗎？[y/N] ")
        if confirm.lower() == 'y':
            count = manager.clear_cache()
            print(f"🗑️  已清理所有 {count} 個快取檔案")
        else:
            print("❌ 已取消")


def export_report(manager: SchemaCacheManager, output: str):
    """匯出快取報告"""
    report = {"caches": []}
    
    for method_name, filename in manager.list_caches():
        cache_file = os.path.join(manager._get_cache_path(method_name), filename)
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            schema = data.get("schema", {})
            report["caches"].append({
                "method": method_name,
                "filename": filename,
                "metadata": data.get("metadata", {}),
                "entity_count": len(schema.get("entities", [])),
                "relation_count": len(schema.get("relations", [])),
                "timestamp": data.get("timestamp", ""),
                "file_size_kb": os.path.getsize(cache_file) / 1024
            })
        except Exception as e:
            print(f"⚠️  無法讀取 {cache_file}: {e}")
    
    # 加入統計資訊
    report["summary"] = {
        "total_caches": len(report["caches"]),
        "methods": list(set(c["method"] for c in report["caches"])),
        "total_entities": sum(c["entity_count"] for c in report["caches"]),
        "total_relations": sum(c["relation_count"] for c in report["caches"])
    }
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 已匯出報告至 {output}")
    print(f"📊 統計: {report['summary']['total_caches']} 個快取, "
          f"{report['summary']['total_entities']} 個實體類型, "
          f"{report['summary']['total_relations']} 個關係類型")


def show_info(manager: SchemaCacheManager, method: str = None):
    """顯示快取資訊統計"""
    info = manager.get_cache_info(method)
    
    print(f"📊 Schema Cache 統計資訊")
    print(f"   總快取數: {info['total_caches']}")
    print()
    
    for method_name, method_info in sorted(info['methods'].items()):
        print(f"📦 {method_name}:")
        print(f"   快取數量: {method_info['count']}")
        print(f"   檔案: {', '.join(method_info['files'][:3])}")
        if len(method_info['files']) > 3:
            print(f"         ... 還有 {len(method_info['files']) - 3} 個")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="管理 Schema Cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 列出所有快取
  python manage_schema_cache.py --list
  
  # 列出特定方法的快取
  python manage_schema_cache.py --list --method iterative_evolution
  
  # 清理特定方法的快取
  python manage_schema_cache.py --clean --method autoschema
  
  # 匯出快取報告
  python manage_schema_cache.py --export --output schema_report.json
  
  # 顯示快取統計
  python manage_schema_cache.py --info
        """
    )
    
    parser.add_argument("--list", action="store_true", help="列出所有快取")
    parser.add_argument("--clean", action="store_true", help="清理快取")
    parser.add_argument("--export", action="store_true", help="匯出快取報告")
    parser.add_argument("--info", action="store_true", help="顯示快取統計資訊")
    parser.add_argument("--method", type=str, help="指定方法名稱")
    parser.add_argument("--output", type=str, default="schema_cache_report.json", 
                       help="輸出檔案路徑 (預設: schema_cache_report.json)")
    parser.add_argument("--cache-root", type=str, 
                       default="/home/End_to_End_RAG/storage/schema_cache",
                       help="快取根目錄 (預設: /home/End_to_End_RAG/storage/schema_cache)")
    
    args = parser.parse_args()
    
    # 初始化管理器
    manager = SchemaCacheManager(cache_root=args.cache_root)
    
    # 執行操作
    if args.list:
        list_caches(manager, args.method)
    elif args.clean:
        clean_caches(manager, args.method)
    elif args.export:
        export_report(manager, args.output)
    elif args.info:
        show_info(manager, args.method)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
